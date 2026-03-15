#!/usr/bin/env python3
"""
LSP MCP Server for Claude Code (Container-side)
Wraps pre-installed LSP language servers (pyright, typescript-language-server, gopls,
rust-analyzer, clangd) as MCP tools. Provides diagnostics, hover, go-to-definition,
references, symbols, rename preview, and completions.

Includes an HTTP sidecar on a Unix socket for PostToolUse hook integration.
"""

import asyncio
import json
import os
import subprocess
import sys
import threading
import socket
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Sequence

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Initialize MCP server
app = Server("lsp-bridge")

CURRENT_PROJECT = os.environ.get("PROJECT_NAME", "unknown")
WORKSPACE_PATH = os.environ.get("WORKSPACE_PATH", f"/workspaces/{CURRENT_PROJECT}")

# Server binary mapping
LSP_BINARIES = {
    "python": {"cmd": "pyright-langserver", "args": ["--stdio"]},
    "typescript": {"cmd": "typescript-language-server", "args": ["--stdio"]},
    "go": {"cmd": "gopls", "args": ["serve"]},
    "rust": {"cmd": "rust-analyzer", "args": []},
    "c_cpp": {"cmd": "clangd", "args": ["--log=error"]},
}

# Extension to language mapping
EXTENSION_MAP = {
    ".py": "python",
    ".js": "typescript",
    ".jsx": "typescript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".c": "c_cpp",
    ".h": "c_cpp",
    ".cpp": "c_cpp",
    ".hpp": "c_cpp",
    ".cc": "c_cpp",
}


def detect_language(file_path: str) -> str | None:
    """Detect language from file extension."""
    for ext, lang in EXTENSION_MAP.items():
        if file_path.endswith(ext):
            return lang
    return None


def file_uri(path: str) -> str:
    """Convert file path to URI."""
    if not path.startswith("/"):
        path = os.path.abspath(path)
    return f"file://{path}"


class LspManager:
    """Manages LSP server processes with lazy start and crash recovery."""

    def __init__(self):
        self._servers: dict[str, subprocess.Popen] = {}
        self._initialized: dict[str, bool] = {}
        self._open_files: dict[str, set[str]] = {}  # lang -> set of open file URIs
        self._request_id: int = 0
        self._lock = threading.Lock()

    def _next_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _is_alive(self, lang: str) -> bool:
        proc = self._servers.get(lang)
        return proc is not None and proc.poll() is None

    def get_server(self, language: str) -> subprocess.Popen | None:
        """Get or start an LSP server for the given language."""
        with self._lock:
            if self._is_alive(language):
                return self._servers[language]

            # Clean up dead process
            if language in self._servers:
                try:
                    self._servers[language].kill()
                except Exception:
                    pass
                del self._servers[language]
                self._initialized.pop(language, None)
                self._open_files.pop(language, None)

            # Start new server
            binary_info = LSP_BINARIES.get(language)
            if not binary_info:
                return None

            cmd = binary_info["cmd"]
            args = binary_info["args"]

            # Check if binary exists
            try:
                subprocess.run(["which", cmd], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                return None

            try:
                proc = subprocess.Popen(
                    [cmd] + args,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                self._servers[language] = proc
                self._initialized[language] = False
                self._open_files[language] = set()

                # Initialize the server
                self._do_initialize(language)
                return proc
            except Exception:
                return None

    def _do_initialize(self, language: str):
        """Send initialize request to LSP server."""
        proc = self._servers.get(language)
        if not proc:
            return

        init_params = {
            "processId": os.getpid(),
            "rootUri": file_uri(WORKSPACE_PATH),
            "capabilities": {
                "textDocument": {
                    "publishDiagnostics": {"relatedInformation": True},
                    "hover": {"contentFormat": ["plaintext", "markdown"]},
                    "completion": {
                        "completionItem": {"snippetSupport": False}
                    },
                    "definition": {},
                    "references": {},
                    "documentSymbol": {
                        "hierarchicalDocumentSymbolSupport": True
                    },
                    "rename": {"prepareSupport": True},
                }
            },
            "workspaceFolders": [
                {"uri": file_uri(WORKSPACE_PATH), "name": CURRENT_PROJECT}
            ],
        }

        response = self.send_request(language, "initialize", init_params)
        if response and "result" in response:
            self._initialized[language] = True
            self.send_notification(language, "initialized", {})

    def send_request(self, lang: str, method: str, params: dict) -> dict | None:
        """Send a JSON-RPC request and wait for response."""
        proc = self._servers.get(lang)
        if not proc or proc.poll() is not None:
            return None

        req_id = self._next_id()
        message = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        }

        try:
            content = json.dumps(message)
            header = f"Content-Length: {len(content)}\r\n\r\n"
            proc.stdin.write(header.encode("utf-8"))
            proc.stdin.write(content.encode("utf-8"))
            proc.stdin.flush()

            # Read response with timeout
            return self._read_response(proc, req_id, timeout=30)
        except Exception:
            return None

    def send_notification(self, lang: str, method: str, params: dict):
        """Send a JSON-RPC notification (no response expected)."""
        proc = self._servers.get(lang)
        if not proc or proc.poll() is not None:
            return

        message = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        try:
            content = json.dumps(message)
            header = f"Content-Length: {len(content)}\r\n\r\n"
            proc.stdin.write(header.encode("utf-8"))
            proc.stdin.write(content.encode("utf-8"))
            proc.stdin.flush()
        except Exception:
            pass

    def _read_response(self, proc: subprocess.Popen, req_id: int, timeout: float = 30) -> dict | None:
        """Read JSON-RPC response matching the request ID."""
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                # Read Content-Length header
                header_line = b""
                while True:
                    byte = proc.stdout.read(1)
                    if not byte:
                        return None
                    header_line += byte
                    if header_line.endswith(b"\r\n\r\n"):
                        break
                    if header_line.endswith(b"\n\n"):
                        break

                # Parse content length
                content_length = 0
                for line in header_line.decode("utf-8", errors="replace").split("\r\n"):
                    if line.lower().startswith("content-length:"):
                        content_length = int(line.split(":")[1].strip())
                        break

                if content_length == 0:
                    continue

                # Read content
                content = proc.stdout.read(content_length)
                if not content:
                    return None

                msg = json.loads(content.decode("utf-8"))

                # Check if this is our response
                if msg.get("id") == req_id:
                    return msg

                # Skip notifications and other responses
                continue

            except Exception:
                return None

        return None

    def ensure_file_open(self, lang: str, file_path: str):
        """Ensure a file is opened in the LSP server via textDocument/didOpen."""
        uri = file_uri(file_path)
        if lang not in self._open_files:
            self._open_files[lang] = set()

        if uri in self._open_files[lang]:
            # Send didChange to refresh content
            try:
                text = open(file_path, "r", errors="replace").read()
            except Exception:
                return
            self.send_notification(lang, "textDocument/didChange", {
                "textDocument": {"uri": uri, "version": int(time.time())},
                "contentChanges": [{"text": text}],
            })
            return

        # Open the file
        try:
            text = open(file_path, "r", errors="replace").read()
        except Exception:
            return

        lang_id_map = {
            "python": "python",
            "typescript": "typescriptreact" if file_path.endswith((".tsx", ".jsx")) else ("typescript" if file_path.endswith(".ts") else "javascript"),
            "go": "go",
            "rust": "rust",
            "c_cpp": "cpp" if file_path.endswith((".cpp", ".hpp", ".cc")) else "c",
        }

        self.send_notification(lang, "textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": lang_id_map.get(lang, lang),
                "version": 1,
                "text": text,
            }
        })
        self._open_files[lang].add(uri)

    def shutdown_all(self):
        """Shutdown all running LSP servers."""
        for lang, proc in list(self._servers.items()):
            try:
                self.send_request(lang, "shutdown", {})
                self.send_notification(lang, "exit", {})
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._servers.clear()
        self._initialized.clear()
        self._open_files.clear()


# Global LSP manager
lsp_manager = LspManager()


def get_diagnostics(file_path: str) -> list[dict]:
    """Get diagnostics for a file. Waits for LSP to analyze (up to 15s)."""
    import select as _select

    lang = detect_language(file_path)
    if not lang:
        return []

    proc = lsp_manager.get_server(lang)
    if not proc:
        return []

    lsp_manager.ensure_file_open(lang, file_path)

    # For diagnostics, we need to wait for publishDiagnostics notification.
    # LSP servers (especially pyright) send several window/logMessage
    # notifications before diagnostics. We must keep reading through
    # those until we get publishDiagnostics for our file or hit the deadline.
    uri = file_uri(file_path)
    diagnostics = []

    # Wait up to 15s total (cold start can take 5-10s)
    deadline = time.time() + 15
    empty_polls = 0

    while time.time() < deadline:
        try:
            readable, _, _ = _select.select([proc.stdout], [], [], 1.0)
            if not readable:
                empty_polls += 1
                # Allow several empty polls — server may be analyzing
                if empty_polls > 10:
                    break
                continue
            empty_polls = 0

            # Read header
            header_line = b""
            while True:
                byte = proc.stdout.read(1)
                if not byte:
                    break
                header_line += byte
                if header_line.endswith(b"\r\n\r\n") or header_line.endswith(b"\n\n"):
                    break

            content_length = 0
            for line in header_line.decode("utf-8", errors="replace").split("\r\n"):
                if line.lower().startswith("content-length:"):
                    content_length = int(line.split(":")[1].strip())
                    break

            if content_length == 0:
                continue

            content = proc.stdout.read(content_length)
            msg = json.loads(content.decode("utf-8"))

            if msg.get("method") == "textDocument/publishDiagnostics":
                params = msg.get("params", {})
                if params.get("uri") == uri:
                    for d in params.get("diagnostics", []):
                        diagnostics.append({
                            "line": d.get("range", {}).get("start", {}).get("line", 0) + 1,
                            "character": d.get("range", {}).get("start", {}).get("character", 0),
                            "severity": d.get("severity", 3),
                            "message": d.get("message", ""),
                            "source": d.get("source", ""),
                        })
                    break  # Got diagnostics for our file
            # Other notifications (window/logMessage etc.) — keep reading
        except Exception:
            break

    return diagnostics


def do_lsp_request(file_path: str, method: str, line: int = 0, character: int = 0, extra_params: dict | None = None) -> dict | None:
    """Generic LSP request for a text document position."""
    lang = detect_language(file_path)
    if not lang:
        return {"error": f"Unsupported file type: {file_path}"}

    proc = lsp_manager.get_server(lang)
    if not proc:
        return {"error": f"LSP server not available for {lang}"}

    lsp_manager.ensure_file_open(lang, file_path)
    time.sleep(0.5)  # Brief pause for analysis

    params = {
        "textDocument": {"uri": file_uri(file_path)},
        "position": {"line": line, "character": character},
    }
    if extra_params:
        params.update(extra_params)

    response = lsp_manager.send_request(lang, method, params)
    return response


# ─── MCP Tool Definitions ────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available LSP tools."""
    return [
        Tool(
            name="lsp_diagnostics",
            description=(
                "Get type errors, warnings, and diagnostics for a source file. "
                "Supports Python (.py), TypeScript/JavaScript (.ts/.tsx/.js/.jsx), "
                "Go (.go), Rust (.rs), and C/C++ (.c/.cpp/.h/.hpp). "
                "Returns severity, line number, and error message for each issue."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the source file"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="lsp_hover",
            description=(
                "Get type information and documentation for a symbol at a specific position. "
                "Returns type signature, docstring, and other hover info."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the source file"},
                    "line": {"type": "number", "description": "0-based line number"},
                    "character": {"type": "number", "description": "0-based character offset"},
                },
                "required": ["file_path", "line", "character"]
            }
        ),
        Tool(
            name="lsp_definition",
            description=(
                "Go to the definition of a symbol at a specific position. "
                "Returns the file path and line where the symbol is defined."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the source file"},
                    "line": {"type": "number", "description": "0-based line number"},
                    "character": {"type": "number", "description": "0-based character offset"},
                },
                "required": ["file_path", "line", "character"]
            }
        ),
        Tool(
            name="lsp_references",
            description=(
                "Find all references to a symbol at a specific position. "
                "Returns file paths and line numbers of all usages."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the source file"},
                    "line": {"type": "number", "description": "0-based line number"},
                    "character": {"type": "number", "description": "0-based character offset"},
                },
                "required": ["file_path", "line", "character"]
            }
        ),
        Tool(
            name="lsp_document_symbols",
            description=(
                "List all symbols (functions, classes, variables) in a source file. "
                "Returns symbol names, kinds, and line ranges."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the source file"},
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="lsp_rename_preview",
            description=(
                "Preview what a rename operation would change. "
                "Shows all files and locations that would be modified."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the source file"},
                    "line": {"type": "number", "description": "0-based line number"},
                    "character": {"type": "number", "description": "0-based character offset"},
                    "new_name": {"type": "string", "description": "The new name for the symbol"},
                },
                "required": ["file_path", "line", "character", "new_name"]
            }
        ),
        Tool(
            name="lsp_completions",
            description=(
                "Get code completion suggestions at a specific position. "
                "Returns completion items with labels and types."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the source file"},
                    "line": {"type": "number", "description": "0-based line number"},
                    "character": {"type": "number", "description": "0-based character offset"},
                },
                "required": ["file_path", "line", "character"]
            }
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
    """Execute LSP tools."""

    if name == "lsp_diagnostics":
        return await handle_diagnostics(arguments)
    elif name == "lsp_hover":
        return await handle_hover(arguments)
    elif name == "lsp_definition":
        return await handle_definition(arguments)
    elif name == "lsp_references":
        return await handle_references(arguments)
    elif name == "lsp_document_symbols":
        return await handle_document_symbols(arguments)
    elif name == "lsp_rename_preview":
        return await handle_rename_preview(arguments)
    elif name == "lsp_completions":
        return await handle_completions(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")


async def handle_diagnostics(args: dict) -> Sequence[TextContent]:
    file_path = args.get("file_path", "")
    if not file_path or not os.path.isfile(file_path):
        return [TextContent(type="text", text=f"File not found: {file_path}")]

    diagnostics = get_diagnostics(file_path)

    if not diagnostics:
        return [TextContent(type="text", text=f"No diagnostics for {os.path.basename(file_path)} (clean)")]

    severity_map = {1: "ERROR", 2: "WARN", 3: "INFO", 4: "HINT"}
    lines = [f"# Diagnostics for {os.path.basename(file_path)}", f"Found {len(diagnostics)} issue(s)\n"]

    for d in diagnostics:
        sev = severity_map.get(d["severity"], "INFO")
        source = f" ({d['source']})" if d.get("source") else ""
        lines.append(f"  {sev} line {d['line']}: {d['message']}{source}")

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_hover(args: dict) -> Sequence[TextContent]:
    file_path = args["file_path"]
    response = do_lsp_request(file_path, "textDocument/hover", args["line"], args["character"])

    if not response or "error" in response:
        error = response.get("error", "No response") if response else "No response"
        return [TextContent(type="text", text=f"Hover failed: {error}")]

    result = response.get("result")
    if not result:
        return [TextContent(type="text", text="No hover information available at this position")]

    contents = result.get("contents", "")
    if isinstance(contents, dict):
        text = contents.get("value", str(contents))
    elif isinstance(contents, list):
        text = "\n".join(c.get("value", str(c)) if isinstance(c, dict) else str(c) for c in contents)
    else:
        text = str(contents)

    return [TextContent(type="text", text=f"# Hover Info\n\n{text}")]


async def handle_definition(args: dict) -> Sequence[TextContent]:
    file_path = args["file_path"]
    response = do_lsp_request(file_path, "textDocument/definition", args["line"], args["character"])

    if not response or "error" in response:
        error = response.get("error", "No response") if response else "No response"
        return [TextContent(type="text", text=f"Definition lookup failed: {error}")]

    result = response.get("result")
    if not result:
        return [TextContent(type="text", text="No definition found")]

    # Result can be Location, Location[], or LocationLink[]
    locations = result if isinstance(result, list) else [result]

    lines = ["# Definition Location(s)\n"]
    for loc in locations:
        uri = loc.get("uri", loc.get("targetUri", ""))
        path = uri.replace("file://", "")
        range_info = loc.get("range", loc.get("targetRange", {}))
        start = range_info.get("start", {})
        line_num = start.get("line", 0) + 1
        lines.append(f"- {path}:{line_num}")

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_references(args: dict) -> Sequence[TextContent]:
    file_path = args["file_path"]
    response = do_lsp_request(
        file_path, "textDocument/references", args["line"], args["character"],
        extra_params={"context": {"includeDeclaration": True}}
    )

    if not response or "error" in response:
        error = response.get("error", "No response") if response else "No response"
        return [TextContent(type="text", text=f"References lookup failed: {error}")]

    result = response.get("result", [])
    if not result:
        return [TextContent(type="text", text="No references found")]

    lines = [f"# References ({len(result)} found)\n"]
    for ref in result[:50]:  # Cap at 50
        uri = ref.get("uri", "")
        path = uri.replace("file://", "")
        start = ref.get("range", {}).get("start", {})
        line_num = start.get("line", 0) + 1
        lines.append(f"- {path}:{line_num}")

    if len(result) > 50:
        lines.append(f"\n... and {len(result) - 50} more")

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_document_symbols(args: dict) -> Sequence[TextContent]:
    file_path = args["file_path"]
    lang = detect_language(file_path)
    if not lang:
        return [TextContent(type="text", text=f"Unsupported file type: {file_path}")]

    proc = lsp_manager.get_server(lang)
    if not proc:
        return [TextContent(type="text", text=f"LSP server not available for {lang}")]

    lsp_manager.ensure_file_open(lang, file_path)
    time.sleep(0.5)

    response = lsp_manager.send_request(lang, "textDocument/documentSymbol", {
        "textDocument": {"uri": file_uri(file_path)}
    })

    if not response or "error" in response:
        error = response.get("error", "No response") if response else "No response"
        return [TextContent(type="text", text=f"Symbol lookup failed: {error}")]

    result = response.get("result", [])
    if not result:
        return [TextContent(type="text", text="No symbols found")]

    symbol_kinds = {
        1: "File", 2: "Module", 3: "Namespace", 4: "Package", 5: "Class",
        6: "Method", 7: "Property", 8: "Field", 9: "Constructor", 10: "Enum",
        11: "Interface", 12: "Function", 13: "Variable", 14: "Constant",
        15: "String", 16: "Number", 17: "Boolean", 18: "Array", 19: "Object",
        20: "Key", 21: "Null", 22: "EnumMember", 23: "Struct", 24: "Event",
        25: "Operator", 26: "TypeParameter",
    }

    lines = [f"# Symbols in {os.path.basename(file_path)}\n"]

    def format_symbols(symbols, indent=0):
        for sym in symbols:
            kind_num = sym.get("kind", 0)
            kind = symbol_kinds.get(kind_num, f"Kind({kind_num})")
            name = sym.get("name", "?")
            start_line = sym.get("range", sym.get("location", {}).get("range", {})).get("start", {}).get("line", 0) + 1
            prefix = "  " * indent
            lines.append(f"{prefix}- [{kind}] {name} (line {start_line})")
            # Recurse for DocumentSymbol children
            children = sym.get("children", [])
            if children:
                format_symbols(children, indent + 1)

    format_symbols(result)
    return [TextContent(type="text", text="\n".join(lines))]


async def handle_rename_preview(args: dict) -> Sequence[TextContent]:
    file_path = args["file_path"]
    new_name = args["new_name"]
    response = do_lsp_request(
        file_path, "textDocument/rename", args["line"], args["character"],
        extra_params={"newName": new_name}
    )

    if not response or "error" in response:
        error = response.get("error", "No response") if response else "No response"
        return [TextContent(type="text", text=f"Rename preview failed: {error}")]

    result = response.get("result")
    if not result:
        return [TextContent(type="text", text="Rename not supported at this position")]

    changes = result.get("changes", {})
    document_changes = result.get("documentChanges", [])

    lines = [f"# Rename Preview: -> '{new_name}'\n"]

    if changes:
        total = 0
        for uri, edits in changes.items():
            path = uri.replace("file://", "")
            lines.append(f"\n## {path} ({len(edits)} changes)")
            for edit in edits:
                start = edit.get("range", {}).get("start", {})
                lines.append(f"  - line {start.get('line', 0) + 1}")
                total += 1
        lines.insert(1, f"Total: {total} change(s) across {len(changes)} file(s)")
    elif document_changes:
        total = 0
        for doc_change in document_changes:
            text_doc = doc_change.get("textDocument", {})
            uri = text_doc.get("uri", "")
            path = uri.replace("file://", "")
            edits = doc_change.get("edits", [])
            lines.append(f"\n## {path} ({len(edits)} changes)")
            for edit in edits:
                start = edit.get("range", {}).get("start", {})
                lines.append(f"  - line {start.get('line', 0) + 1}")
                total += 1
        lines.insert(1, f"Total: {total} change(s) across {len(document_changes)} file(s)")
    else:
        lines.append("No changes would be made")

    return [TextContent(type="text", text="\n".join(lines))]


async def handle_completions(args: dict) -> Sequence[TextContent]:
    file_path = args["file_path"]
    response = do_lsp_request(file_path, "textDocument/completion", args["line"], args["character"])

    if not response or "error" in response:
        error = response.get("error", "No response") if response else "No response"
        return [TextContent(type="text", text=f"Completions failed: {error}")]

    result = response.get("result")
    if not result:
        return [TextContent(type="text", text="No completions available")]

    items = result if isinstance(result, list) else result.get("items", [])
    if not items:
        return [TextContent(type="text", text="No completions available")]

    completion_kinds = {
        1: "Text", 2: "Method", 3: "Function", 4: "Constructor", 5: "Field",
        6: "Variable", 7: "Class", 8: "Interface", 9: "Module", 10: "Property",
        11: "Unit", 12: "Value", 13: "Enum", 14: "Keyword", 15: "Snippet",
        16: "Color", 17: "File", 18: "Reference", 19: "Folder",
        20: "EnumMember", 21: "Constant", 22: "Struct", 23: "Event",
        24: "Operator", 25: "TypeParameter",
    }

    lines = [f"# Completions ({min(len(items), 20)} of {len(items)})\n"]
    for item in items[:20]:
        label = item.get("label", "?")
        kind = completion_kinds.get(item.get("kind", 0), "")
        detail = item.get("detail", "")
        kind_str = f" [{kind}]" if kind else ""
        detail_str = f" - {detail}" if detail else ""
        lines.append(f"- {label}{kind_str}{detail_str}")

    return [TextContent(type="text", text="\n".join(lines))]


# ─── HTTP Sidecar for PostToolUse Hooks ──────────────────────

SOCKET_PATH = "/tmp/lsp-mcp.sock"


class DiagnosticsHandler(BaseHTTPRequestHandler):
    """HTTP handler for the diagnostics sidecar."""

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def do_POST(self):
        if self.path != "/diagnostics":
            self.send_response(404)
            self.end_headers()
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length else {}
            file_path = body.get("file_path", "")

            if not file_path or not os.path.isfile(file_path):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b"[]")
                return

            diagnostics = get_diagnostics(file_path)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(diagnostics).encode("utf-8"))

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))


class UnixSocketHTTPServer(HTTPServer):
    """HTTP server listening on a Unix domain socket."""

    address_family = socket.AF_UNIX

    def server_bind(self):
        # Remove old socket file if exists
        if os.path.exists(self.server_address):
            os.unlink(self.server_address)
        HTTPServer.server_bind(self)
        os.chmod(self.server_address, 0o666)


def start_sidecar():
    """Start the HTTP sidecar in a background thread."""
    try:
        server = UnixSocketHTTPServer(SOCKET_PATH, DiagnosticsHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
    except Exception:
        pass  # Sidecar is optional — MCP tools still work


# ─── Main ────────────────────────────────────────────────────

async def main():
    """Run the MCP server with HTTP sidecar."""
    start_sidecar()

    try:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())
    finally:
        lsp_manager.shutdown_all()
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)


if __name__ == "__main__":
    asyncio.run(main())
