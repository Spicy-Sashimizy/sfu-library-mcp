#!/usr/bin/env python3
"""
Ollama API proxy for Pommel compatibility.

Pommel calls Ollama with batch embedding requests that may exceed context limits.
This proxy:
1. Splits large batches into smaller chunks
2. Transforms request/response formats
3. Combines results transparently
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request
import urllib.parse
import json
import sys


class OllamaProxyHandler(BaseHTTPRequestHandler):
    """Proxy handler that transforms Ollama API requests."""

    # The upstream Ollama server (use host.docker.internal from container)
    OLLAMA_UPSTREAM = "http://host.docker.internal:11434"

    def log_message(self, format, *args):
        """Log messages for debugging."""
        print(f"[PROXY] {format % args}", file=sys.stderr)

    def _forward_batch_request(self, url: str, req_data: dict, headers: dict, max_batch_size: int):
        """Split large batch request into smaller chunks and combine results.

        Args:
            url: Ollama API URL
            req_data: Request data with 'input' array
            headers: HTTP headers
            max_batch_size: Maximum chunks per batch

        Returns:
            Tuple of (status_code, combined_body, response_headers)
        """
        input_array = req_data.get('input', [])
        model = req_data.get('model', 'nomic-embed-text')

        print(f"[PROXY] Splitting {len(input_array)} chunks into batches of {max_batch_size}", file=sys.stderr)

        all_embeddings = []
        total_batches = (len(input_array) + max_batch_size - 1) // max_batch_size

        for i in range(0, len(input_array), max_batch_size):
            batch_num = i // max_batch_size + 1
            batch_chunks = input_array[i:i + max_batch_size]

            print(f"[PROXY] Batch {batch_num}/{total_batches}: {len(batch_chunks)} chunks", file=sys.stderr)

            # Create batch request - use 'input' for Ollama's current API
            batch_data = {
                'model': model,
                'input': batch_chunks
            }

            json_body = json.dumps(batch_data).encode('utf-8')
            req = urllib.request.Request(url, data=json_body, headers=headers, method='POST')

            try:
                with urllib.request.urlopen(req, timeout=60) as response:
                    if response.getcode() != 200:
                        error_body = response.read().decode()
                        print(f"[PROXY] Batch {batch_num} failed: HTTP {response.getcode()} - {error_body}", file=sys.stderr)
                        return response.getcode(), error_body.encode(), response.headers

                    body = response.read()
                    resp_data = json.loads(body.decode())

                    # Extract embeddings from response
                    embeddings = resp_data.get('embeddings', resp_data.get('embedding', []))

                    # Handle different response formats
                    if isinstance(embeddings, list) and len(embeddings) > 0:
                        if isinstance(embeddings[0], list):
                            # Already an array of embeddings
                            all_embeddings.extend(embeddings)
                        else:
                            # Single embedding, wrap in array
                            all_embeddings.append(embeddings)

                    print(f"[PROXY] Batch {batch_num} complete: got {len(embeddings) if isinstance(embeddings, list) else 1} embeddings", file=sys.stderr)

            except urllib.error.HTTPError as e:
                error_msg = f"Batch {batch_num} failed: HTTP {e.code} - {e.read().decode()}"
                print(f"[PROXY] {error_msg}", file=sys.stderr)
                return e.code, json.dumps({'error': error_msg}).encode(), {}
            except Exception as e:
                error_msg = f"Batch {batch_num} failed: {str(e)}"
                print(f"[PROXY] {error_msg}", file=sys.stderr)
                return 500, json.dumps({'error': error_msg}).encode(), {}

        # Combine all embeddings into Pommel's expected format
        combined_response = {
            'embeddings': all_embeddings
        }

        print(f"[PROXY] All batches complete: {len(all_embeddings)} total embeddings", file=sys.stderr)

        return 200, json.dumps(combined_response).encode(), {}

    # Valid path prefixes for Ollama API
    ALLOWED_PATH_PREFIXES = ("/api/", "/v1/", "/")

    def _validate_path(self, path):
        """Validate the request path to prevent SSRF attacks."""
        # Block path traversal attempts
        if ".." in path or "://" in path:
            return False
        # Must start with a known prefix
        if not any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in self.ALLOWED_PATH_PREFIXES):
            return False
        return True

    def _get_upstream_url(self, path):
        """Construct the upstream URL."""
        # Keep the same path - Ollama's current API uses /api/embed
        return f"{self.OLLAMA_UPSTREAM}{path}"

    def _forward_request(self, method, path, data=None):
        """Forward request to upstream Ollama."""
        if not self._validate_path(path):
            print(f"[PROXY] Blocked invalid path: {path}", file=sys.stderr)
            return 403, json.dumps({'error': 'Forbidden: invalid path'}).encode(), {}

        url = self._get_upstream_url(path)

        headers = {}
        if self.headers.get('Content-Type'):
            headers['Content-Type'] = self.headers.get('Content-Type')

        if data:
            # Check for batch requests that need splitting
            try:
                req_data = json.loads(data.decode() if isinstance(data, bytes) else data)
                # Check if input is a batch (array) and needs splitting
                input_data = req_data.get('input')
                if isinstance(input_data, list):
                    batch_size = len(input_data)
                    max_batch_size = 10  # Split batches larger than 10 chunks

                    if batch_size > max_batch_size:
                        print(f"[PROXY] Large batch detected ({batch_size} chunks), splitting into {max_batch_size}-chunk batches", file=sys.stderr)
                        return self._forward_batch_request(url, req_data, headers, max_batch_size)

                print(f"[PROXY] Forwarding to Ollama: {url}, data_len={len(data)}", file=sys.stderr)
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(f"[PROXY] Parse error: {e}", file=sys.stderr)
                pass  # Not JSON, forward as-is

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            # Debug logging for embeddings
            if 'embed' in path and data:
                print(f"[PROXY] Request URL: {url}", file=sys.stderr)
                print(f"[PROXY] Request body: {data.decode() if data else 'None'}", file=sys.stderr)
            with urllib.request.urlopen(req, timeout=30) as response:
                body = response.read()
                if 'embed' in path:
                    print(f"[PROXY] Response status: {response.getcode()}", file=sys.stderr)
                    print(f"[PROXY] Response body: {body.decode()[:200]}", file=sys.stderr)

                    # Transform Ollama response to Pommel's expected format
                    # Ollama returns: {"embedding": [...]} (singular)
                    # Pommel expects: {"embeddings": [[...]]} (plural, array of arrays)
                    try:
                        ollama_resp = json.loads(body.decode())
                        if 'embedding' in ollama_resp and 'embeddings' not in ollama_resp:
                            # Convert {"embedding": [0.1, 0.2, ...]}
                            # to {"embeddings": [[0.1, 0.2, ...]]}
                            embedding = ollama_resp['embedding']
                            if isinstance(embedding, list) and len(embedding) > 0:
                                # If it's a single embedding (list of floats), wrap it in an array
                                if isinstance(embedding[0], (int, float)):
                                    ollama_resp['embeddings'] = [embedding]
                                    del ollama_resp['embedding']
                                    body = json.dumps(ollama_resp).encode()
                                    print(f"[PROXY] Transformed response: embedding->embeddings[{len(embedding)}]", file=sys.stderr)
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        print(f"[PROXY] Response transform error: {e}", file=sys.stderr)

                return response.getcode(), body, response.headers
        except urllib.error.HTTPError as e:
            return e.code, e.read(), e.headers
        except Exception as e:
            return 500, json.dumps({'error': str(e)}).encode(), {}

    def do_GET(self):
        """Handle GET requests."""
        code, body, headers = self._forward_request('GET', self.path)
        self.send_response(code)
        for header, value in headers.items():
            if header.lower() not in ('transfer-encoding', 'connection'):
                self.send_header(header, value)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        """Handle POST requests."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        code, resp_body, headers = self._forward_request('POST', self.path, body)
        # Debug: log request/response for embeddings
        if 'embed' in self.path:
            try:
                req_data = json.loads(body.decode() if isinstance(body, bytes) else body)
                resp_data = json.loads(resp_body.decode() if isinstance(resp_body, bytes) else resp_body)
                print(f"[PROXY] Request: {req_data.get('model', '?')}, input={list(req_data.keys())}", file=sys.stderr)
                print(f"[PROXY] Response: embedding_len={len(resp_data.get('embedding', []))}", file=sys.stderr)
            except:
                pass

        self.send_response(code)
        for header, value in headers.items():
            if header.lower() not in ('transfer-encoding', 'connection'):
                self.send_header(header, value)
        self.end_headers()
        self.wfile.write(resp_body)


def main(port=11434):
    """Start the Ollama proxy server."""
    server_address = ('', port)
    httpd = HTTPServer(server_address, OllamaProxyHandler)
    print(f"Ollama proxy listening on port {port}", file=sys.stderr)
    print(f"Forwarding to {OllamaProxyHandler.OLLAMA_UPSTREAM}", file=sys.stderr)
    httpd.serve_forever()


if __name__ == '__main__':
    main()
