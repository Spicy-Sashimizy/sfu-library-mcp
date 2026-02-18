---
## MANDATORY: GIT WORKFLOW - READ THIS FIRST
---

**THIS IS NON-NEGOTIABLE. BEFORE DOING ANYTHING ELSE, FOLLOW THIS WORKFLOW.**

### BEFORE ANY Edit/Write Tool Call:
```
Step 1: Run `git pull` FIRST (or check if remote exists)
Step 2: Only THEN proceed with edits
```

### IMMEDIATELY AFTER Any Edit/Write Tool Call:
```
Step 3: Run `git add .`
Step 4: Run `git commit -m "description"`
Step 5: Run `git push` (if remote configured)
```

### REQUIRED SEQUENCE (No Exceptions):
1. `git pull` -> 2. Edit file -> 3. `git add . && git commit && git push`

**DO NOT:**
- Skip the pull step
- Make multiple edits without committing
- Forget to push after committing
- Say "I'll commit later" - commit IMMEDIATELY

**IF NO REMOTE:** Still commit locally. Inform user no remote is configured.

---

## MANDATORY: Virtual Environment (venv)

**The MCP server runs using the project venv, NOT the system Python.**

- Venv path: `/workspaces/sfu-library-mcp/.venv/bin/python3`
- The MCP server is launched via Docker exec using this venv
- **ALL package installs MUST target the venv:**
  ```bash
  # Correct — installs into the venv
  sudo /workspaces/sfu-library-mcp/.venv/bin/pip install <package>

  # WRONG — installs into system Python, MCP server won't see it
  pip install <package>
  pip3 install <package>
  ```
- **After installing new packages:** The MCP server process must be restarted (restart Claude Desktop or close/reopen conversation)
- **To verify a package is available to the MCP server:**
  ```bash
  /workspaces/sfu-library-mcp/.venv/bin/python3 -c "import <package>; print('OK')"
  ```
- The venv is owned by root, so `sudo` is required for pip installs

---
# Code Search Preferences

When searching for code patterns or understanding the codebase:
- **Prefer Pommel MCP tools** for semantic code search
- Use Pommel for:
  - Architecture understanding
  - Finding related code
  - Intent-based searches (e.g., "how does authentication work?")
  - Discovering patterns across the codebase
- Use Grep/Glob only for:
  - Exact string matching
  - Simple text patterns
  - File name searches

## Available Pommel Tools

- `mcp__pommel__pommel_search_project` - Semantic search within this project
- `mcp__pommel__pommel_search_all` - Search across all ClaudeBox projects
- `mcp__pommel__pommel_list_projects` - List available projects for search
- `mcp__pommel__pommel_reindex` - Reindex project after major changes

---

# Pommel Semantic Code Search

## Primary Method: Pommel MCP Tools (DEFAULT)

**ALWAYS use Pommel MCP tools first for code exploration:**

- mcp__pommel__pommel_search_local - Search current project (FASTEST)
- mcp__pommel__pommel_search_project - Search specific project
- mcp__pommel__pommel_search_all - Search across all projects
- mcp__pommel__pommel_list_projects - List available projects
- mcp__pommel__pommel_reindex - Trigger reindex after changes

**For Explore agents: Start with mcp__pommel__pommel_search_local for current project.**

## Fallback Method: HTTP API (When MCP unavailable)

### From Within App Container (Cross-Container Access)
**Uses internal port 7420 for all containers:**

\\\ash
# Search current project's Pommel
curl -X POST http://claudebox-\${PROJECT_NAME}-pommel:7420/search \
  -H "Content-Type: application/json" \
  -d '{"query": "YOUR_SEARCH_QUERY", "limit": 5}'

# Example: From dashboard container, search sfu-library-mcp
curl -X POST http://claudebox-sfu-library-mcp-pommel:7420/search \
  -H "Content-Type: application/json" \
  -d '{"query": "database connection", "limit": 5}'
\\\

### From Host Machine
**Each project has unique external port:**

\\\ash
# Dashboard (port 7421)
curl -X POST http://localhost:7421/search -H "Content-Type: application/json" \
  -d '{"query": "search term", "limit": 5}'

# sfu-library-mcp (port 7424)
curl -X POST http://localhost:7424/search -H "Content-Type: application/json" \
  -d '{"query": "search term", "limit": 5}'

# See POMMEL_PORT_MAPPING.md for complete port list
\\\

### Port Mapping Reference (Quick)

| Project | Container Name | Internal Port | External Port |
|---------|----------------|---------------|---------------|
| dashboard | claudebox-dashboard-pommel | 7420 | 7421 |
| business-plan-cataloger | claudebox-business-plan-cataloger-pommel | 7420 | 7422 |
| ccpsandyass-cybermap | claudebox-ccpsandyass-cybermap-pommel | 7420 | 7423 |
| sfu-library-mcp | claudebox-sfu-library-mcp-pommel | 7420 | 7424 |
| ccpsandyass-scripts | claudebox-ccpsandyass-scripts-pommel | 7420 | 7425 |
| ccpsandyass-separate | claudebox-ccpsandyass-separate-pommel | 7420 | 7426 |
| essay-script | claudebox-essay-script-pommel | 7420 | 7427 |
| geo-property-data | claudebox-geo-property-data-pommel | 7420 | 7428 |
| graph-visualization | claudebox-graph-visualization-pommel | 7420 | 7429 |
| llm-api | claudebox-llm-api-pommel | 7420 | 7430 |
| misc-scripts | claudebox-misc-scripts-pommel | 7420 | 7431 |
| sfu-auto-researcher | claudebox-sfu-auto-researcher-pommel | 7420 | 7432 |
| sfu-library-api | claudebox-sfu-library-api-pommel | 7420 | 7433 |
| website-cataloger | claudebox-website-cataloger-pommel | 7420 | 7434 |

**Key insight: All Pommel containers use port 7420 internally. Use container names for cross-container access. Use unique external ports (7421-7434) for host access.**

## Quick Reference for Explore Agents
1. Start: mcp__pommel__pommel_search_local (current project)
2. Cross-project: mcp__pommel__pommel_search_all
3. Fallback: HTTP API via curl (use container name:7420 for cross-container)
4. Last resort: Grep/Glob

---

# Git Version Control

This project uses Git for version control. Git is pre-installed in the container and ready to use.

## Git Setup

Git is automatically initialized when the project is created. The repository is configured with:
- Default branch: `main`
- User: `ClaudeBox` (claudebox@local)

## Common Git Commands

```bash
# Check status
git status

# Stage all changes
git add .

# Commit changes
git commit -m "Your commit message"

# Push to remote (after setting up remote)
git push

# Pull from remote
git pull

# View commit history
git log --oneline

# Create and switch to new branch
git checkout -b feature-name

# Switch to existing branch
git checkout main
```

## Setting Up Remote Repository

```bash
# Add remote origin
git remote add origin https://github.com/username/repo.git

# Push and set upstream
git push -u origin main
```

## Git Best Practices in ClaudeBox

1. **Commit frequently** - Make small, focused commits
2. **Write clear commit messages** - Describe what changed and why
3. **Check status before committing** - Review changes with `git status` and `git diff`
4. **Push regularly** - Keep remote in sync with local changes
5. **Use branches for features** - Keep main branch stable

## Dashboard Git Management

The ClaudeBox Dashboard provides a Git Management panel for:
- Viewing repository status
- Creating commits
- Managing branches
- Configuring remotes
- Push/Pull operations

Access via the **Git Management** button in the dashboard header.

---

## MANDATORY Git Workflow for Claude Code

**CRITICAL: Follow this workflow when making ANY code changes.**

### Before Starting Work (PULL)
1. **ALWAYS pull before making changes** to ensure you have the latest code
2. Run `git pull` in the project directory before any edits
3. If there are conflicts, resolve them before proceeding

### After Making Changes (COMMIT + PUSH)
1. **After completing a logical unit of work**, commit immediately
2. Use descriptive commit messages that explain the "why" not just the "what"
3. **PUSH after every commit** to sync changes with remote

### Git Workflow Commands
```bash
# Before starting work
git pull

# After making changes
git add .
git commit -m "$(cat <<'EOF'
Description of changes

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"
git push
```

### When to Commit
- After completing each feature or fix
- Before switching to a different task
- Before ending a session
- After any significant code change

### Git Subtree Workflow (Shared Code)
When working with shared code via subtrees:
1. Pull subtree updates before starting: use Dashboard â†’ Git â†’ Subtrees â†’ Pull
2. After modifying shared code, commit to the current project
3. Consider pushing changes back to source project if applicable

---
