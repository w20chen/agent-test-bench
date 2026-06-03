# Gantt Viewer Agent Interface

Use this file when an agent needs to add, remove, inspect, or render traces
against a running Gantt viewer server.

Assume the server is already running at `http://127.0.0.1:8765`.

## Safe Defaults

- Prefer HTTP over editing config files
- Prefer `register` for existing paths and `upload` for ad hoc JSONL blobs
- `register` / `upload` auto-import raw Claude Code session JSONL into canonical
  trace JSONL before tracking
- `unregister` only stops tracking; it never deletes files
- After changing the registry, verify with `GET /api/traces` before calling
  `POST /api/payload`

## Endpoints

### Health check

```bash
curl -s http://127.0.0.1:8765/api/health
```

Response: `{"status": "ok", "n_discovered": <int>}`

### List currently tracked traces

```bash
curl -s http://127.0.0.1:8765/api/traces
```

Response:

```json
{
  "traces": [
    {"id": "ac1-...", "label": "...", "source_format": "trace", "path": "/abs/path/trace.jsonl", "size_bytes": 12345, "mtime": 1712345678.0}
  ],
  "registries": {"spans": {...}, "markers": {...}}
}
```

### Register one or more existing trace files

Absolute or repo-relative paths both work. Canonical trace JSONL is accepted
directly. Raw Claude Code session JSONL is auto-imported behind the scenes and
then registered as a runtime canonical trace. For path registration, adjacent
Claude Code sidechains are preserved when the original session file lives next
to its `<session_uuid>/subagents/` directory.

```bash
curl -s http://127.0.0.1:8765/api/traces/register \
  -H 'Content-Type: application/json' \
  -d '{
    "paths": [
      "traces/swe-rebench/smoke-20260407T121213Z/openclaw__sepal_ui/_workspaces/12rambau__sepal_ui-411/trace.jsonl"
    ]
  }'
```

Optional custom labels:

```bash
curl -s http://127.0.0.1:8765/api/traces/register \
  -H 'Content-Type: application/json' \
  -d '{
    "paths": ["/abs/path/to/trace.jsonl"],
    "labels_by_path": {
      "/abs/path/to/trace.jsonl": "my-runtime-trace"
    }
  }'
```

### Upload a new trace blob

Canonical trace JSONL is accepted directly. Raw Claude Code session JSONL is
auto-imported behind the scenes and then tracked as a runtime canonical trace.
Because upload only sends one blob, adjacent Claude Code `subagents/` sidechains
are not available on this path.

```bash
curl -s http://127.0.0.1:8765/api/traces/upload \
  -F 'file=@/abs/path/to/trace.jsonl'
```

Response includes the descriptor and a pre-built Gantt payload fragment:

```json
{
  "descriptor": {"id": "...", "label": "...", "source_format": "trace", "path": "...", "size_bytes": 0, "mtime": 0.0},
  "payload_fragment": {"id": "...", "label": "...", "metadata": {...}, "t0": 0.0, "lanes": [...], "resource_timeline": [...]}
}
```

### Stop tracking one or more traces

This only unregisters ids from the runtime registry.

```bash
curl -s http://127.0.0.1:8765/api/traces/unregister \
  -H 'Content-Type: application/json' \
  -d '{
    "ids": ["runtime-my-trace-1234567890"]
  }'
```

### Rebuild tracked state

This re-reads config discovery plus the persisted runtime state file.

```bash
curl -s -X POST http://127.0.0.1:8765/api/traces/reload
```

### Render one or more traces into payload form

```bash
curl -s http://127.0.0.1:8765/api/payload \
  -H 'Content-Type: application/json' \
  -d '{
    "ids": [
      "ac1-12rambau__sepal_ui-411"
    ]
  }'
```

Response:

```json
{
  "registries": {
    "spans": {"llm": {"color": "...", "label": "...", "order": 0}, ...},
    "markers": {"scheduling": {"symbol": "...", "color": "...", "label": "..."}, ...}
  },
  "traces": [
    {
      "id": "ac1-...",
      "label": "...",
      "metadata": {"scaffold": "openclaw", "model": "...", "n_actions": 42, "n_iterations": 10, "n_events": 15, "elapsed_s": 120.0},
      "t0": 1712345678.0,
      "lanes": [{"agent_id": "...", "spans": [...], "markers": [...]}],
      "resource_timeline": [{"t": 0.0, "t_abs": 0.0, "cpu_percent": 45.0, "memory_mb": 1024.0, ...}]
    }
  ],
  "errors": []
}
```

Partial failures return mixed `traces` + `errors`; each error is
`{"trace_id": "...", "stage": "trace_load|payload_build", "error": "..."}`.

## Recommended Agent Sequence

1. Register or upload the trace.
2. Call `GET /api/traces` and capture the returned id.
3. Call `POST /api/payload` with that id.
4. When finished, call `POST /api/traces/unregister`.

Example using Python:

```python
import requests

base = "http://127.0.0.1:8765"

registered = requests.post(
    f"{base}/api/traces/register",
    json={"paths": ["/abs/path/to/trace.jsonl"]},
).json()["registered"][0]

trace_id = registered["id"]

payload = requests.post(
    f"{base}/api/payload",
    json={"ids": [trace_id]},
).json()

requests.post(
    f"{base}/api/traces/unregister",
    json={"ids": [trace_id]},
)
```

## Failure Semantics

- `409` means the trace is already tracked or the requested registration would
  collide with an existing id/path
- `422` means the path or file content is invalid or not sniffable as a canonical trace JSONL
- `404` on `payload` or `unregister` means the id is not known to the current
  merged registry

## Persistence Model

- Runtime additions and suppressions are stored in a runtime state file under
  the user cache directory
- Restarting the server preserves:
  - `register` results
  - `upload` results
  - `unregister` suppressions
- Underlying trace files are never deleted by the API
