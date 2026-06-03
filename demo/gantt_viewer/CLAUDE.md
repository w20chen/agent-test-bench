# Gantt Viewer — Agent Guide

Interactive web application for visualizing LLM agent execution traces as
multi-lane Gantt charts with resource overlays.

**Stack**: FastAPI backend + Solid.js / Vite frontend, Canvas 2D rendering.

---

## Launch

```bash
# Development (Vite HMR on :5173, backend on :8765)
python -m trace_collect.cli gantt-serve --dev

# Production (static dist/ served from :8765)
python -m trace_collect.cli gantt-serve

# Standalone HTML export for the curated GLM/OpenClaw/100 SWE-rebench cohort
PYTHONPATH=src:. python -m trace_collect.cli gantt-export \
  --preset swe-rebench-glm-openclaw-100 \
  --group all \
  --output-dir results/gantt-viewer

# Full options
python -m trace_collect.cli gantt-serve \
  --config demo/gantt_viewer/configs/example.yaml \
  --port 8765 --host 127.0.0.1 \
  --dev --no-browser
```

| Flag | Default | Notes |
|------|---------|-------|
| `--config` | `demo/gantt_viewer/configs/example.yaml` | Discovery YAML |
| `--dev` | off | Starts Vite dev server on 5173, sets `GANTT_VIEWER_DEV=1` |
| `--port` | `8765` | Backend listen port |
| `--host` | `127.0.0.1` | Backend listen host |
| `--no-browser` | off | Skip auto-opening browser |

### Environment Variables

| Var | Default | Notes |
|-----|---------|-------|
| `GANTT_VIEWER_CONFIG` | `demo/gantt_viewer/configs/example.yaml` | Config path override |
| `GANTT_VIEWER_RUNTIME_STATE` | `~/.cache/agent-sched-bench/gantt-runtime-state.json` | State file |
| `GANTT_VIEWER_DEV` | `0` | Set to `1` by `--dev` flag |

---

## Architecture

```
                  ┌─────────────────────────────────┐
                  │  Solid.js SPA (Canvas 2D)       │
                  │  Port 5173 (dev) / 8765 (prod)  │
                  └──────────┬──────────────────────┘
                             │ fetch /api/*
                  ┌──────────▼──────────────────────┐
                  │  FastAPI backend (:8765)         │
                  │  routes.py → payload.py          │
                  ├─────────────────────────────────┤
                  │  RuntimeTraceRegistry            │
                  │   = DiscoveryState (config YAML) │
                  │   + runtime additions/removals   │
                  │   + uploaded traces               │
                  └──────────┬──────────────────────┘
                             │ reads
                  ┌──────────▼──────────────────────┐
                  │  Trace files on disk             │
                  │  trace.jsonl + resources.json    │
                  └─────────────────────────────────┘
```

---

## REST API

All data endpoints are under `/api`. For curl examples and the recommended
agent workflow, see `AGENT_INTERFACE.md` in this directory.

### Endpoints

| Method | Path | Request | Response | Notes |
|--------|------|---------|----------|-------|
| GET | `/api/health` | — | `{status, n_discovered}` | Health check |
| GET | `/api/traces` | — | `{traces: TraceDescriptor[], registries}` | List all tracked traces |
| POST | `/api/traces/reload` | — | same as GET /api/traces | Re-read config + state file |
| POST | `/api/traces/register` | `{paths, labels_by_path?}` | `{registered: TraceDescriptor[]}` | Register existing trace files |
| POST | `/api/traces/unregister` | `{ids}` | `{removed_ids, missing_ids}` | Stop tracking (never deletes files) |
| POST | `/api/traces/upload` | FormData `file` | `{descriptor, payload_fragment}` | Upload ad-hoc JSONL blob |
| POST | `/api/payload` | `{ids: string[]}` | `{registries, traces: TracePayload[], errors?}` | Build Gantt payloads |

### Error Semantics

| Code | Meaning |
|------|---------|
| `409` | Trace already tracked or ID collision |
| `422` | Invalid file format, unparseable trace, or all payloads failed |
| `404` | Unknown trace ID in payload or unregister |

Partial failures on `/api/payload` return mixed `traces` + `errors` arrays.

### Auto-Import

Both `register` and `upload` auto-detect raw Claude Code session JSONL and
import it to canonical trace format before tracking. For `register`, adjacent
`subagents/` sidechains are preserved; `upload` (single blob) skips sidechains.

---

## Discovery & Registry

### Config-based Discovery

YAML config defines trace groups with glob patterns:

```yaml
# demo/gantt_viewer/configs/example.yaml
groups:
  - name: "AC1 — openclaw sepal_ui"
    paths:
      - traces/swe-rebench/smoke-*/openclaw__sepal_ui/**/trace.jsonl
```

Loaded via `DiscoveryState.from_config_path()`. Paths resolved relative to repo root.
Format detection: `sniff_format()` checks first JSONL line for `type: "trace_metadata"`
with matching `CURRENT_TRACE_FORMAT_VERSION`.

### RuntimeTraceRegistry

Overlays runtime state on top of config discovery:

| Operation | Effect |
|-----------|--------|
| `register_paths()` | Add existing files to runtime tracking |
| `register_uploaded_descriptor()` | Add uploaded file |
| `unregister_ids()` | Runtime registrations: deleted. Config traces: suppressed. |
| `reload()` | Re-read config discovery + state file |
| `get_descriptor(id)` | Lookup by trace ID |

**Persistence**: Runtime state (registered + suppressed IDs) saved to
`~/.cache/agent-sched-bench/gantt-runtime-state.json`. Survives server restarts.
Underlying trace files are **never** deleted by the API.

**ID Generation**: Stable hash of canonical path + label slug.

---

## Gantt Payload Structure

`POST /api/payload` returns `TracePayload` objects with:

### Lanes

```json
{"id": "agent-id", "label": "agent-id", "trace_id": "..."}
```

One lane per `agent_id` per trace. Multi-lane for traces with subagents.

### Spans

```json
{
  "type": "llm",
  "start": 0.0, "end": 1.5,
  "start_abs": 1712345678.0, "end_abs": 1712345679.5,
  "start_real": 0.0, "end_real": 1.2,
  "iteration": 0,
  "lane_id": "agent-id",
  "detail": {"llm_content": "...", "tool_calls_requested": 2}
}
```

Span types: `llm`, `tool`, `mcp`, `scheduling`.

- `start`/`end`: relative to trace t0
- `start_abs`/`end_abs`: epoch seconds
- `start_real`/`end_real`: gap-compressed (idle time removed)
- LLM timing prefers `llm_call_time_ms` > `openrouter_generation_time_ms` > `llm_latency_ms` > wall duration
- `scheduling` spans: auto-synthesized gaps between actions, contain gap events

### Markers

```json
{
  "type": "scheduling",
  "event": "iteration_start",
  "t": 0.0, "t_abs": 1712345678.0,
  "t_real": 0.0,
  "iteration": 0,
  "detail": {}
}
```

Marker categories (from trace events): `scheduling`, `session`, `context`, `mcp`.

### Resource Timeline

```json
{
  "t": 0.0, "t_abs": 1712345678.0,
  "t_real": 0.0,
  "cpu_percent": 45.2, "memory_mb": 1024,
  "disk_read_mb": 10.5, "disk_write_mb": 20.3,
  "net_rx_mb": 1.0, "net_tx_mb": 0.5,
  "context_switches": 150
}
```

Loaded from `resources.json` (same directory as trace.jsonl). Aligned to
trace t0 and gap-compressed to match real timeline.

### Registries

Color/shape registries for spans and markers returned with every payload:

```json
{
  "registries": {
    "spans": {"llm": {"color": "#00E5FF"}, "tool": {"color": "#FF6D00"}, ...},
    "markers": {"scheduling": {"shape": "diamond"}, ...}
  }
}
```

---

## Frontend

### Tech Stack

- **Solid.js** (reactive framework) + **Vite** (build tool) + **TypeScript**
- Canvas 2D rendering (not SVG/DOM) for performance with large traces

### View Modes

| Setting | Options | Effect |
|---------|---------|--------|
| TimeMode | `sync` / `abs` | Synchronized vs absolute timeline |
| ClockMode | `wall` / `real` | Wall-clock vs gap-compressed time |
| ViewMode | `layered` / `concise` | Full vs compact lane height |
| ThemeMode | `dark` / `light` | Color theme |
| Zoom | 0.25x – 32x | 8 presets + Ctrl/Cmd+wheel free-form |
| ResourceMetric | `cpu` / `memory` / `disk_io` / `net_io` | Resource chart metric |
| ShowResourceChart | toggle | Show/hide resource timeline |

### Interactive Features

- **Zoom**: Ctrl/Cmd+wheel (cursor-anchored) or preset buttons
- **Pan**: horizontal scroll
- **Click span/marker**: pin tooltip with detail card
- **Hover**: floating tooltip
- **ESC**: close pinned tooltip
- **Trace chips**: load/toggle/remove traces via header bar
- **Drag & drop**: upload JSONL files directly
- **URL param**: `?autoload=all` loads all discovered traces on startup

### Key Frontend Files

| File | Purpose |
|------|---------|
| `frontend/src/App.tsx` | Root component, state orchestration |
| `frontend/src/api/client.ts` | HTTP client (fetch wrappers) |
| `frontend/src/canvas/CanvasRenderer.ts` | Canvas rendering engine (650+ lines) |
| `frontend/src/state/signals.ts` | Solid.js reactive state |
| `frontend/src/bootstrap/autoload.ts` | Auto-load logic (first trace or all) |
| `frontend/src/components/` | UI components (header, sidebar, tooltip, etc.) |

---

## Backend Modules

| File | Purpose |
|------|---------|
| `backend/app.py` | FastAPI factory, static mount, registry init |
| `backend/routes.py` | All REST endpoints |
| `backend/runtime_registry.py` | Registry overlay (config + runtime + uploads) |
| `backend/payload.py` | Trace → Gantt payload transformation (spans, markers, gaps, resources) |
| `backend/discovery.py` | YAML config parsing, glob-based trace discovery, format sniffing |
| `backend/ingest.py` | Format detection + Claude Code auto-import |
| `backend/uploads.py` | Upload persistence to cache dir |
| `backend/dev.py` | CLI launcher, Vite subprocess management |
| `backend/schema.py` | Pydantic models for all API request/response types |

---

## Data Flow Summary

```
1. Startup: DiscoveryState reads config YAML → glob for trace files
2. RuntimeTraceRegistry merges config traces + runtime state file
3. Frontend GET /api/traces → descriptor list
4. User clicks "Load" → POST /api/payload {ids: [...]}
5. Backend: TraceData.load(path) → build_gantt_payload_multi()
   - Extract spans from actions (llm_call → llm, tool_exec → tool/mcp)
   - Synthesize scheduling gap spans
   - Extract markers from events
   - Apply gap compression (_apply_real_timeline)
   - Load resource timeline from resources.json
6. Frontend receives TracePayload → CanvasRenderer draws on canvas
7. User interactions: zoom, pan, click, hover → tooltip/detail
```

---

## Adding New Traces

### Via API (preferred for agents)

See `AGENT_INTERFACE.md` for the full curl-based workflow.

```python
import requests
base = "http://127.0.0.1:8765"

# Register existing file
r = requests.post(f"{base}/api/traces/register",
    json={"paths": ["traces/.../trace.jsonl"]})
trace_id = r.json()["registered"][0]["id"]

# Get Gantt payload
payload = requests.post(f"{base}/api/payload",
    json={"ids": [trace_id]}).json()

# Cleanup
requests.post(f"{base}/api/traces/unregister",
    json={"ids": [trace_id]})
```

### Via Config

Add a group to the discovery YAML, then reload:

```yaml
groups:
  - name: "My Experiment"
    paths:
      - traces/my-run/**/trace.jsonl
```

```bash
curl -s -X POST http://127.0.0.1:8765/api/traces/reload
```

### Via Upload

```bash
curl -s http://127.0.0.1:8765/api/traces/upload \
  -F 'file=@path/to/trace.jsonl'
```

Or drag & drop onto the web UI.

---

## Design Notes

- **In-memory only**: no database. RuntimeTraceRegistry is a dict. Restart
  preserves state via the runtime state JSON file.
- **Single-process**: not designed for multi-worker deployments.
- **No auth**: no access control on endpoints.
- **File access**: server needs filesystem access to trace directories.
- **Canvas over DOM**: chosen for performance with traces containing 1000+ spans.
- **Gap compression**: removes idle time between actions so the visual timeline
  shows actual work, not waiting. Configurable via ClockMode toggle.
