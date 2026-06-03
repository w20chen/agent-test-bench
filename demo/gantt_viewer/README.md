# Gantt Viewer

Dynamic local viewer for benchmark traces.

This replaces the old static `trace_collect.cli gantt` HTML generator with a
FastAPI backend plus a Solid/Vite frontend under `demo/gantt_viewer/`.

## Current Status

Implemented:

- `GET /api/health`
- `GET /api/traces`
- `POST /api/traces/register`
- `POST /api/payload`
- `POST /api/traces/reload`
- `POST /api/traces/unregister`
- `POST /api/traces/upload`
- Solid/Vite frontend with:
  - `SYNC` / `ABS` time mode
  - `WALL` / `REAL` timing mode
  - `LAYERED` / `CONCISE` layout mode
  - `Load all` and per-trace load/toggle/remove controls
  - tooltip hover + pin
  - drag-drop upload
  - Ctrl/Cmd+wheel zoom

Partially implemented:

- canvas parity with the deleted static viewer is good on the main path, but
  there is still room to polish multi-trace ergonomics and deeper tooltip/scroll
  behavior

## Layout

```text
demo/gantt_viewer/
├── backend/
├── configs/
├── frontend/
└── tests/
```

## Prerequisites

- Conda env "ML" active (`conda activate ML` — see project root README)
- Node/npm available for the frontend

Recommended setup:

```bash
conda activate ML
pip install -e ".[dev,gantt-viewer]"
make gantt-viewer-install
```

## Run

Development mode starts the backend and a Vite dev server:

```bash
make gantt-viewer-dev
```

Equivalent CLI:

```bash
PYTHONPATH=src:. python -m trace_collect.cli gantt-serve --dev
```

Production mode serves `frontend/dist` directly from FastAPI:

```bash
make gantt-viewer-build
PYTHONPATH=src:. python -m trace_collect.cli gantt-serve
```

## Static HTML Export

Export the curated SWE-rebench GLM/OpenClaw/100-iteration cohort as standalone
HTML snapshots:

```bash
make gantt-viewer-build
PYTHONPATH=src:. conda run -n ML python -m trace_collect.cli gantt-export \
  --preset swe-rebench-glm-openclaw-100 \
  --group all \
  --output-dir results/gantt-viewer
```

The preset is intentionally narrow: it validates the 19 source traces in
`configs/simulate/openclaw-glm-19-manifest.json` as `scaffold=openclaw`,
`model=z-ai/glm-5.1`, and `max_iterations=100`. It exports the raw traces,
closed-loop simulation traces, and the five Poisson sweep groups. Resource
timelines are loaded from each trace directory's adjacent `resources.json`;
missing resource samples are reported in `manifest.json` rather than filled in.

Use the example discovery config explicitly if you want the acceptance layout:

```bash
PYTHONPATH=src:. python -m trace_collect.cli gantt-serve \
  --config demo/gantt_viewer/configs/example.yaml
```

## Test

Backend pytest + frontend vitest:

```bash
make gantt-viewer-test
```

Browser smoke:

```bash
make gantt-viewer-smoke
```

Current smoke coverage:

- page boot
- default AC1 auto-load
- synthetic drag-drop upload
- button-triggered JSONL upload
- pinned tooltip after lane click
- `Load all` across the discovered trace set

For agent-driven workflows, use `demo/gantt_viewer/AGENT_INTERFACE.md`.

Config discovery still requires canonical trace JSONL. At runtime, the REST
API (`/api/traces/register` and `/api/traces/upload`) can auto-import raw
Claude Code session JSONL by reusing `python -m trace_collect.cli import-claude-code`
internally before registering the resulting canonical trace. `register` keeps
adjacent Claude Code sidechains when they exist next to the session file;
single-file `upload` imports only the uploaded main session JSONL.

Frontend bundle check only:

```bash
make gantt-viewer-build
```

## Acceptance Checks

The shipped example config currently discovers:

- `1` canonical openclaw trace

Quick API check:

```bash
python - <<'PY'
from fastapi.testclient import TestClient
from demo.gantt_viewer.backend.app import create_app

client = TestClient(create_app())
traces = client.get("/api/traces").json()["traces"]
print("n_traces", len(traces))
print("n_trace", sum(t["source_format"] == "trace" for t in traces))
PY
```

Expected today:

- `n_traces 1`
- `n_trace 1`

AC1 payload check:

```bash
python - <<'PY'
from fastapi.testclient import TestClient
from demo.gantt_viewer.backend.app import create_app

client = TestClient(create_app())
payload = client.post("/api/payload", json={"ids": ["ac1-12rambau__sepal_ui-411"]}).json()
print(payload["traces"][0]["metadata"]["scaffold"])
print(len(payload["traces"][0]["lanes"]))
PY
```

Expected:

- `openclaw`
- `1`

## OpenAPI Types

When backend schema changes, regenerate both the frozen snapshot and the
frontend types together:

```bash
python - <<'PY' > demo/gantt_viewer/tests/fixtures/openapi.snapshot.json
import json
from demo.gantt_viewer.backend.app import create_app
print(json.dumps(create_app().openapi(), indent=2, sort_keys=True))
PY

cd demo/gantt_viewer/frontend
npx openapi-typescript ../tests/fixtures/openapi.snapshot.json -o src/api/schema.gen.ts
```
