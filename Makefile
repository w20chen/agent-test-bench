PYTHON ?= python3
UV ?= uv
TRACE_COLLECT = PYTHONPATH=src $(PYTHON) -m trace_collect.cli --provider $(PROVIDER)
GANTT_VIEWER_FRONTEND = demo/gantt_viewer/frontend

.PHONY: help pull test lint serve-vllm run-smoke run-sweep collect-results setup-swebench-repos build-swebench-images download-swebench-verified download-swe-rebench setup-swe-rebench-repos setup-swe-rebench setup-arm-host smoke-swe-rebench-openclaw gantt-viewer-install gantt-viewer-dev gantt-viewer-build gantt-viewer-test gantt-viewer-smoke gantt-viewer-clean

help:
	@printf "Targets:\n"
	@printf "  pull              Fast-forward pull the current branch\n"
	@printf "  test              Run the full test suite\n"
	@printf "  lint              Run ruff\n"
	@printf "  serve-vllm        Run the raw vLLM launcher\n"
	@printf "  run-smoke         Run the current infrastructure smoke suite\n"
	@printf "  run-sweep         Run the harness sweep when HARNESS-1 is available\n"
	@printf "  collect-results   Pull result artifacts back via rsync\n"
	@printf "  download-swebench-verified  Download & select 32 tasks from SWE-bench Verified\n"
	@printf "  setup-swebench-repos        Clone repos referenced by selected tasks\n"
	@printf "  build-swebench-images       Build Podman container images\n"
	@printf "  download-swe-rebench        Download SWE-rebench (nebius/SWE-rebench) filtered split\n"
	@printf "  setup-swe-rebench-repos     Clone repos referenced by SWE-rebench tasks\n"
	@printf "  setup-swe-rebench           Shortcut: download-swe-rebench + setup-swe-rebench-repos\n"
	@printf "  setup-arm-host              Enable amd64 container execution on ARM Docker hosts (uses sudo if needed)\n"
	@printf "  smoke-swe-rebench-openclaw  Run $(SMOKE_N) SWE-rebench tasks through openclaw\n"
	@printf "  gantt-viewer-install        Install frontend dependencies with npm\n"
	@printf "  gantt-viewer-dev            Launch the dynamic Gantt viewer in dev mode\n"
	@printf "  gantt-viewer-build          Build the frontend bundle\n"
	@printf "  gantt-viewer-test           Run backend pytest plus frontend vitest\n"
	@printf "  gantt-viewer-smoke          Run a browser smoke test against the built viewer\n"
	@printf "  gantt-viewer-clean          Remove viewer build/cache artifacts\n"

pull:
	./scripts/pull_repo.sh

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

serve-vllm:
	./scripts/serve_vllm.sh

run-smoke:
	./scripts/run_smoke.sh

run-sweep:
	./scripts/run_sweep.sh

collect-results:
	./scripts/collect_results.sh

setup-swebench-repos:
	./scripts/setup/clone_repos.sh data/swebench_verified/tasks.json

build-swebench-images:
	./scripts/setup/build_images.sh

download-swebench-verified:
	./scripts/setup/swebench_data.sh

download-swe-rebench:
	./scripts/setup/swe_rebench_data.sh

setup-swe-rebench-repos:
	./scripts/setup/clone_repos.sh data/swe-rebench/tasks.json data/swe-rebench/repos

setup-swe-rebench: download-swe-rebench setup-swe-rebench-repos

setup-arm-host:
	@if [ "$$(id -u)" -eq 0 ]; then \
		./scripts/setup/arm_setup.sh install; \
	else \
		sudo ./scripts/setup/arm_setup.sh install; \
	fi

# ── SWE-rebench smoke runs ─────────────────────────────────────────────
# Override PROVIDER (default: dashscope) and SMOKE_N (default: 2).
PROVIDER ?= dashscope
SMOKE_N ?= 2

smoke-swe-rebench-openclaw:
	$(TRACE_COLLECT) \
	    --benchmark swe-rebench \
	    --scaffold openclaw \
	    --sample $(SMOKE_N) \
	    --verbose

gantt-viewer-install:
	cd $(GANTT_VIEWER_FRONTEND) && npm install

gantt-viewer-dev:
	PYTHONPATH=src:. $(PYTHON) -m trace_collect.cli gantt-serve --dev

gantt-viewer-build:
	cd $(GANTT_VIEWER_FRONTEND) && npm run build

gantt-viewer-test:
	PYTHONPATH=src:. $(PYTHON) -m pytest demo/gantt_viewer/tests -v
	cd $(GANTT_VIEWER_FRONTEND) && npm test

gantt-viewer-smoke:
	./scripts/smoke_gantt_viewer.sh

gantt-viewer-clean:
	rm -rf $(GANTT_VIEWER_FRONTEND)/dist $(GANTT_VIEWER_FRONTEND)/node_modules
	rm -rf ~/.cache/agent-sched-bench/gantt-cc-import
	rm -rf ~/.cache/agent-sched-bench/gantt-uploads
	rm -f ~/.cache/agent-sched-bench/gantt-runtime-state.json
