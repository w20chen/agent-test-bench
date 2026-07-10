# Task Container Environment Model

This document explains what happens to a benchmark task container when
OpenClaw is run through `python -m trace_collect.cli`, why the container
environment is prepared this way, and what the runtime deliberately does not
provide.

The short version is:

- The benchmark image remains the source of truth for task and project
  dependencies.
- The collector installs only the Python packages needed to run the OpenClaw
  controller inside the task container.
- The collector does not install `pytest`, project test dependencies, or
  repository-specific packages.
- Bare `pip` and `pip3` are made available only as task-local aliases for the
  resolved container Python's `python -m pip`.

## Example Command

```bash
ARM_IMAGE_MODE=qemu PYTHONPATH=src python -m trace_collect.cli \
  --provider deepseek --model deepseek-v4-flash \
  --benchmark swe-rebench --scaffold openclaw \
  --container docker --mcp-config none \
  --sample 100
```

This command uses the benchmark plugin for `swe-rebench`, selects up to 100
tasks, starts one task container per attempt, runs OpenClaw inside that
container, and records traces under the benchmark's `trace_root`.

`ARM_IMAGE_MODE=qemu` matters only on ARM64 hosts. In that mode, container-mode
SWE benchmarks use their original x86_64 task images through Docker/QEMU
emulation. On non-ARM hosts, this setting is effectively unused.

## Phase 1: Benchmark and Task Selection

The CLI loads `configs/benchmarks/<slug>.yaml`, instantiates the registered
benchmark plugin, and asks the plugin to load and normalize tasks.

For `swe-rebench`, the YAML declares:

```yaml
slug: swe-rebench
harness_dataset: nebius/SWE-rebench
harness_split: filtered
trace_root: traces/swe-rebench
default_max_iterations: 100
default_prompt_template: cc_aligned
arm_base_image: swe-arm-base:latest
```

The plugin normalizes each dataset row while preserving benchmark-specific
fields such as `repo`, `base_commit`, `docker_image`, `install_config`, and
`test_cmd`. The `docker_image` field becomes the task image used for
container-mode execution.

`--sample 100` selects the first 100 tasks after the benchmark plugin's normal
selection logic. It is not a dependency or environment setting.

## Phase 2: Image Choice

For container benchmarks, the collector asks the benchmark plugin for a task
image.

On normal x86_64 hosts, or on ARM hosts with `ARM_IMAGE_MODE=qemu`, the
collector uses the task image declared by the benchmark row.

On ARM hosts without QEMU mode, SWE-style benchmarks use an ARM-native base
image and reconstruct the task from local repo mirrors. That path is separate
from QEMU mode and requires the benchmark's `repos_root` setup.

The image may also go through `ensure_fixed_image`, which repairs container
permission issues so the agent can edit `/testbed` as the runtime user. This
does not install project test dependencies.

## Phase 3: Container Start

Starting the container does not install Python packages. It creates a long-lived
container with `sleep infinity` and sets a controlled baseline environment.

Conceptually, the collector runs something like:

```bash
docker run -d --rm \
  --network=host \
  -v "$HOME:$HOME" \
  -v "<attempt_dir>:<attempt_dir>" \
  -v "<repo_root>:<repo_root>" \
  -w /testbed \
  -e HOME="$HOME" \
  -e PATH=/tmp/openclaw-task-userbase/bin:/usr/local/bin:/usr/bin:/bin \
  -e OPENCLAW_TASK_USERBASE=/tmp/openclaw-task-userbase \
  -e PIP_BREAK_SYSTEM_PACKAGES=1 \
  <image> sleep infinity
```

The exact command also includes container-runtime user arguments and, when
needed, image platform arguments.

The collector passes through a small set of host environment variables:

```text
HTTP_PROXY, HTTPS_PROXY, ALL_PROXY, NO_PROXY
http_proxy, https_proxy, all_proxy, no_proxy
PIP_INDEX_URL
TASK_CONTAINER_PIP_INDEX_URL
NANOBOT_MAX_CONCURRENT_REQUESTS
```

Set `TASK_CONTAINER_NO_PROXY=1` to avoid forwarding proxy variables into task
containers.

The task-local userbase is always:

```text
/tmp/openclaw-task-userbase
```

This path is inside the task container. It is not the shared OpenClaw bootstrap
cache.

## Phase 4: Python Runtime Probe

After the container starts, the collector finds a Python interpreter inside the
container. It tries common candidates in a fixed order, including:

```text
/usr/local/bin/python3
/usr/local/bin/python
/opt/miniconda3/bin/python3
/opt/conda/bin/python3
/usr/bin/python3
/usr/bin/python
python3
python
```

The selected interpreter must be Python 3.11 or newer. The selected path and
Python ABI tag are recorded in the task-container execution config.

This resolved Python is used for OpenClaw controller execution and for the
task-local `pip`/`pip3` aliases described below. The runtime does not replace
the container's original `python` or `python3` commands.

## Phase 5: OpenClaw Runtime Bootstrap

OpenClaw itself needs a few Python libraries to run inside the task container.
Those libraries are not benchmark project dependencies. They are controller
dependencies.

The base OpenClaw runtime dependencies are:

```text
openai
httpx
PyYAML
json-repair
loguru
pydantic
socksio
tiktoken
```

If `--mcp-config` is not `none`, the runtime also adds:

```text
mcp>=1.0
```

For `--mcp-config none`, `mcp` is not added.

The bootstrap cache is versioned by:

```text
bootstrap cache version
container architecture
Python ABI tag
requirements hash
```

This prevents cross-contamination between architectures, Python ABI versions,
and dependency sets. For example, an ARM64 wheel cache is not reused for an
x86_64 QEMU container, and a Python 3.11 cache is not reused for Python 3.12.

The runtime dependencies are installed with `pip install --target` into the
versioned `pydeps` directory under the host bootstrap cache. That directory is
then injected into the controller process with `PYTHONPATH`.

The bootstrap code also performs import health checks for the OpenClaw runtime
modules. A cache hit must pass those checks before it is reused.

## Phase 6: Task-Local pip and pip3 Aliases

Many agents try bare `pip` or `pip3` before trying `python -m pip`. Some task
images have a usable `python -m pip` but no `pip` executable on `PATH`, which
wastes iterations and can cause avoidable failures.

To smooth that edge without changing benchmark project dependencies, the
collector writes two task-local shims:

```text
/tmp/openclaw-task-userbase/bin/pip
/tmp/openclaw-task-userbase/bin/pip3
```

Each shim executes the resolved Python's `pip` module:

```sh
<resolved-container-python> -m pip "$@"
```

This is intentionally narrow:

- It does not install `pip` if the selected Python lacks the `pip` module.
- It does not install `pytest`.
- It does not provide a `pytest` command.
- It does not provide or override `python` or `python3`.
- It does not expose the shared OpenClaw bootstrap pip on `PATH`.

If the selected Python cannot run `-m pip`, then `pip` and `pip3` still fail.
That failure reflects the task image's actual Python capability. If later
project dependency installation changes the task-local Python import path in a
way that breaks pip itself, that remains a task-solving environment issue for
the agent to diagnose rather than a runtime-managed dependency repair.

## Phase 7: Preflight and Agent Execution

Before the agent starts, preflight verifies that the in-container controller can
import the OpenClaw runtime modules.

Then the OpenClaw runner starts inside the task container. It receives:

```text
PYTHONPATH=<OpenClaw runtime pydeps>:<repo src>:<repo root>
PYTHONNOUSERSITE=1
```

Those settings are for the controller process, not for arbitrary shell commands
the agent runs while solving the benchmark task.

When the agent uses the `exec` tool, the shell command environment is isolated
from the controller runtime:

```text
PYTHONPATH is removed
PYTHONNOUSERSITE is removed
PYTHONUSERBASE=/tmp/openclaw-task-userbase
PATH starts with /tmp/openclaw-task-userbase/bin
```

This means tool commands such as `pip install --user ...` install into the
task-local userbase for that container. They do not contaminate the shared
OpenClaw bootstrap cache.

## Replay Compatibility

Container-mode replay re-executes recorded tool commands inside a prepared
container. The replay agent applies the same task-command isolation:

```text
PYTHONPATH is removed
PYTHONNOUSERSITE is removed
PYTHONUSERBASE=/tmp/openclaw-task-userbase
PATH includes /tmp/openclaw-task-userbase/bin
```

This means old traces that contain bare `pip` or `pip3` commands are less
likely to fail with `command not found` under the new runtime.

Replay does not rewrite the original trace. Historical outputs recorded in the
trace remain historical outputs. The compatibility improvement applies when the
tool command is re-executed in container replay.

## What the Runtime Does Not Do

The task-container runtime deliberately does not do the following:

- It does not install `pytest`.
- It does not install benchmark project dependencies.
- It does not run repository-specific `install_config` on behalf of the agent.
- It does not install `apt` packages.
- It does not grant root to the agent just to make package installation easier.
- It does not add repo-specific or package-specific special cases.
- It does not hardcode behavior for a particular SWE-rebench task.
- It does not replace the benchmark image's original Python commands.

These boundaries matter for research integrity: benchmark tasks should be
solved under their intended image/project environment, with agent-visible
actions recorded in the trace.

## Applicability Across Benchmarks

This environment model applies to container-mode OpenClaw runs that use the
task-container agent runtime. It is most relevant to SWE-style Docker
benchmarks such as SWE-Bench Verified and SWE-rebench.

It is not universal across every benchmark:

- Host-mode benchmarks do not start task containers.
- Terminal-Bench has its own benchmark runner and environment preparation.
- A container image without Python 3.11+ cannot run this OpenClaw
  task-container runtime.
- A container image whose selected Python lacks `pip` will still fail for
  `pip` commands unless the agent or project has a valid way to bootstrap pip.

The design goal is benchmark-agnostic controller portability, not benchmark
environment augmentation.

## Troubleshooting Signals

`pip: not found` or `pip3: not found` should be rare after the shim setup. If
it appears, check whether the task-local bin directory is on `PATH`.

`No module named pip` means the command reached the selected Python, but that
Python does not provide pip. This is different from `pip: not found`.

`No module named pytest` means the project test runner is not installed in the
Python environment being used by that command. The runtime does not fix this
automatically; the agent must install the project's test dependencies or use
the project-prescribed environment.

`apt` permission errors are expected when the task user is non-root. The agent
should prefer project-level or user-level Python dependency installation when
appropriate.

If OpenClaw controller imports fail, inspect the bootstrap log and the
versioned cache marker. Controller dependency failures are runtime bootstrap
issues; project test dependency failures are task-solving issues.
