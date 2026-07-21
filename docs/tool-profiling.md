# Tool Profiler and Tool Scheduler

This project has two prototype wrappers for studying individual tool
invocations inside an agent run:

- `tool_profiler`
- `tool_scheduler`

They are both opt-in. In normal trace collection, choose one with
`--tool-profiling`; they are not meant to be enabled together by default.

## `tool_profiler`

`tool_profiler` answers: **what resources did this tool invocation use?**

It wraps a command, samples the command's process tree while it runs, and writes
a JSONL profile. The profile records CPU time, effective core usage, process and
thread counts, RSS, I/O counters, context switches, page faults, runtime, exit
code, and weak behavior labels such as CPU-serial, CPU-parallel, I/O-active, or
mixed.

Use it when the goal is to understand tool behavior:

```bash
python -m prototype.tool_profiler --output profiles.jsonl -- pytest -q
```

For existing shell command strings, use `--shell-command`:

```bash
python -m prototype.tool_profiler \
  --output profiles.jsonl \
  --shell-command -- "pytest -q && echo done"
```

In `trace_collect`, select it with:

```bash
--tool-profiling tool_profiler
```

The profiler does not make placement decisions and does not apply CPU affinity.
It is a measurement tool.

## `tool_scheduler`

`tool_scheduler` answers: **given this tool's observed demand and the hardware
topology, where might it run better?**

It also wraps and monitors a command, but its main output is a sequence of
dry-run scheduling decisions. It predicts CPU core demand online, discovers
NUMA/LLC topology, checks allowed CPUs, estimates candidate placement costs, and
records whether the current placement should be kept or whether a move would be
recommended.

Use it when the goal is to study hardware-aware placement policy:

```bash
python -m prototype.tool_scheduler --output scheduler.jsonl --dry-run -- pytest -q
```

For existing shell command strings:

```bash
python -m prototype.tool_scheduler \
  --output scheduler.jsonl \
  --dry-run \
  --shell-command -- "pytest -q && echo done"
```

In `trace_collect`, select it with:

```bash
--tool-profiling tool_scheduler
```

The scheduler is currently a recommender only. It records `keep` or
`recommend_move` decisions, but it does not migrate the process or enforce CPU
affinity.

## Choosing Between Them

Use `tool_profiler` when you need a resource profile for analysis.

Use `tool_scheduler` when you need online placement recommendations for a tool
scheduling experiment.

Both wrappers preserve the wrapped command's stdout/stderr for the agent and
write wrapper diagnostics under their own `[tool-profiler]` or
`[tool-scheduler]` prefixes.
