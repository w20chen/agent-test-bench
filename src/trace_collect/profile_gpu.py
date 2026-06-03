"""profile-gpu subcommand entry: replay a trace's LLM calls through an in-process vLLM,
recording per-step attn/mlp memory breakdowns."""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

from harness.component_memory_profiler import attach_component_hooks
# in_process_engine is imported lazily inside main() so importing this
# module on machines without vllm doesn't crash.


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    from serving.in_process_engine import InProcessEngine  # lazy

    source_actions = _load_source_trace(args.source_trace)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"profile_gpu_{int(time.time())}.jsonl"

    engine = InProcessEngine(model=args.model, dtype=args.dtype, max_model_len=args.max_model_len)
    profiler = attach_component_hooks(engine.get_model_module())

    try:
        with open(output_path, "w", encoding="utf-8") as fout:
            fout.write(json.dumps({
                "type": "trace_metadata",
                "trace_format_version": 5,
                "scaffold": "profile-gpu",
                "model": args.model,
                "source_trace": str(args.source_trace),
            }) + "\n")
            i = 0
            for action in source_actions:
                if action.get("action_type") != "llm_call":
                    continue
                if i >= args.max_iterations:
                    break
                messages_in = action.get("data", {}).get("messages_in")
                completion_tokens = action.get("data", {}).get("completion_tokens", 64)
                if not messages_in:
                    continue
                ts_start = time.time()
                prompt = _messages_to_prompt(messages_in)
                _ = engine.generate([prompt], sampling_params=_sampling_params(completion_tokens))
                # record_step() runs inside the timed region. profile-gpu is
                # explicitly NOT a timing benchmark (hook overhead pollutes
                # latency); ts_end here is for ordering only. If you swap the
                # measurement callback for peak-tracking with reset_peak_*,
                # consider moving record_step() outside the timed region.
                profiler.record_step()
                ts_end = time.time()
                step = profiler.steps[-1]
                rec = {
                    "type": "action",
                    "action_type": "llm_call",
                    "action_id": f"profile_{i}",
                    "iteration": i,
                    "ts_start": ts_start,
                    "ts_end": ts_end,
                    "data": {
                        "sim_metrics": {
                            "gpu_component_breakdown": dataclasses.asdict(step),
                            "warmup": False,
                        },
                    },
                }
                fout.write(json.dumps(rec) + "\n")
                i += 1
    finally:
        profiler.detach()
    print(f"profile-gpu wrote {output_path}")
    return 0


def _load_source_trace(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(c) for c in content)
        parts.append(f"{role}: {content}")
    return "\n".join(parts) + "\nassistant:"


def _sampling_params(max_tokens: int) -> Any:
    from vllm import SamplingParams
    return SamplingParams(max_tokens=max_tokens, temperature=0.0)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="profile-gpu: in-process deep-profile of attn/mlp memory")
    p.add_argument("--source-trace", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-iterations", type=int, default=5)
    p.add_argument("--output-dir", default="traces/profile_gpu")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
