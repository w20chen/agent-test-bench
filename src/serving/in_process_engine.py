"""In-process vLLM engine for deep-profile mode.

This module is for the deep-profile path only — it imports vllm lazily
so the main simulator (which uses vLLM via HTTP) does not pay the
import cost or fail when vllm is not installed.
"""
from __future__ import annotations

from typing import Any


class InProcessEngine:
    """Thin wrapper around `vllm.LLM(...)` exposing the underlying
    nn.Module so module-level forward hooks can be attached.

    Call sites should:
        engine = InProcessEngine(model="...", dtype="float16", tensor_parallel_size=1)
        model_module = engine.get_model_module()
        profiler = attach_component_hooks(model_module)
        # ... run engine.generate(...) ...
        profiler.detach()
    """

    def __init__(
        self,
        *,
        model: str,
        dtype: str = "float16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 4096,
        **extra_vllm_kwargs: Any,
    ) -> None:
        try:
            from vllm import LLM  # lazy: only required in deep-profile mode
        except ImportError as exc:
            raise ImportError(
                "deep-profile mode requires the 'vllm' package; install via "
                "pip install -e .[profile] or pip install 'vllm>=0.5,<0.8'"
            ) from exc
        if tensor_parallel_size != 1:
            raise NotImplementedError(
                f"deep-profile mode currently supports tensor_parallel_size=1 only "
                f"(got {tensor_parallel_size}); see plan §Out of Scope"
            )
        self._llm = LLM(
            model=model,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            **extra_vllm_kwargs,
        )
        self.model_name = model
        self.dtype = dtype

    def get_model_module(self):  # type: ignore[no-untyped-def]
        """Walk vLLM's executor stack to retrieve the underlying nn.Module.

        Path varies slightly across vLLM versions; this tries the modern
        path first and falls back. For TP=1 the chain is:
        engine -> llm_engine -> model_executor -> driver_worker -> model_runner -> model
        """
        engine = self._llm.llm_engine
        executor = engine.model_executor
        worker = getattr(executor, "driver_worker", None) or executor.workers[0]
        runner = worker.model_runner
        return runner.model

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return self._llm.generate(*args, **kwargs)
