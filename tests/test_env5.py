from __future__ import annotations

from harness.scheduler_hooks import (
    build_report,
    parse_eviction_events,
    parse_prometheus_metrics,
)


def test_parse_prometheus_metrics_extracts_preemption_fields() -> None:
    metrics_payload = "\n".join(
        [
            "vllm:num_preemptions_total 3",
            "vllm:gpu_cache_usage_perc 75.0",
            "vllm:cpu_cache_usage_perc 0.0",
            "vllm:gpu_prefix_cache_hit_rate 0.5",
            "vllm:cpu_prefix_cache_hit_rate 0.0",
        ]
    )
    snapshot = parse_prometheus_metrics(metrics_payload)
    assert snapshot.num_preemptions_total == 3.0
    assert snapshot.gpu_cache_usage_perc == 75.0
    assert snapshot.gpu_prefix_cache_hit_rate == 0.5


def test_parse_eviction_events_extracts_structured_records() -> None:
    events = parse_eviction_events(
        "INFO EVICT seq_id=req-1 tokens=128 reason=pressure gpu_usage=0.92\n"
    )
    assert len(events) == 1
    assert events[0].seq_id == "req-1"
    assert events[0].tokens == 128


def test_build_report_marks_evidence_scope_and_runtime_confirmation() -> None:
    report = build_report(
        metrics_payload="vllm:num_preemptions_total 1\n",
        log_text="INFO EVICT seq_id=req-1 tokens=128 reason=pressure gpu_usage=0.92\n",
    )
    assert report["metrics_fetch_ok"] is True
    assert report["scheduler_log_provided"] is True
    assert report["scheduler_hook_runtime_confirmed"] is True
    assert report["evidence_scope"] == "current_run_log"
