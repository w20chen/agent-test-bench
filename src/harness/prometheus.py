from __future__ import annotations


def parse_prometheus_metric_values(
    metrics_payload: str,
    metric_names: dict[str, str],
    *,
    include_missing: bool,
) -> dict[str, float | None]:
    """Extract numeric metric values keyed by their caller-provided aliases.

    Args:
        metrics_payload: Raw Prometheus text-format response body.
        metric_names: Mapping of Prometheus metric name → result alias.
            **No key may be a prefix of another key** — the inner loop
            breaks on the first match per line, so a prefix would shadow
            any longer metric name that starts with the same string.
        include_missing: When True, aliases for unseen metrics are set to
            None. When False, missing metrics are omitted from the result.
    """
    values: dict[str, float | None] = {}
    if include_missing:
        values = {alias: None for alias in metric_names.values()}

    for line in metrics_payload.splitlines():
        if not line or line.startswith("#"):
            continue
        for metric_name, alias in metric_names.items():
            if line.startswith(metric_name):
                values[alias] = float(line.split()[-1])
                break
    return values
