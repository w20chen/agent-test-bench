from __future__ import annotations

import argparse
import asyncio
import time
from typing import Any

import httpx

from serving._common import (
    append_followup_turn,
    validate_chat_responses,
    write_json_report,
)
from serving.health_check import run_chat_smoke, wait_for_models_endpoint


async def fetch_json(client: httpx.AsyncClient, url: str) -> dict[str, Any]:
    """GET a JSON endpoint and return the parsed body."""
    response = await client.get(url)
    response.raise_for_status()
    return response.json()


async def verify_proxy(args: argparse.Namespace) -> dict[str, Any]:
    """Run ThunderAgent-specific program tracking checks."""
    timeout = httpx.Timeout(args.timeout_s)
    async with httpx.AsyncClient(timeout=timeout) as client:
        models_payload = await wait_for_models_endpoint(
            client=client,
            api_base=args.api_base,
            timeout_s=args.timeout_s,
            poll_interval_s=args.poll_interval_s,
        )
        models = models_payload.get("data") or []
        if not models:
            raise ValueError("no models available through ThunderAgent")
        resolved_model = models[0]["id"]
        pre_programs_payload = await fetch_json(client, f"{args.base_url}/programs")

        messages = [{"role": "user", "content": args.prompt}]
        chat_responses = []
        for index in range(2):
            response = await run_chat_smoke(
                client=client,
                api_base=args.api_base,
                model=resolved_model,
                messages=messages,
                program_id=args.program_id,
            )
            chat_responses.append(response)
            if index == 0:
                messages = append_followup_turn(
                    messages,
                    chat_response=response,
                    followup_prompt=args.followup_prompt,
                )

        programs_payload = await fetch_json(client, f"{args.base_url}/programs")
        profile_payload = await fetch_json(
            client, f"{args.base_url}/profiles/{args.program_id}"
        )
        metrics_payload = await fetch_json(client, f"{args.base_url}/metrics")

    return {
        "timestamp": int(time.time()),
        "api_base": args.api_base,
        "base_url": args.base_url,
        "program_id": args.program_id,
        "resolved_model": resolved_model,
        "models_response": models_payload,
        "pre_programs_response": pre_programs_payload,
        "chat_responses": chat_responses,
        "programs_response": programs_payload,
        "profile_response": profile_payload,
        "metrics_response": metrics_payload,
    }


def validate_report(report: dict[str, Any]) -> list[str]:
    """Validate ThunderAgent proxy state tracking signals."""
    errors: list[str] = []
    if not report["models_response"].get("data"):
        errors.append("/v1/models returned an empty model list")
    if report["program_id"] not in report["programs_response"]:
        errors.append("program_id was not tracked in /programs")
    if not report["profile_response"]:
        errors.append("profile endpoint returned an empty payload")
    pre_program = (report.get("pre_programs_response") or {}).get(
        report["program_id"], {}
    )
    post_program = report["programs_response"].get(report["program_id"], {})
    step_delta = int(post_program.get("step_count", 0)) - int(
        pre_program.get("step_count", 0)
    )
    if step_delta < 2:
        errors.append(
            "program step_count did not increase by at least 2 during this run"
        )
    metrics_response = report["metrics_response"]
    if not metrics_response.get("metrics_enabled"):
        errors.append("ThunderAgent /metrics reported metrics disabled")
    if not metrics_response.get("backends"):
        errors.append("ThunderAgent /metrics returned no backend state")
    return [*errors, *validate_chat_responses(report["chat_responses"])]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify ThunderAgent proxy program tracking and profiling."
    )
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--program-id", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--followup-prompt", required=True)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--poll-interval-s", type=float, default=2.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = asyncio.run(verify_proxy(args))
    write_json_report(report, args.output)
    if args.fail_on_mismatch:
        errors = validate_report(report)
        if errors:
            raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
