from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

def resolve_python_sibling_executable(name: str) -> str:
    return str(Path(sys.executable).with_name(name))

def print_or_exec_command(
    *,
    config: Any,
    command: list[str],
    print_only: bool,
) -> None:
    if print_only:
        print(json.dumps({"config": asdict(config), "command": command}, indent=2))
        return
    os.execvpe(command[0], command, os.environ)

def write_json_report(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

def first_choice_content(chat_response: dict[str, Any]) -> str:
    return (
        (((chat_response.get("choices") or [{}])[0].get("message") or {}).get("content"))
        or ""
    )

def append_followup_turn(
    messages: list[dict[str, str]],
    *,
    chat_response: dict[str, Any],
    followup_prompt: str,
) -> list[dict[str, str]]:
    return [
        *messages,
        {"role": "assistant", "content": first_choice_content(chat_response)},
        {"role": "user", "content": followup_prompt},
    ]

def validate_chat_responses(chat_responses: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for index, chat_response in enumerate(chat_responses):
        choices = chat_response.get("choices") or []
        if not choices:
            errors.append(f"chat completion #{index} returned no choices")
            continue
        if not first_choice_content(chat_response):
            errors.append(f"chat completion #{index} returned empty content")
    return errors
