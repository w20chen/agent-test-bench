"""Tests for ``trace_collect.exec_classifier`` — exec tool name classification.

Covers:
* Basic command → category mapping (grep → exec-grep, pytest → exec-pytest, …)
* Preamble stripping (sudo, nice, timeout, …)
* ``python -m <module>`` redirection (python -m pytest → exec-pytest)
* Compound commands with ``&&`` / ``;`` / ``|`` (highest priority wins)
* Shell quoting awareness (``|`` inside quotes does NOT split)
* Navigation-command override (``cd /x 88 timeout 120 python3 -m pytest`` → exec-pytest)
* Pass-through for non-``exec`` tool names
"""

from __future__ import annotations

import pytest
from trace_collect.exec_classifier import (
    classify_exec_tool_name,
    classify_tool_data,
    _tokenize_segment,
    _extract_base_command,
)


# ── classify_exec_tool_name ────────────────────────────────────────────


@pytest.mark.parametrize(
    "command, expected",
    [
        # ── User's problematic pattern ────────────────────────────────
        (
            'cd /testbed && timeout 120 python3 -m pytest tests/ -v '
            '--ignore=tests/test_PlanetModel.py 2>&1 '
            '| grep -E "(PASSED|FAILED|ERROR)"',
            "exec-pytest",
        ),
        (
            'cd /testbed 88 timeout 120 python3 -m pytest tests/ -v '
            '2>&1 | grep -E "(PASSED|FAILED|ERROR)"',
            "exec-pytest",
        ),
        # ── Basic classification ──────────────────────────────────────
        ('grep -r "pattern" .', "exec-grep"),
        ("egrep -i foo bar", "exec-grep"),
        ("rg pattern", "exec-grep"),
        ("find . -name '*.py'", "exec-find"),
        ("fd py", "exec-find"),
        ("python3 -m pytest tests/ -v", "exec-pytest"),
        ("pip install requests", "exec-pip"),
        ("pip3 install requests", "exec-pip"),
        ("python3 script.py", "exec-python"),
        ("python script.py", "exec-python"),
        ("git clone https://github.com/foo/bar.git", "exec-git"),
        ("make -j4", "exec-make"),
        ("npm install express", "exec-npm"),
        ("npx jest", "exec-npm"),
        ("curl -O https://example.com/file.tar.gz", "exec-curl"),
        ("wget https://example.com/file.tar.gz", "exec-curl"),
        ("docker run -it ubuntu:latest bash", "exec-docker"),
        ("apt-get install -y python3", "exec-apt"),
        ("conda install numpy pandas", "exec-conda"),
        ("cat file.txt", "exec-cat"),
        ("head -n 10 file.txt", "exec-head"),
        ("tail -f log.txt", "exec-tail"),
        ("sed 's/foo/bar/g' file.txt", "exec-sed"),
        ("awk '{print $1}' file.txt", "exec-awk"),
        ("diff a.txt b.txt", "exec-diff"),
        ("ls -la", "exec-ls"),
        ("cd /some/path", "exec-cd"),
        ("pwd", "exec-pwd"),
        ("mkdir -p /a/b/c", "exec-mkdir"),
        ("cp a b", "exec-cp"),
        ("mv a b", "exec-mv"),
        ("rm -rf /tmp/foo", "exec-rm"),
        ("chmod +x script.sh", "exec-chmod"),
        ("touch file.txt", "exec-touch"),
        ("echo hello world", "exec-echo"),
        ("export VAR=val", "exec-export"),
        ("tar -xzf archive.tar.gz", "exec-tar"),
        ("unzip archive.zip", "exec-tar"),
        ("systemctl restart nginx", "exec-systemctl"),
        ("ps aux", "exec-ps"),
        ("kill 1234", "exec-kill"),
        ("df -h", "exec-df"),
        ("free -m", "exec-free"),
        ("mount /dev/sda1 /mnt", "exec-mount"),
        ("gcc -o foo foo.c", "exec-gcc"),
        ("clang++ -o foo foo.cpp", "exec-gcc"),
        ("true", "exec-true"),
        ("sleep 5", "exec-sleep"),
        ("date", "exec-date"),
        ("time ls", "exec-time"),
        ("man ls", "exec-man"),
        ("bash", "exec-bash"),
        ("sh script.sh", "exec-bash"),
        ("sort file.txt", "exec-sort"),
        ("uniq file.txt", "exec-uniq"),
        ("wc -l file.txt", "exec-wc"),
        ("tee out.txt", "exec-tee"),
        ("cut -d',' -f1 file.csv", "exec-cut"),
        ("xargs rm", "exec-rm"),
        ("xargs -0 grep pattern", "exec-grep"),
        # ── Preamble handling ─────────────────────────────────────────
        ("timeout 120 python3 -m pytest tests/", "exec-pytest"),
        ("timeout -k 5 120 python3 -m pytest tests/", "exec-pytest"),
        ("sudo pip install foo", "exec-pip"),
        ("nice -n 10 make -j4", "exec-make"),
        ("nohup python3 server.py", "exec-python"),
        ("sudo apt-get install -y python3", "exec-apt"),
        # ── Compound commands (&&, ;, |) ──────────────────────────────
        ("cd /app && pip install -r requirements.txt", "exec-pip"),
        ("cd build && make -j4", "exec-make"),
        ('find . -name "*.py" | xargs grep -l "TODO"', "exec-grep"),
        ("cd /x; python3 script.py", "exec-python"),
        # ── Shell quoting: | inside quotes must NOT split ─────────────
        ('grep -E "(PASS|FAIL)" log.txt', "exec-grep"),
        ("grep -E '(PASS|FAIL)' log.txt", "exec-grep"),
        ("sed 's/foo|bar/baz/g' file.txt", "exec-sed"),
        ("awk '{print $1 \"|\" $2}' file.txt", "exec-awk"),
        # ── Edge: navigation setup should not override echo ───────────
        ("echo pytest results: all good", "exec-echo"),
        ('echo "done"', "exec-echo"),
        # ── Unrecognised commands stay as exec ────────────────────────
        ("my_custom_tool --flag", "exec"),
        ("/usr/local/bin/custom-script arg1", "exec"),
        # ── env-var assignments ───────────────────────────────────────
        ("VAR=val grep pattern file", "exec-grep"),
        ("FOO=bar BAZ=qux python3 -m pytest", "exec-pytest"),
    ],
)
def test_classify_exec_tool_name(command: str, expected: str) -> None:
    result = classify_exec_tool_name("exec", {"command": command})
    assert result == expected, f"Command: {command!r}"


# ── Pass-through (non-exec tool names) ─────────────────────────────────


def test_pass_through_non_exec_tool_names() -> None:
    """Non-``exec`` tool names are returned unchanged."""
    assert classify_exec_tool_name("bash", {"command": "something"}) == "bash"
    assert classify_exec_tool_name("read", {"command": "foo"}) == "read"
    assert classify_exec_tool_name("write", {"command": "bar"}) == "write"
    assert classify_exec_tool_name("str_replace_editor", {"command": "..."}) == "str_replace_editor"


def test_pass_through_null_or_empty() -> None:
    """Null / empty args return 'exec' unchanged."""
    assert classify_exec_tool_name("exec", None) == "exec"
    assert classify_exec_tool_name("exec", {}) == "exec"
    assert classify_exec_tool_name("exec", {"command": ""}) == "exec"


def test_pass_through_non_dict_args() -> None:
    """Non-dict / non-JSON args return 'exec' unchanged."""
    assert classify_exec_tool_name("exec", "not a dict") == "exec"
    assert classify_exec_tool_name("exec", 42) == "exec"  # type: ignore[arg-type]


# ── classify_tool_data (dict helper) ───────────────────────────────────


def test_classify_tool_data_updates_exec() -> None:
    data = {"tool_name": "exec", "tool_args": {"command": "grep foo"}}
    result = classify_tool_data(data)
    assert result["tool_name"] == "exec-grep"


def test_classify_tool_data_passes_through_non_exec() -> None:
    data = {"tool_name": "bash", "tool_args": {"command": "something"}}
    result = classify_tool_data(data)
    assert result is data  # same object, unchanged


def test_classify_tool_data_does_not_mutate_original() -> None:
    data = {"tool_name": "exec", "tool_args": {"command": "grep foo"}}
    result = classify_tool_data(data)
    assert result is not data  # shallow copy
    assert data["tool_name"] == "exec"  # original untouched


# ── _tokenize_segment ──────────────────────────────────────────────────


def test_tokenize_simple() -> None:
    assert _tokenize_segment("grep -r pattern .") == "grep"


def test_tokenize_with_env() -> None:
    assert _tokenize_segment("VAR=val grep pattern") == "grep"


def test_tokenize_with_preamble() -> None:
    assert _tokenize_segment("timeout 120 python3 script.py") == "python3"


def test_tokenize_python_m_pytest() -> None:
    assert _tokenize_segment("python3 -m pytest tests/") == "pytest"


def test_tokenize_python_m_unknown() -> None:
    """Unknown ``-m`` modules stay as python, not redirected."""
    assert _tokenize_segment("python3 -m http.server 8080") == "python3"


def test_tokenize_cd_navigation_override() -> None:
    """cd followed by a higher-priority action → prefer the action."""
    assert _tokenize_segment("cd /testbed 88 timeout 120 python3 -m pytest") == "pytest"


def test_tokenize_cd_only() -> None:
    """cd alone should stay cd."""
    assert _tokenize_segment("cd /some/path") == "cd"


def test_tokenize_echo_not_overridden() -> None:
    """echo should not be overridden by its arguments."""
    assert _tokenize_segment("echo pytest results") == "echo"


def test_tokenize_command_builtin() -> None:
    assert _tokenize_segment("command grep pattern") == "grep"


def test_tokenize_xargs_plumbing() -> None:
    assert _tokenize_segment("xargs -0 rm") == "rm"


def test_tokenize_path_prefix() -> None:
    assert _tokenize_segment("/usr/bin/grep pattern") == "grep"


# ── _extract_base_command ──────────────────────────────────────────────


def test_extract_simple() -> None:
    assert _extract_base_command("grep pattern") == "grep"


def test_extract_compound_and() -> None:
    assert _extract_base_command("cd /x && python3 -m pytest") == "pytest"


def test_extract_compound_semicolon() -> None:
    assert _extract_base_command("cd /x; python3 script.py") == "python3"


def test_extract_pipe_rightmost_wins() -> None:
    assert _extract_base_command("find . | xargs grep TODO") == "grep"


def test_extract_quoted_pipe_not_split() -> None:
    """| inside double quotes must not split the command."""
    assert _extract_base_command('grep -E "(PASS|FAIL)" log.txt') == "grep"


def test_extract_quoted_and_not_split() -> None:
    """&& inside quotes must not split."""
    assert _extract_base_command('echo "a && b"') == "echo"


def test_extract_quoted_semicolon_not_split() -> None:
    """; inside quotes must not split."""
    assert _extract_base_command("echo 'a;b'") == "echo"


def test_extract_empty() -> None:
    assert _extract_base_command("") == "exec"


def test_extract_none() -> None:
    assert _extract_base_command(None) == "exec"  # type: ignore[arg-type]
