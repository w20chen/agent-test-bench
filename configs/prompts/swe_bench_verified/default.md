<pr_description>
Consider the following PR description:
{{task}}
</pr_description>

<instructions>
# Task Instructions

## Overview

You're a software engineer interacting continuously with a computer by submitting commands.
Your task is to make changes to non-test files in the current working directory to fix the
issue described in the PR description in a way that is general and consistent with the codebase.

For each response:
1. Include a THOUGHT section explaining your reasoning.
2. Provide one or more bash tool calls to execute.

## Recommended Workflow

1. Analyse the codebase by finding and reading relevant files.
2. Create a script to reproduce the issue.
3. Edit the source code to resolve the issue.
4. Verify your fix works by running your script again.
5. Test edge cases to ensure your fix is robust.

## Constraints

- MODIFY: Regular source code files in the current working directory.
- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.).

## Submission

When done, submit your changes as a git patch using SEPARATE commands:

  Step 1 – create patch:   git diff -- path/to/changed_file > patch.txt
  Step 2 – verify patch:   cat patch.txt
  Step 3 – submit (EXACT): echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt

You CANNOT continue working after submitting.
</instructions>
