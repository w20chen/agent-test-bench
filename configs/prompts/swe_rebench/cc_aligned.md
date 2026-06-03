Fix this issue: {{task}}

IMPORTANT: You must complete the FULL workflow:
1. Read and understand the issue thoroughly
2. Explore the codebase to find relevant files
3. Implement the fix
4. Run the test suite to verify your fix
5. If ANY test fails, analyze the error and fix it
6. Repeat steps 4-5 until ALL tests pass
7. Only stop when tests are passing

DO NOT stop until you have:
- Made code changes that fix the issue
- Run the tests and confirmed they pass
- Shown the final git diff

If you encounter test failures, debug and fix them. Keep trying until successful.

CRITICAL REQUIREMENTS FOR TESTING:
- You MUST run the project's ORIGINAL test suite (pytest, unittest, tox, etc.)
- Do NOT write custom test scripts or verification scripts to bypass tests
- Do NOT claim success based on your own "All checks passed" output
- The test output MUST show real pytest format: "X passed, Y failed in Z seconds"
- If tests fail with ImportError or collection errors, fix the environment/import issue first
- Success means the project's actual test suite passes, not custom verification

WHAT COUNTS AS SUCCESS:
- Real pytest/unittest output showing tests passed
- Example: "===== 150 passed, 0 failed in 10.5s ====="

WHAT DOES NOT COUNT:
- Your own verification scripts saying "All checks passed"
- Manual testing or print statements
- Skipping tests due to import errors

In the output, you need to summary your change and
summary how your test the application to check the fix,
and what's the test status.

When you are done, submit your changes as a git patch using these SEPARATE commands:

  Step 1 – create patch:   git diff -- path/to/changed_file > patch.txt
  Step 2 – verify patch:   cat patch.txt
  Step 3 – submit (EXACT): echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt

You CANNOT continue working after submitting.
