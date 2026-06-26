# Project Agent Rules

## Source of truth
- Read docs/SPEC.md, docs/ARCHITECTURE.md, docs/ACCEPTANCE_CRITERIA.md,
  and task/todo.md before changing code.
- Implement only one unchecked task or tightly related group of tasks at a time.

## Implementation loop
For every task:
1. Inspect the relevant existing code.
2. Add or update tests before or together with implementation.
3. Implement the smallest correct solution.
4. Run formatting, linting, type checks, unit tests, build, and relevant E2E tests.
5. If any command fails, investigate the root cause and fix it.
6. Repeat until all required checks pass.
7. Review the git diff for regressions, security issues, dead code, hard-coded secrets,
   broken error handling, and unmet acceptance criteria.
8. Update task/todo.md only after verification passes.

## Safety
- Never expose secrets, API keys, .env contents, or credentials.
- Never delete unrelated files.
- Do not change deployment, production databases, git remote settings, or CI secrets.
- Do not mark a task complete when tests are skipped or failing.
- Report failures honestly, including exact failing command and root cause.

## Git policy
- Only run git add, git commit, or git push when the user's current request explicitly asks for it.
- Before committing, inspect git status and git diff --check, then run relevant tests.
- Stage only task-related files; never use `git add .` unless explicitly asked.
- Never force-push, amend, reset, clean, delete branches, switch branches, or change remotes without explicit approval.
- Push only the current branch to origin.