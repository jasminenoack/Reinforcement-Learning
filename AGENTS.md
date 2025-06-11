# AGENTS.md

## Voice
Use a clear, professional tone. Prioritize readable, idiomatic code with minimal cleverness.

## Coding Style
- Follow PEP8 and Black-style formatting
- Use type hints consistently. Do not remove or weaken them.
- Prefer expressive variable and function names
- Avoid magic values. Use named constants where possible.

## Tests
- Do not modify or remove assertions to make tests pass.
- Always include assertions in tests to validate expected behavior.
- Prefer minimal mocking for local or internal functions.
  - If mocking is needed, use `mocker` and enable `autospec=True`.
- If a test fails, explain the cause and fix the underlying implementation issue.
- Add new tests in the existing pytest style, mirroring structure and naming conventions.
- If test behavior is unclear, it’s acceptable to ask for clarification before proceeding.

## Typing and Linting
- Fix pyright, and flake8 errors properly. Do not suppress them.
- Do not remove or ignore typing to bypass errors

## General Behavior
- Prefer small, focused changes over large rewrites.
- Make only the minimal edits needed to address the task.
- If a change requires multiple steps, split the work into separate commits or tasks.
- Do not refactor unrelated code unless explicitly requested.
- Be cautious. If unsure, leave a comment like “# FIXME: unsure about this logic”.
- Never comment out code unless asked to.
- Document non-obvious or ambiguous changes with inline comments.
- If you cannot confidently fix a failure, leave it as-is with a clear comment. Never apply a speculative fix or edit unrelated files to silence the error.

## Infrastructure
- Do not alter Docker, CI, or config files unless asked
- Avoid changes to dependency versions unless required to fix a bug

## Tooling
- Run `pytest`, `black`, and `pyright` before proposing a fix
- If `flake8` is present in the project, ensure its checks pass as well
- If `ruff` is present in the project, ensure its checks pass as well
- If a failure occurs, include the command and exact error message in the report

## Review Expectations
- Your changes will be reviewed by a human. Be transparent. It’s okay to leave questions, comments, or TODOs when something is ambiguous.

## Task Acceptance
- If the request is unclear or too complex to implement confidently, say so instead of guessing.
- Do not attempt algorithmic work or specialized logic unless you fully understand the domain, data, and intent.
- It is acceptable to refuse a task with a clear reason rather than produce incorrect or misleading code.

## Project Context
- This is an algorithmic, ML-focused project.
- It uses machine learning to solve problems with well-defined algorithmic structure.

## Known Constraints
- All code must run on CPU (no GPU dependencies).
- This code is executed interactively with a user waiting. Avoid slow operations or long runtimes.

## Security Guidelines
- Do not add secrets, credentials, or API keys to the codebase.
- Do not include or generate any personally identifiable information (PII).

## Deployment Rules
- This repo is not deployed. It is used via scripts and Jupyter notebooks.
- Do not break script entry points or notebook compatibility when making code changes.

## Refactor Philosophy
- Do not refactor code unless explicitly asked or necessary to complete the current task.
- Require a passing test suite before beginning any refactor. Do not proceed if tests are failing.
- Refuse to refactor features that lack full test coverage. Coverage must be added first.
- All refactors must preserve existing behavior. If behavior must change, treat it as a separate feature request.
- Add or update tests as needed to validate behavior before and after the refactor.
- Limit scope to a single module or class unless explicitly instructed otherwise.
- Clearly document the purpose of any non-trivial structural changes.

## PR Guidelines
- Use descriptive PR titles (e.g., `feat:`, `fix:`, `test:`).
- Include a summary of the logic, scope of changes, and test coverage.
- Include the actual test coverage percentage in the PR if available.
- Note any limitations, skipped areas, or ambiguities encountered.
- Highlight any security concerns or risky behavior introduced.
- Avoid vague phrases like “updated code” or “fixed tests” — be specific.