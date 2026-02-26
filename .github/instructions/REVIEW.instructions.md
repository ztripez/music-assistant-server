---
applyTo: "**"
excludeAgent: "code-agent" # Only used by the code review agent in Github
---

# PR Review Standards

## What to Analyze

Review all code changes for:
- Code quality and style consistency with the existing codebase
- Potential bugs or issues
- Performance implications
- Security concerns
- Test coverage
- Documentation updates if needed

## PR Title

The PR title must be a functional description of the change. It must NOT contain conventional commit prefixes such as `feat:`, `fix:`, `refactor:`, `chore:`, etc. Labels are used to categorize PRs, not the title. Flag as `[PROBLEM]` if the title uses such prefixes.

## Existing Review Comments

Ensure any existing review comments on the PR have been addressed before approving.

## Issue Categories

Categorize every issue found as one of:
- `[CRITICAL]` — must be fixed before merging (bugs, security issues, broken functionality)
- `[PROBLEM]` — should be fixed (code quality, bad patterns, missing tests)

## Output

- Post inline comments on GitHub for every `[CRITICAL]` and `[PROBLEM]` issue found.
- Do NOT post `[SUGGESTION]` items to GitHub.
- Do not list things that are already correct
