---
applyTo: "**"
excludeAgent: "code-agent" # Only used by the code review agent in Github
---

# PR Review Standards

When reviewing a pull request, apply the standards defined in:
https://raw.githubusercontent.com/music-assistant/server/refs/heads/dev/.claude/skills/review-pr/REVIEW_STANDARDS.md

## Output

- Post inline comments on GitHub for every `[CRITICAL]` and `[PROBLEM]` issue found.
- Do NOT post `[SUGGESTION]` items to GitHub.
- Do not list things that are already correct
