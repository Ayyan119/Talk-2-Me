---
name: git-commit
description: Creates a Git commit by staging all changes with git add . and committing them with a clear, descriptive message. Trigger after completing every big change, feature milestone, major refactor, or when asked by the user.
---

# Git Commit Skill

## When to Trigger
- **After Every Big Change**: Always commit after completing a significant feature step, new adapter implementation, or major refactor.
- **After Test Verification**: Ensure all relevant unit and integration tests pass before committing.
- **User Invocation**: When the user explicitly requests a commit or runs `/git-commit`.

## Workflow

### Step 1 — Check Git Status
Run `git status` to see all unstaged/staged modifications.

### Step 2 — Stage Changes
Run:
```bash
git add .
```

### Step 3 — Commit Changes
Run:
```bash
git commit -m "<user-provided-message>"
```

### Step 4 — Verify
Run:
```bash
git log -1 --oneline
```
