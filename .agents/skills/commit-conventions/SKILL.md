---
name: commit-conventions
description: Enforce commit message format and authorship for Chaojun Zhang's vLLM contributions. Use whenever creating or amending a git commit, before `git commit`/`git commit --amend`, and before opening or updating a pull request in this repo.
---

# Commit Message and Authorship Conventions

Apply these rules to every commit made on behalf of Chaojun Zhang in this repo (and any fork/clone of it), regardless of how the change was produced.

## Message format

Exactly three lines (plus the standard blank-line separation Git expects between subject, body, and trailers):

```
<line 1: short imperative summary of the commit's purpose>

Signed-off-by: Chaojun Zhang <chaojun.zhang@intel.com>
```

- Line 1: a single concise sentence describing *why*/*what* the commit does. No period required, no body/details beyond this unless the user explicitly asks for more.
- Line 2: blank.
- Line 3: the sign-off trailer, verbatim: `Signed-off-by: Chaojun Zhang <chaojun.zhang@intel.com>`.

Do not add a multi-paragraph body, bullet lists, or extra trailers unless the user explicitly asks for them. Keep it to the three lines above by default.

## Authorship rules

- Always sign off as the user: use `git commit -s` (or append the trailer manually) with the exact identity above.
- Never add `Co-authored-by: Copilot <...>` or any other AI-attribution trailer/line (e.g. `Generated-by:`, `Assisted-by:`). If one is already present — from a template, amend, cherry-pick, or rebase — remove it before the commit is finalized.
- If amending or rebasing existing commits that are missing the sign-off, add it (`git commit --amend -s` or `git rebase --exec 'git commit --amend --no-edit -s'`).
- Verify the final commit with `git log -1 --format='%B'` (or `git show -s --format='%B' HEAD`) and confirm: three lines, correct Signed-off-by, and no Copilot/AI trailer, before pushing or opening a PR.
