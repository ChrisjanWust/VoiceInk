#!/bin/bash
# Claude Code SessionStart hook. Fallback for linked worktrees that never ran
# scripts/setup_worktree: Claude Code's own worktrees under .claude/worktrees/,
# or a T3 Code worktree made before the project's setup script was configured.
#
# No-op in the main checkout and once setup has run in a worktree, so it is
# cheap on every session start, resume and compaction.

START_DIR="${CLAUDE_PROJECT_DIR:-$PWD}"
GIT_DIR="$(git -C "$START_DIR" rev-parse --path-format=absolute --git-dir 2>/dev/null)" || exit 0
COMMON_DIR="$(git -C "$START_DIR" rev-parse --path-format=absolute --git-common-dir)"
# Only linked worktrees have a git dir separate from the common dir
[ "$GIT_DIR" = "$COMMON_DIR" ] && exit 0
[ -e "$GIT_DIR/voiceink-setup-worktree-done" ] && exit 0

WORKTREE_PATH="$(git -C "$START_DIR" rev-parse --show-toplevel)"
[ -x "$WORKTREE_PATH/scripts/setup_worktree" ] || exit 0

# The common git dir always lives in the main checkout
"$WORKTREE_PATH/scripts/setup_worktree" "$(dirname "$COMMON_DIR")" "$WORKTREE_PATH" >&2 || exit 0

# A rule file written during this hook may not load until the next session,
# so put it in this session's context too.
echo "scripts/setup_worktree just set up this worktree. Its build rules, also saved to .claude/rules/worktree.md:"
cat "$WORKTREE_PATH/.claude/rules/worktree.md"
