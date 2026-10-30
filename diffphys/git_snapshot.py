"""Record the commit and worktree patch needed to reproduce a training run."""

import json
import os
import subprocess
from pathlib import Path

PATCH_NAME = "git_diff.patch"
STATE_NAME = "git_state.json"


def capture_git_snapshot(dest_dir, repo_dir=None):
    """Write ``git_state.json`` and ``git_diff.patch`` under ``dest_dir``.

    The patch is the worktree relative to HEAD, including non-ignored
    untracked files. Ignored files are omitted. Applying it reproduces file
    contents; it does not recreate which of those changes were staged.

    From the repository root, with this run's patch:

        git checkout <commit>
        git apply --binary git_diff.patch
    """
    start = Path(__file__).resolve().parents[1] if repo_dir is None else Path(repo_dir)
    repo = _toplevel(start)
    commit = _text(repo, ["rev-parse", "HEAD"]).strip()
    branch = _text(repo, ["rev-parse", "--abbrev-ref", "HEAD"]).strip()
    patch = _worktree_patch(repo)
    restore = [f"git checkout {commit}"]
    if patch:
        restore.append(f"git apply --binary {PATCH_NAME}")
    state = {
        "commit": commit,
        "branch": branch,
        "dirty": bool(patch),
        "patch": PATCH_NAME,
        "status": _text(repo, ["status", "--porcelain=v1", "-uall"]).rstrip("\n"),
        "restore": restore,
    }
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / PATCH_NAME).write_bytes(patch)
    (dest / STATE_NAME).write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return state


def _worktree_patch(repo):
    # `git diff HEAD` is every tracked change actually on disk. Untracked
    # files are appended with `--no-index`, which does not stage them or
    # write blobs into .git.
    parts = [_git(repo, ["diff", "HEAD", "--binary", "--full-index", "--no-renames"], ok=(0,))]
    listing = _git(repo, ["ls-files", "--others", "--exclude-standard", "-z"], ok=(0,))
    for raw in listing.split(b"\0"):
        if not raw:
            continue
        parts.append(
            _git(
                repo,
                ["diff", "--no-index", "--binary", "--full-index", "--", "/dev/null", os.fsdecode(raw)],
                ok=(0, 1),
            )
        )
    return _concat(parts)


def _concat(parts):
    out = bytearray()
    for part in parts:
        if not part:
            continue
        if out and not out.endswith(b"\n"):
            out += b"\n"
        out += part
    return bytes(out)


def _toplevel(start):
    return Path(_text(start, ["rev-parse", "--show-toplevel"]).strip())


def _text(repo, args):
    return _git(repo, args, ok=(0,)).decode()


def _git(repo, args, ok):
    proc = subprocess.run(
        ["git", "-c", "core.quotepath=false", *args],
        cwd=repo,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        capture_output=True,
    )
    if proc.returncode not in ok:
        detail = proc.stderr.decode(errors="replace").strip() or "git failed"
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return proc.stdout
