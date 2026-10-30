"""The training snapshot must reproduce the worktree that actually ran."""

import json
import os
import subprocess
from pathlib import Path

from diffphys.git_snapshot import capture_git_snapshot
from diffphys.profiles import get_profile
from main_cuda import build_parser, format_banner


def git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout


def object_names(repo):
    root = repo / ".git" / "objects"
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())


def worktree(root):
    files = {}
    for path in root.rglob("*"):
        if ".git" in path.parts:
            continue
        if path.is_symlink():
            files[path.relative_to(root).as_posix()] = ("symlink", os.readlink(path))
        elif path.is_file():
            files[path.relative_to(root).as_posix()] = ("file", path.read_bytes())
    return files


def init_repo(path):
    path.mkdir()
    git(path, "init", "-q", "-b", "main")
    git(path, "config", "user.email", "snapshot@example.com")
    git(path, "config", "user.name", "snapshot")
    (path / "tracked.txt").write_text("v1\n")
    (path / "staged.txt").write_text("old\n")
    (path / "gone.txt").write_text("bye\n")
    (path / "keep.txt").write_text("same\n")
    (path / "binary.bin").write_bytes(b"\x00\x01")
    script = path / "run.sh"
    script.write_text("#!/bin/sh\necho hi\n")
    script.chmod(0o755)
    git(path, "add", "-A")
    git(path, "commit", "-q", "-m", "base")


def dirty_worktree(repo):
    (repo / "tracked.txt").write_text("v2\n")
    (repo / "staged.txt").write_text("index only\n")
    git(repo, "add", "--", "staged.txt")
    (repo / "staged.txt").write_text("worktree version\n")
    (repo / "gone.txt").unlink()
    os.rename(repo / "keep.txt", repo / "moved.txt")
    (repo / "new.txt").write_text("untracked\n")
    (repo / "sub").mkdir()
    (repo / "sub" / "nested.txt").write_text("nested\n")
    (repo / "space name.txt").write_text("space\n")
    (repo / "empty.txt").write_bytes(b"")
    (repo / "extra.bin").write_bytes(b"\xff\x00")
    tool = repo / "tools"
    tool.mkdir()
    script = tool / "run.sh"
    script.write_text("#!/bin/sh\necho tool\n")
    script.chmod(0o755)
    (repo / "link.txt").symlink_to("tracked.txt")
    (repo / "ignored.log").write_text("secret-ignored\n")
    (repo / ".gitignore").write_text("ignored.log\n")


def restore(state, source, patch, dest):
    git(dest.parent, "clone", "-q", str(source), str(dest))
    for command in state["restore"]:
        argv = command.split()
        if argv[1] == "apply":
            argv[-1] = str(patch)
        git(dest, *argv[1:])


def test_snapshot_restores_dirty_worktree_without_touching_git(tmp_path):
    repo = tmp_path / "repo"
    init_repo(repo)
    dirty_worktree(repo)
    staged_blob = git(repo, "rev-parse", ":staged.txt")
    cached = git(repo, "diff", "--cached")
    objects = object_names(repo)
    original = worktree(repo)

    state = capture_git_snapshot(tmp_path / "run", repo)

    assert git(repo, "rev-parse", ":staged.txt") == staged_blob
    assert git(repo, "diff", "--cached") == cached
    assert object_names(repo) == objects
    assert worktree(repo) == original
    assert state["commit"] == git(repo, "rev-parse", "HEAD").decode().strip()
    assert state["dirty"] is True
    assert b"index only" not in (tmp_path / "run" / "git_diff.patch").read_bytes()
    assert b"secret-ignored" not in (tmp_path / "run" / "git_diff.patch").read_bytes()
    saved = json.loads((tmp_path / "run" / "git_state.json").read_text())
    assert saved == state

    restored = tmp_path / "restored"
    restore(state, repo, tmp_path / "run" / "git_diff.patch", restored)
    expected = dict(original)
    expected.pop("ignored.log")
    assert worktree(restored) == expected
    assert os.access(restored / "tools" / "run.sh", os.X_OK)


def test_clean_tree_records_the_commit_and_an_empty_patch(tmp_path):
    repo = tmp_path / "repo"
    init_repo(repo)
    state = capture_git_snapshot(tmp_path / "run", repo)
    assert state["dirty"] is False
    assert state["restore"] == [f"git checkout {state['commit']}"]
    assert (tmp_path / "run" / "git_diff.patch").read_bytes() == b""
    restored = tmp_path / "restored"
    restore(state, repo, tmp_path / "run" / "git_diff.patch", restored)
    assert worktree(restored) == worktree(repo)


def test_banner_shows_the_recorded_commit():
    args = build_parser().parse_args(["--profile", "vanilla"])
    text = format_banner(
        get_profile("vanilla"),
        args,
        "/tmp/log",
        0,
        10,
        {"commit": "abcdef1234567890", "dirty": True},
    )
    assert "abcdef123456 (dirty)" in text
