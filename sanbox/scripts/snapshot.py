#!/usr/bin/env python3
"""Create and verify immutable research experiment source snapshots."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Optional, Set, Tuple


EXACT_SOURCE_FILES = (
    "main.py",
    "requirements.txt",
    "generative_recommenders/__init__.py",
)
SOURCE_DIRECTORIES = (
    "configs",
    "generative_recommenders/research",
    "scripts",
    "tests",
)
SOURCE_SUFFIXES = {
    ".gin",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
IGNORED_DIRECTORY_NAMES = {"__pycache__", ".pytest_cache"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}
MANIFEST_FILE = "SOURCE_SHA256SUMS"
MANIFEST_ID_FILE = "SOURCE_MANIFEST_SHA256"
COMMIT_FILE = "SOURCE_COMMIT"
TREE_FILE = "SOURCE_TREE"
HEX_SHA256 = re.compile(r"[0-9a-f]{64}")


class SnapshotError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_relative_path(path: str) -> None:
    parsed = PurePosixPath(path)
    if (
        not path
        or parsed.is_absolute()
        or ".." in parsed.parts
        or "\n" in path
        or "\r" in path
        or "\\" in path
    ):
        raise SnapshotError(f"unsafe snapshot path: {path!r}")


def _scan_tree(root: Path) -> Tuple[Set[str], Set[str]]:
    if root.is_symlink() or not root.is_dir():
        raise SnapshotError(f"snapshot root must be a non-symlink directory: {root}")

    files: Set[str] = set()
    directories: Set[str] = set()
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        dirnames.sort()
        filenames.sort()
        for name in dirnames:
            path = current_path / name
            mode = path.lstat().st_mode
            relative = path.relative_to(root).as_posix()
            _validate_relative_path(relative)
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise SnapshotError(
                    f"snapshot contains a symlink or special node: {relative}"
                )
            directories.add(relative)
        for name in filenames:
            path = current_path / name
            mode = path.lstat().st_mode
            relative = path.relative_to(root).as_posix()
            _validate_relative_path(relative)
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise SnapshotError(
                    f"snapshot contains a symlink or special node: {relative}"
                )
            files.add(relative)
    return files, directories


def _iter_source_files(source_root: Path) -> Iterable[Tuple[Path, str]]:
    selected: Dict[str, Path] = {}

    for relative in EXACT_SOURCE_FILES:
        path = source_root / relative
        if path.is_symlink() or not path.is_file():
            raise SnapshotError(f"required regular source file is missing: {relative}")
        selected[relative] = path

    for relative_directory in SOURCE_DIRECTORIES:
        directory = source_root / relative_directory
        if directory.is_symlink() or not directory.is_dir():
            raise SnapshotError(
                f"required source directory is missing: {relative_directory}"
            )
        files, _ = _scan_tree(directory)
        for relative in files:
            source_path = directory / relative
            path_parts = PurePosixPath(relative).parts
            if any(part in IGNORED_DIRECTORY_NAMES for part in path_parts):
                continue
            if source_path.suffix in IGNORED_SUFFIXES:
                continue
            if source_path.suffix not in SOURCE_SUFFIXES:
                continue
            destination_relative = (
                PurePosixPath(relative_directory) / relative
            ).as_posix()
            selected[destination_relative] = source_path

    for relative in sorted(selected):
        yield selected[relative], relative


def _run_git(source_root: Path, args: Iterable[str]) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SnapshotError(f"could not determine Git provenance: {error}") from error
    return result.stdout.strip()


def _git_provenance(source_root: Path) -> Tuple[str, str]:
    repository_root = Path(
        _run_git(source_root, ["rev-parse", "--show-toplevel"])
    ).resolve()
    try:
        relative_source = source_root.resolve().relative_to(repository_root)
    except ValueError as error:
        raise SnapshotError("source root is outside its Git repository") from error

    status_args = ["status", "--porcelain", "--untracked-files=all", "--"]
    if relative_source.parts:
        status_args.append(relative_source.as_posix())
    if _run_git(repository_root, status_args):
        raise SnapshotError(
            "source root has uncommitted changes; commit them before snapshotting"
        )

    commit_id = _run_git(repository_root, ["rev-parse", "HEAD"])
    tree_expression = (
        f"HEAD:{relative_source.as_posix()}" if relative_source.parts else "HEAD^{tree}"
    )
    tree_id = _run_git(repository_root, ["rev-parse", tree_expression])
    return commit_id, tree_id


def _manifest_entries(snapshot_root: Path) -> Dict[str, str]:
    manifest_path = snapshot_root / MANIFEST_FILE
    try:
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise SnapshotError(f"could not read {MANIFEST_FILE}: {error}") from error

    entries: Dict[str, str] = {}
    for line in lines:
        if "  " not in line:
            raise SnapshotError(f"malformed manifest line: {line!r}")
        digest, relative = line.split("  ", 1)
        _validate_relative_path(relative)
        if HEX_SHA256.fullmatch(digest) is None:
            raise SnapshotError(f"invalid SHA-256 in manifest: {digest!r}")
        if relative in entries:
            raise SnapshotError(f"duplicate manifest entry: {relative}")
        entries[relative] = digest
    return entries


def _expected_directories(files: Iterable[str]) -> Set[str]:
    directories: Set[str] = set()
    for relative in files:
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return directories


def verify_snapshot(
    snapshot_root: Path, expected_manifest: Optional[str] = None
) -> Dict[str, str]:
    snapshot_root = snapshot_root.expanduser().absolute()
    files, directories = _scan_tree(snapshot_root)
    required_metadata = {MANIFEST_FILE, MANIFEST_ID_FILE, COMMIT_FILE, TREE_FILE}
    missing = required_metadata - files
    if missing:
        raise SnapshotError(f"snapshot metadata is missing: {sorted(missing)}")

    manifest_id = (snapshot_root / MANIFEST_ID_FILE).read_text(encoding="utf-8").strip()
    if HEX_SHA256.fullmatch(manifest_id) is None:
        raise SnapshotError(f"invalid {MANIFEST_ID_FILE}")
    actual_manifest_id = _sha256(snapshot_root / MANIFEST_FILE)
    if actual_manifest_id != manifest_id:
        raise SnapshotError("snapshot manifest checksum mismatch")
    if expected_manifest is not None and actual_manifest_id != expected_manifest:
        raise SnapshotError("snapshot does not match the externally pinned manifest")

    entries = _manifest_entries(snapshot_root)
    expected_files = files - {MANIFEST_FILE, MANIFEST_ID_FILE}
    if set(entries) != expected_files:
        added = sorted(expected_files - set(entries))
        missing_entries = sorted(set(entries) - expected_files)
        raise SnapshotError(
            "snapshot file inventory mismatch: "
            f"unlisted={added}, absent={missing_entries}"
        )
    if directories != _expected_directories(files):
        raise SnapshotError("snapshot directory inventory mismatch")

    for relative, expected_digest in entries.items():
        if _sha256(snapshot_root / relative) != expected_digest:
            raise SnapshotError(f"snapshot file checksum mismatch: {relative}")

    writable_mask = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    for relative in sorted(files | directories):
        if (snapshot_root / relative).lstat().st_mode & writable_mask:
            raise SnapshotError(f"snapshot node is writable: {relative}")
    if snapshot_root.lstat().st_mode & writable_mask:
        raise SnapshotError("snapshot root is writable")

    commit_id = (snapshot_root / COMMIT_FILE).read_text(encoding="utf-8").strip()
    tree_id = (snapshot_root / TREE_FILE).read_text(encoding="utf-8").strip()
    if not commit_id or not tree_id:
        raise SnapshotError("snapshot Git provenance is empty")
    return {
        "source_root": str(snapshot_root),
        "source_commit": commit_id,
        "source_tree": tree_id,
        "source_manifest": actual_manifest_id,
    }


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for current, dirnames, filenames in os.walk(path, topdown=False):
        for name in filenames:
            os.chmod(Path(current) / name, 0o600, follow_symlinks=False)
        for name in dirnames:
            node = Path(current) / name
            if not node.is_symlink():
                os.chmod(node, 0o700)
    os.chmod(path, 0o700)
    shutil.rmtree(path)


def create_snapshot(
    source_root: Path,
    destination: Path,
    *,
    commit_id: Optional[str] = None,
    tree_id: Optional[str] = None,
) -> Dict[str, str]:
    source_root = source_root.expanduser().absolute()
    if source_root.is_symlink():
        raise SnapshotError("source root must not be a symlink")
    source_root = source_root.resolve()
    destination = destination.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise SnapshotError(f"snapshot destination already exists: {destination}")
    if destination == source_root or source_root in destination.parents:
        raise SnapshotError("snapshot destination must be outside the source root")
    if (commit_id is None) != (tree_id is None):
        raise SnapshotError("commit_id and tree_id must be supplied together")
    if commit_id is None or tree_id is None:
        commit_id, tree_id = _git_provenance(source_root)

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    try:
        for source_path, relative in _iter_source_files(source_root):
            destination_path = destination / relative
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, destination_path)

        (destination / COMMIT_FILE).write_text(f"{commit_id}\n", encoding="utf-8")
        (destination / TREE_FILE).write_text(f"{tree_id}\n", encoding="utf-8")

        files, _ = _scan_tree(destination)
        manifest_lines = []
        for relative in sorted(files):
            if relative in {MANIFEST_FILE, MANIFEST_ID_FILE}:
                continue
            manifest_lines.append(f"{_sha256(destination / relative)}  {relative}\n")
        (destination / MANIFEST_FILE).write_text(
            "".join(manifest_lines), encoding="utf-8"
        )
        manifest_id = _sha256(destination / MANIFEST_FILE)
        (destination / MANIFEST_ID_FILE).write_text(
            f"{manifest_id}\n", encoding="utf-8"
        )

        for current, dirnames, filenames in os.walk(destination, topdown=False):
            for name in filenames:
                os.chmod(Path(current) / name, 0o444)
            for name in dirnames:
                os.chmod(Path(current) / name, 0o555)
        os.chmod(destination, 0o555)
        return verify_snapshot(destination, expected_manifest=manifest_id)
    except Exception:
        _remove_tree(destination)
        raise


def _shell_exports(provenance: Dict[str, str]) -> str:
    names = {
        "source_root": "GR_SOURCE_ROOT",
        "source_commit": "GR_SOURCE_COMMIT",
        "source_tree": "GR_SOURCE_TREE",
        "source_manifest": "GR_SOURCE_MANIFEST",
    }
    return "\n".join(
        f"export {names[key]}={shlex.quote(value)}" for key, value in provenance.items()
    )


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("destination", type=Path)
    create_parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("snapshot", type=Path)
    verify_parser.add_argument("--expected-manifest")
    verify_parser.add_argument("--shell", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        if args.command == "create":
            provenance = create_snapshot(args.source, args.destination)
        else:
            provenance = verify_snapshot(args.snapshot, args.expected_manifest)
    except SnapshotError as error:
        print(f"snapshot error: {error}", file=sys.stderr)
        return 1

    if getattr(args, "shell", False):
        print(_shell_exports(provenance))
    else:
        for key, value in provenance.items():
            print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
