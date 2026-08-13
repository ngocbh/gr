#!/usr/bin/env python3

import os
import stat
import tempfile
import unittest
from pathlib import Path

from scripts.snapshot import SnapshotError, create_snapshot, verify_snapshot


class SnapshotTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "source"
        for directory in (
            self.source / "configs",
            self.source / "generative_recommenders" / "research",
            self.source / "scripts",
            self.source / "tests",
        ):
            directory.mkdir(parents=True, exist_ok=True)
        (self.source / "main.py").write_text("print('main')\n", encoding="utf-8")
        (self.source / "preprocess_public_data.py").write_text(
            "print('preprocess')\n", encoding="utf-8"
        )
        (self.source / "requirements.txt").write_text("torch\n", encoding="utf-8")
        (self.source / "generative_recommenders" / "__init__.py").write_text(
            "", encoding="utf-8"
        )
        (self.source / "generative_recommenders" / "research" / "model.py").write_text(
            "VALUE = 1\n", encoding="utf-8"
        )
        (self.source / "configs" / "experiment.gin").write_text(
            "train.seed = 42\n", encoding="utf-8"
        )
        (self.source / "scripts" / "run.sh").write_text(
            "#!/usr/bin/env bash\n", encoding="utf-8"
        )
        self.snapshot = self.root / "snapshot"

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _create(self) -> None:
        create_snapshot(
            self.source,
            self.snapshot,
            commit_id="commit-test",
            tree_id="tree-test",
        )

    def _make_writable(self, path: Path) -> None:
        os.chmod(path, path.stat().st_mode | stat.S_IWUSR)

    def test_clean_snapshot_verifies_and_is_immutable(self) -> None:
        self._create()
        provenance = verify_snapshot(self.snapshot)
        self.assertEqual(provenance["source_commit"], "commit-test")
        self.assertEqual(provenance["source_tree"], "tree-test")
        self.assertRegex(provenance["source_manifest"], r"^[0-9a-f]{64}$")
        self.assertEqual(self.snapshot.stat().st_mode & 0o222, 0)

    def test_research_preprocessor_is_included(self) -> None:
        self._create()
        self.assertEqual(
            (self.snapshot / "preprocess_public_data.py").read_text(encoding="utf-8"),
            "print('preprocess')\n",
        )

    def test_file_tampering_is_rejected(self) -> None:
        self._create()
        target = self.snapshot / "main.py"
        self._make_writable(target)
        target.write_text("print('tampered')\n", encoding="utf-8")
        with self.assertRaises(SnapshotError):
            verify_snapshot(self.snapshot)

    def test_unlisted_file_is_rejected(self) -> None:
        self._create()
        self._make_writable(self.snapshot)
        (self.snapshot / "added.py").write_text("added = True\n", encoding="utf-8")
        with self.assertRaises(SnapshotError):
            verify_snapshot(self.snapshot)

    def test_symlink_is_rejected(self) -> None:
        self._create()
        self._make_writable(self.snapshot)
        (self.snapshot / "link.py").symlink_to("main.py")
        with self.assertRaisesRegex(SnapshotError, "symlink or special node"):
            verify_snapshot(self.snapshot)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "mkfifo is unavailable")
    def test_special_node_is_rejected(self) -> None:
        self._create()
        self._make_writable(self.snapshot)
        os.mkfifo(self.snapshot / "pipe")
        with self.assertRaisesRegex(SnapshotError, "symlink or special node"):
            verify_snapshot(self.snapshot)

    def test_source_symlink_is_rejected(self) -> None:
        (self.source / "scripts" / "link.py").symlink_to("run.sh")
        with self.assertRaisesRegex(SnapshotError, "symlink or special node"):
            create_snapshot(
                self.source,
                self.snapshot,
                commit_id="commit-test",
                tree_id="tree-test",
            )

    def test_snapshot_root_symlink_is_rejected(self) -> None:
        self._create()
        root_link = self.root / "snapshot-link"
        root_link.symlink_to(self.snapshot, target_is_directory=True)
        with self.assertRaisesRegex(SnapshotError, "non-symlink directory"):
            verify_snapshot(root_link)


if __name__ == "__main__":
    unittest.main()
