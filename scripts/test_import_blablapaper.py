from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from import_blablapaper import BundleError, import_bundle


class ImportBlaBlaPaperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.source = self.root / "outputs" / "test-paper"
        (self.source / "images").mkdir(parents=True)
        (self.source / "images" / "figure.png").write_bytes(b"image")
        (self.source / "info.json").write_text(
            json.dumps(
                {
                    "index": "test-paper",
                    "paper_title": "Test Paper",
                    "metadata": {"authors": ["Ada"], "venue": "TestConf", "year": 2026},
                    "description": "A generated test paper.",
                }
            ),
            encoding="utf-8",
        )
        for name in (
            "paper_notes.md",
            "ELI5_notes.md",
            "figs_notes.md",
            "translation_notes.md",
        ):
            (self.source / name).write_text(
                f"# {name}\n\n![figure](images/figure.png)\n",
                encoding="utf-8",
            )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_imports_complete_bundle(self) -> None:
        destination = import_bundle(
            self.source,
            self.root / "content",
            "ISCA26",
            "llm, inference",
            "abc123",
        )
        self.assertEqual(destination, self.root / "content" / "ISCA26" / "test-paper")
        self.assertTrue((destination / "translation_notes.md").is_file())
        self.assertTrue((destination / "images" / "figure.png").is_file())
        self.assertIn("[原文翻译](translation_notes.md)", (destination / "index.md").read_text())
        manifest = json.loads((destination / "paper.json").read_text())
        self.assertEqual(manifest["source_sha256"], "abc123")
        self.assertEqual(manifest["tags"], ["paper", "llm", "inference"])

    def test_rejects_path_traversal_collection(self) -> None:
        with self.assertRaises(BundleError):
            import_bundle(self.source, self.root / "content", "../outside", "paper")

    def test_rejects_missing_local_image(self) -> None:
        (self.source / "paper_notes.md").write_text(
            "# Missing image\n\n![](images/missing.png)\n",
            encoding="utf-8",
        )
        with self.assertRaises(BundleError):
            import_bundle(self.source, self.root / "content", "misc", "paper")


if __name__ == "__main__":
    unittest.main()
