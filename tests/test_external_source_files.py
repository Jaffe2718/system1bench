"""The CLINC API receipt must not block real, pinned evaluation inputs."""
import unittest

from system1bench.common import ROOT, read, required_external_files


class ExternalSourceFilesTests(unittest.TestCase):
    def test_only_clinc_api_receipt_is_excluded(self):
        registry = read(ROOT / "external_sources.json")
        excluded = []
        for name, spec in registry.items():
            required = required_external_files(name, spec)
            excluded.extend((name, f["path"]) for f in spec["files"] if f not in required)
        self.assertEqual(excluded, [("clinc", "api_metadata.json")])
        self.assertEqual(
            [f["path"] for f in required_external_files("clinc", registry["clinc"])],
            ["README.md", "plus/test-00000-of-00001.parquet"],
        )

    def test_other_files_and_sources_are_never_exempt(self):
        files = [{"path": "api_metadata.json", "sha256": "pinned"},
                 {"path": "test.json", "sha256": "pinned"}]
        self.assertEqual(required_external_files("other", {"files": files}), files)
        self.assertEqual(required_external_files("clinc", {"files": files}), files[1:])


if __name__ == "__main__":
    unittest.main()
