"""Exercise real runner persistence with failing cases and model processes."""
import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from system1bench import colab_batch, run
from system1bench.common import digest, sha, write


class BrokenCasesAdapter:
    metadata = {"adapter": "test-no-inference"}

    def __init__(self, *args):
        self.state = None

    def audit(self, state, questions, budget):
        self.state = state["kind"]
        self.sync_count = 0
        if self.state == "audit":
            raise ImportError("audit dependency unavailable")
        return {qid: {"complete": True} for qid in questions}

    def synchronize(self):
        self.sync_count += 1
        if self.state == "sync_before" or (self.state == "sync_after" and self.sync_count == 2):
            raise RuntimeError("device error")

    def predict(self, *args):
        if self.state == "predict":
            raise ValueError("prediction failed")
        if self.state == "output":
            return [42]
        if self.state == "decode":
            return [{"answers": {"q": {"choice": "invalid", "probabilities": {}}}}]
        if self.state == "serialize":
            return [{"answers": {"q": {"choice": "a", "probabilities": {"a": 1., "b": 0.}, "raw": object()}}}]
        return [{"answers": {"q": {"choice": "a", "probabilities": {"a": 1., "b": 0.}}}}]


class ContinueOnErrorTests(unittest.TestCase):
    def test_failed_cases_are_recorded_and_last_case_still_runs(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            self.assertTrue(root.resolve().is_relative_to(Path.cwd().resolve()))
            data = root / "data"
            cases = []
            for i, kind in enumerate(["audit", "predict", "sync_before", "sync_after", "output", "decode", "serialize", "ok"]):
                q = {"q": {"type": "choice", "instructions": "Pick", "criteria": {"a": "A", "b": "B"}}}
                payload = dict(state={"kind": kind}, questions=q)
                cases.append(dict(id=str(i), **payload, gold={"q": {"label": "a"}},
                                  group=str(i), family="test", language="en", request_sha256=digest(payload)))
            frozen = dict(budget={"max_len": 64, "head_max_len": 32},
                          suites=[dict(name="test", cases=cases)])
            write(data / "frozen.json", frozen)
            write(root / "protocol_manifest.json", {"prepared_sha256": sha(data / "frozen.json")})
            argv = ["run", "--model", "english", "--adapter", "test:Adapter", "--batch-size", "1",
                    "--continue-on-error", "--output", str(root / "results")]
            with patch.object(run, "ROOT", root), patch.object(run, "DATA", data), \
                    patch.object(run.importlib, "import_module", return_value=types.SimpleNamespace(Adapter=BrokenCasesAdapter)), \
                    patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                run.main()
            out = root / "results/english"
            rows = run.read_run(out / "test.json.gz")["rows"]
            self.assertEqual(len(rows), 8)
            self.assertEqual(sum(r["error"] is not None for r in rows), 7)
            self.assertEqual(rows[-1]["prediction"], "a")
            self.assertIsNone(rows[-1]["error"])
            self.assertFalse(rows[0]["audit"]["complete"])
            meta = json.loads((out / "metadata.json").read_text())
            self.assertEqual(meta["status"], "DONE")
            self.assertEqual(meta["suites"]["test"]["errors"], 7)
            failures = [json.loads(line) for line in (out / "errors.jsonl").read_text().splitlines()]
            self.assertEqual({f["stage"] for f in failures},
                             {"audit", "predict", "synchronize_before", "synchronize_after", "output", "decode", "serialize_answer"})
            self.assertTrue(all(f["traceback"] for f in failures))

    def test_default_audit_failure_still_raises(self):
        adapter = BrokenCasesAdapter()
        with self.assertRaises(ImportError):
            run.evaluate_batch(adapter, [{"state": {"kind": "audit"}}], {}, "en", {})

    def test_interrupt_is_not_swallowed(self):
        adapter = BrokenCasesAdapter()
        adapter.audit = lambda *args: (_ for _ in ()).throw(KeyboardInterrupt())
        with self.assertRaises(KeyboardInterrupt):
            run.evaluate_batch(adapter, [{"state": {}}], {}, "en", {}, True)

    def test_failed_model_does_not_block_peer_or_reports(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            output = Path(tmp)
            self.assertTrue(output.resolve().is_relative_to(Path.cwd().resolve()))
            calls = []

            def execute(args, log):
                calls.append(args)
                if "system1bench.run" in args:
                    model = args[args.index("--model") + 1]
                    if model == "english":
                        return 1
                    model_root = Path(args[args.index("--output") + 1])
                    write(model_root / model / "metadata.json",
                          {"status": "DONE", "suites": {"sample": {"decisions": 3, "errors": 1}}})
                return 2 if "system1bench.metrics" in args else 0

            with contextlib.redirect_stdout(io.StringIO()):
                summary = colab_batch.run_models(["english", "multilingual"], output, execute)
            self.assertEqual(summary["models"]["english"]["status"], "FAILED")
            peer = summary["models"]["multilingual"]
            self.assertEqual(peer["status"], "DONE_WITH_ERRORS")
            self.assertEqual(peer["reports"]["metrics"]["status"], "FAILED")
            self.assertEqual(peer["reports"]["report"]["status"], "OK")
            self.assertEqual(summary["status"], "FINISHED_WITH_ERRORS")
            self.assertTrue((output / "run_status.csv").is_file())
            self.assertTrue((output / "failures.csv").is_file())
            self.assertEqual(len(calls), 4)


if __name__ == "__main__":
    unittest.main()
