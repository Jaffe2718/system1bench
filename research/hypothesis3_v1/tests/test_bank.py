import contextlib
import copy
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import build

spec = importlib.util.spec_from_file_location("hypothesis3_runner", HERE / "run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture_items():
    return [build.base_item("fixture", i, {"text": text}, "Choose one", ["alpha", "beta", "gamma", "delta", "epsilon"],
                            [2], text) for i, text in enumerate(("audit failure", "ordinary", "another"))]


class FakeAdapter:
    metadata = {"adapter": "unit-test-only"}
    calls = 0

    def __init__(self, model):
        pass

    def synchronize(self):
        pass

    def audit(self, state, questions, budget):
        if state["text"] == "audit failure":
            raise ImportError("fixture error")
        return {qid: {"complete": True} for qid in questions}

    def predict(self, states, questions, language, budget, batch_size):
        type(self).calls += 1
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "noul":
                answers[qid] = {"noul": .8}
            else:
                first = next(iter(q["criteria"]))
                answers[qid] = {"choice": first,
                                "probabilities": {k: float(k == first) for k in q["criteria"]}}
        return [{"answers": answers}]


class BankTests(unittest.TestCase):
    def test_nested_sets_keep_gold_text_and_sixteen_assignments(self):
        item = fixture_items()[1]
        rows, mothers = build.single_records([item], "paired", {item["base_id"]: "confirmation"})
        self.assertLessEqual(len(rows), 16)
        self.assertEqual(sum(len(r["conditions"]) for r in rows), 16)
        audit = build.audit_bank(rows, mothers, set(), set())
        self.assertEqual(audit["status"], "PASS")
        self.assertEqual({r["k"] for r in rows}, {2, 3, 4, 5})

    def test_gold_leaving_candidates_is_rejected(self):
        item = fixture_items()[1]
        rows, mothers = build.single_records([item], "paired", {item["base_id"]: "confirmation"})
        rows = copy.deepcopy(rows)
        rows[0]["gold_ids"] = ["missing"]
        with self.assertRaises(AssertionError):
            build.audit_bank(rows, mothers, set(), set())

    def test_split_is_by_context_group(self):
        items = fixture_items()
        items[1]["source_group_sha256"] = items[0]["source_group_sha256"]
        split = build.splits(items, "test")
        self.assertEqual(split[items[0]["base_id"]], split[items[1]["base_id"]])

    def test_duplicate_options_and_excluded_context_are_not_selected(self):
        items = fixture_items()
        invalid = copy.deepcopy(items[0])
        invalid["options"][1] = invalid["options"][0]
        selected = build.sample([invalid, *items], 1, set(), {items[0]["source_group_sha256"]}, set(), "test")
        self.assertNotEqual(selected[0]["base_id"], items[0]["base_id"])

    def test_multi_select_keeps_all_candidates_and_gold_outside_input(self):
        item = fixture_items()[1]
        item["gold_indices"] = [0, 2]
        rows = build.multi_records([item], {item["base_id"]: "confirmation"})
        build.audit_bank(rows, [], set(), set())
        for row in rows:
            self.assertEqual(len(row["gold_ids"]), 2)
            self.assertEqual(len(row["questions"]), 5)
            self.assertNotIn("gold_ids", row["state"])
            self.assertEqual(len(row["state"]["candidates"]), 5)

    def test_no_performance_scoring_and_case_failure_does_not_block_resume(self):
        with tempfile.TemporaryDirectory(dir=build.ROOT) as tmp:
            root = Path(tmp)
            self.assertTrue(root.resolve().is_relative_to(build.ROOT.resolve()))
            bank = root / "bank"
            bank.mkdir()
            build.dump(bank / "manifest.json", {"fixture": True})
            items = fixture_items()
            rows, _ = build.single_records(items, "native", {b["base_id"]: "confirmation" for b in items})
            FakeAdapter.calls = 0
            with patch.object(runner, "load_bank", return_value=rows), contextlib.redirect_stdout(io.StringIO()):
                meta = runner.run_model(bank, root / "result", "english", adapter_factory=FakeAdapter)
                calls = FakeAdapter.calls
                runner.run_model(bank, root / "result", "english", adapter_factory=FakeAdapter)
            self.assertEqual(FakeAdapter.calls, calls)
            self.assertEqual(meta["completed_requests"], len(rows))
            self.assertEqual(meta["failed_requests"], 2)
            saved = [json.loads(line) for line in (root / "result/predictions.jsonl").read_text().splitlines()]
            self.assertIsNone(saved[-1]["error"])
            for row in saved:
                self.assertNotIn("correct", row)
                self.assertNotIn("accuracy", row)
                self.assertNotIn("set_f1", row)

    def test_malformed_answers_do_not_abort_the_next_case(self):
        item = fixture_items()[1]
        rows, _ = build.single_records([item], "native", {item["base_id"]: "confirmation"})
        adapter = FakeAdapter("fixture")
        predict = adapter.predict
        adapter.predict = lambda *args: [{"answers": []}]
        errors = []
        bad = runner.predict_case(adapter, rows[0], {}, lambda *args: errors.append(args))
        adapter.predict = predict
        good = runner.predict_case(adapter, rows[1], {}, lambda *args: errors.append(args))
        self.assertEqual(bad["error"], "ValueError")
        self.assertIsNone(good["error"])
        self.assertEqual(errors[0][0], "output")

    def test_multi_select_preserves_raw_probabilities_without_f1(self):
        item = fixture_items()[1]
        item["gold_indices"] = [0, 2]
        row = build.multi_records([item], {item["base_id"]: "confirmation"})[0]
        prediction = runner.predict_case(FakeAdapter("fixture"), row, {}, lambda *args: None)
        self.assertIsNone(prediction["error"])
        self.assertEqual(len(prediction["answers"]), 5)
        self.assertTrue(all(v["answer"]["noul"] == .8 for v in prediction["answers"].values()))
        self.assertNotIn("correct", prediction)
        self.assertNotIn("set_f1", prediction)

    def test_model_failure_continues_and_download_bundle_has_no_scores(self):
        with tempfile.TemporaryDirectory(dir=build.ROOT) as tmp:
            root = Path(tmp)
            self.assertTrue(root.resolve().is_relative_to(build.ROOT.resolve()))
            bank = root / "bank"
            bank.mkdir()
            build.dump(bank / "manifest.json", {"fixture": True})
            build.dump(bank / "audit.json", {"fixture": True})
            items = fixture_items()
            rows, _ = build.single_records(items, "native", {b["base_id"]: "confirmation" for b in items})

            def execute(args, log):
                model = args[args.index("--model") + 1]
                if model == "english":
                    log.write_text("ImportError: simulated startup failure\n")
                    return 1
                runner.run_model(bank, Path(args[args.index("--output") + 1]), model, adapter_factory=FakeAdapter)
                return 0

            with patch.object(runner, "load_bank", return_value=rows), contextlib.redirect_stdout(io.StringIO()):
                summary = runner.run_all(bank, root / "results", ["english", "multilingual"], execute=execute)
            self.assertEqual(summary["models"]["english"]["status"], "FAILED")
            self.assertEqual(summary["models"]["multilingual"]["status"], "DONE_WITH_ERRORS")
            self.assertFalse(summary["evaluation_performed"])
            self.assertTrue((root / "results/failures.csv").is_file())
            self.assertFalse((root / "results/scores.csv").exists())


if __name__ == "__main__":
    unittest.main()
