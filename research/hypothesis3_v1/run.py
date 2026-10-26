"""Collect hypothesis 3 predictions and failures. Research evaluation is deferred."""
import argparse
import csv
import json
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import build
import adaptive
from system1bench.common import decode, labels
from system1bench.run import evaluate_batch, failed_audit


def now():
    return datetime.now(timezone.utc).isoformat()


def load_bank(bank, profile="full"):
    build.verify(bank)
    rows = [json.loads(line) for panel in ("native", "paired", "multi_select")
            for line in (bank / (panel + ".jsonl")).read_text(encoding="utf-8").splitlines()]
    if profile == "pilot":
        pilot = set(json.loads((bank / "pilot_base_ids.json").read_text(encoding="utf-8")))
        rows = [r for r in rows if r["panel"] == "paired" and r["base_id"] in pilot]
    return rows


def predict_case(adapter, row, budget, failure):
    outputs, audits, error, seconds = evaluate_batch(
        adapter, [row], row["questions"], row["language"], budget, True, failure)
    return format_prediction(row, outputs[0] if outputs else None, audits[0], error, seconds, failure)


def format_prediction(row, output, audit, error, seconds, failure):
    answers = {}
    if error is None:
        try:
            answers = output.get("answers", {})
            if not isinstance(answers, dict):
                raise ValueError("Answers must be an object")
        except Exception as exc:
            error = type(exc).__name__
            answers = {}
            failure("output", exc)
    decoded, predicted = {}, []
    for qid, question in row["questions"].items():
        answer, prediction, probabilities, qerror = answers.get(qid), None, None, error
        stage = "serialize_answer"
        if error is None:
            try:
                json.dumps(answer, allow_nan=False)
                stage = "decode"
                prediction, probabilities = decode(question, answer)
                if question["type"] == "choice":
                    predicted = [prediction]
                elif prediction == "true":
                    predicted.append(qid)
            except Exception as exc:
                qerror = type(exc).__name__
                failure(stage, exc, qid)
                if stage == "serialize_answer":
                    answer = None
        decoded[qid] = dict(prediction=prediction, probabilities=probabilities,
                            labels=labels(question), answer=answer, error=qerror)
    invalid = sum(v["error"] is not None for v in decoded.values())
    kept = {k: row[k] for k in ("id", "base_id", "source", "source_id", "group", "panel",
                                "split", "k", "conditions", "gold_ids", "request_sha256")}
    return dict(**kept, predicted_ids=predicted,
                decision_errors=invalid, error=next((v["error"] for v in decoded.values() if v["error"]), None),
                complete_input=all(v.get("complete", False) for v in audit.values()),
                audit=audit, answers=decoded, inference_seconds=seconds)


def journal_rows(path, wanted):
    if not path.exists():
        return {}
    saved = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["id"] in saved or row["id"] not in wanted:
            raise ValueError("Unexpected/duplicate saved case ID")
        original = wanted[row["id"]]
        if row["request_sha256"] != original["request_sha256"] or row["gold_ids"] != original["gold_ids"]:
            raise ValueError("Saved case fingerprint mismatch")
        saved[row["id"]] = row
    return saved


def run_model(bank, output, model, profile="full", adapter_factory=None, options=None):
    options = execution_options(options)
    rows = load_bank(bank, profile)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "metadata.json"
    old = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
    if adapter_factory is None:
        from system1bench.laya_adapter import LayaAdapter
        adapter_factory = LayaAdapter
    # Each model is constructed once, in its own process. No separate smoke call.
    adapter = adapter_factory(model)
    signature = dict(bank_manifest_sha256=build.file_sha(bank / "manifest.json"), profile=profile,
                     source_sha256={p.name: build.text_sha(p) for p in (HERE / "build.py", HERE / "run.py", HERE / "adaptive.py")},
                     adapter_sha256=build.text_sha(ROOT / "system1bench/laya_adapter.py"),
                     common_sha256=build.text_sha(ROOT / "system1bench/common.py"),
                     runner_sha256=build.text_sha(ROOT / "system1bench/run.py"),
                     model=adapter.metadata, budget=dict(max_len=8192, head_max_len=4096), execution=options,
                     inference_seconds_semantics=("row-weighted share of successful CUDA batch forward time; "
                                                 "excludes CPU preparation, decoding and retry overhead"
                                                 if options["scheduler"] == "adaptive" else "serial predict call"),
                     noul_decoding_threshold=.5, evaluation_performed=False)
    if old and old["signature"] != signature:
        raise ValueError("Run signature changed; use a new output directory")
    meta = dict(signature=signature, status="RUNNING", started_at=old["started_at"] if old else now(),
                expected_requests=len(rows), expected_decisions=sum(len(r["questions"]) for r in rows))
    build.dump(metadata_path, meta)
    journal = output / "predictions.jsonl"
    saved = journal_rows(journal, {r["id"]: r for r in rows})
    event_lock = threading.Lock()

    def event(kind, **values):
        with event_lock:
            with (output / "adaptive_events.jsonl").open("a", encoding="utf-8") as events:
                events.write(json.dumps(dict(time=now(), event=kind, **values), ensure_ascii=False) + "\n")
        if kind in ("start", "grow", "shrink", "batch_retry", "stop"):
            print("SCHEDULER", kind, json.dumps(values, ensure_ascii=False), flush=True)

    pending = [r for r in rows if r["id"] not in saved]
    results = (adaptive.collect(pending, adapter, signature["budget"],
                               {k: v for k, v in options.items() if k != "scheduler"}, event)
               if options["scheduler"] == "adaptive" else
               (dict(prepared=dict(row=r)) for r in pending))
    with journal.open("a", encoding="utf-8", newline="\n") as stream:
        for result in results:
            row = result["prepared"]["row"]

            def failure(stage, exc, qid=None):
                detail = dict(model=model, id=row["id"], base_id=row["base_id"], source=row["source"],
                              panel=row["panel"], k=row["k"], stage=stage, qid=qid,
                              error=adaptive.error_name(exc), message=str(exc),
                              traceback=adaptive.error_trace(exc))
                with (output / "errors.jsonl").open("a", encoding="utf-8") as errors:
                    errors.write(json.dumps(detail, ensure_ascii=False) + "\n")
                print("CASE_ERROR", model, row["id"], stage, detail["error"], flush=True)

            if options["scheduler"] == "serial":
                prediction = predict_case(adapter, row, signature["budget"], failure)
            else:
                error = result["error"]
                if error is not None:
                    failure(result["stage"], error)
                prediction = format_prediction(
                    row, result["output"], result["prepared"].get("audit", failed_audit(row["questions"])),
                    adaptive.error_name(error) if error else None, result["seconds"], failure)
                prediction["execution"] = result["execution"]
            stream.write(build.encode(prediction).decode("utf-8") + "\n")
            stream.flush()
            saved[row["id"]] = prediction
            if len(saved) % 100 == 0 or len(saved) == len(rows):
                meta.update(completed_requests=len(saved),
                            failed_requests=sum(r["error"] is not None for r in saved.values()))
                build.dump(metadata_path, meta)
                print("PROGRESS", model, len(saved), "/", len(rows), "errors", meta["failed_requests"], flush=True)
    meta.update(status="DONE", finished_at=now(), completed_requests=len(saved),
                failed_requests=sum(r["error"] is not None for r in saved.values()),
                decision_errors=sum(r["decision_errors"] for r in saved.values()),
                predictions_sha256=build.file_sha(journal))
    build.dump(metadata_path, meta)
    return meta


def command(args, log):
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(list(map(str, args)), cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
                stream.flush()
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def run_all(bank, output, models, profile="full", execute=command, options=None):
    options = execution_options(options)
    expected = load_bank(bank, profile)
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(status="RUNNING", experiment="hypothesis3-v1", profile=profile,
                   evaluation_performed=False, flow="configure-run-download",
                   expected_requests_per_model=len(expected), models={},
                   bank_manifest=json.loads((bank / "manifest.json").read_text(encoding="utf-8")),
                   bank_audit=json.loads((bank / "audit.json").read_text(encoding="utf-8")))
    for model in models:
        log = output / (model + ".log")
        model_output = output / model
        result = dict(status="RUNNING", log=str(log))
        summary["models"][model] = result
        build.dump(output / "run_status.json", summary)
        try:
            code = execute([sys.executable, "-u", str(HERE / "run.py"), "--model", model,
                            "--bank", str(bank), "--output", str(model_output), "--profile", profile,
                            *option_args(options)], log)
            result["exit_code"] = code
            path = model_output / "metadata.json"
            meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            rows = journal_rows(model_output / "predictions.jsonl", {r["id"]: r for r in expected})
            if meta.get("status") == "DONE" and build.file_sha(model_output / "predictions.jsonl") != meta["predictions_sha256"]:
                raise ValueError("Prediction journal hash mismatch")
            complete = code == 0 and meta.get("status") == "DONE" and len(rows) == len(expected)
            result.update(status=("DONE_WITH_ERRORS" if any(r["error"] for r in rows.values()) else "OK") if complete else "FAILED",
                          completed_requests=len(rows), failed_requests=sum(r["error"] is not None for r in rows.values()))
            if not complete:
                result["log_tail"] = "\n".join(log.read_text(encoding="utf-8").splitlines()[-40:]) if log.exists() else ""
        except Exception as exc:
            result.update(status="FAILED", error=type(exc).__name__ + ": " + str(exc))
        build.dump(output / "run_status.json", summary)
        print("MODEL_RESULT", model, result["status"], flush=True)
    summary["status"] = "OK" if all(r["status"] == "OK" for r in summary["models"].values()) else "FINISHED_WITH_ERRORS"
    build.dump(output / "run_status.json", summary)
    with (output / "run_status.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "model", "status", "expected_requests", "completed_requests", "failed_requests"])
        writer.writeheader()
        for model, result in summary["models"].items():
            writer.writerow(dict(model=model, status=result["status"], expected_requests=len(expected),
                completed_requests=result.get("completed_requests", 0),
                failed_requests=result.get("failed_requests", 0)))
    with (output / "failures.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["model", "id", "base_id", "source", "panel", "k", "stage", "qid", "error", "message"]
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for model in models:
            path = output / model / "errors.jsonl"
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    writer.writerow(json.loads(line))
            result = summary["models"][model]
            if result["status"] == "FAILED":
                writer.writerow(dict(model=model, stage="model", error="ModelFailed",
                                     message=result.get("error", result.get("log_tail", result["log"]))))
    return summary


def execution_options(options=None):
    # Existing Python callers keep serial behavior; the notebook/CLI explicitly select adaptive.
    options = dict(options or {"scheduler": "serial"})
    scheduler = options.pop("scheduler", "serial")
    if scheduler not in ("serial", "adaptive"):
        raise ValueError("Unknown scheduler")
    return dict(scheduler=scheduler, **adaptive.validate_options(options)) if scheduler == "adaptive" else dict(scheduler="serial")


def option_args(options):
    result = []
    for key, value in options.items():
        result += ["--" + key.replace("_", "-"), str(value)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=["english", "multilingual"], default=["english", "multilingual"])
    parser.add_argument("--model", choices=["english", "multilingual"])
    parser.add_argument("--profile", choices=["full", "pilot"], default="full")
    parser.add_argument("--scheduler", choices=["serial", "adaptive"], default="adaptive")
    for key, value in adaptive.DEFAULTS.items():
        parser.add_argument("--" + key.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args()
    options = execution_options(dict(scheduler=args.scheduler,
                                     **{k: getattr(args, k) for k in adaptive.DEFAULTS}))
    if args.model:
        run_model(args.bank.resolve(), args.output.resolve(), args.model, args.profile, options=options)
    else:
        result = run_all(args.bank.resolve(), args.output.resolve(), list(dict.fromkeys(args.models)), args.profile, options=options)
        print("HYPOTHESIS3_FINISHED", result["status"], flush=True)


if __name__ == "__main__":
    main()
