"""Real inference, per-suite atomic outputs, input-bound verified resumption."""
import argparse
import gzip
import importlib
import json
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone

from .common import DATA, ROOT, decode, digest, labels, read, request, sha, write


def read_run(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def save_run(path, obj):
    raw = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    temp = path.with_suffix(".tmp")
    temp.write_bytes(gzip.compress(raw, mtime=0))
    temp.replace(path)


def validate_saved(saved, suite, signature):
    if saved["signature"] != signature:
        raise ValueError("Run fingerprint mismatch; use a new output directory")
    wanted = {(c["id"], qid): c for c in suite["cases"] for qid in c["questions"]}
    actual = [(r["id"], r["qid"]) for r in saved["rows"]]
    if len(set(actual)) != len(actual) or set(actual) != set(wanted):
        raise ValueError("Missing, duplicate or unexpected result IDs")
    for r in saved["rows"]:
        c = wanted[r["id"], r["qid"]]
        if r["request_sha256"] != c["request_sha256"] or r["gold"] != str(c["gold"][r["qid"]]["label"]):
            raise ValueError("Saved request/gold mismatch")


def validate_complete_files(out, suites, metadata):
    # A completed run is immutable: do not regenerate a missing file under stale
    # timing/hash metadata. Recovery must use a fresh output directory.
    for suite in suites:
        name = suite["name"]
        path = out / (name + ".json.gz")
        expected = metadata["suites"].get(name)
        if not path.is_file() or expected is None or sha(path) != expected["sha256"]:
            raise ValueError("Completed run has missing/corrupt files; use a new output directory")


def failed_audit(questions):
    # Unknown audit values stay unknown; never claim failed inputs were complete.
    return {qid: dict(complete=False, audit_failed=True, state_tokens=None,
                     head_tokens=None, options=0, unique_encoded_options=0,
                     option_texts_shortened=None, instruction_shortened=None,
                     state_shortened=None, special_mask_sanitized=None)
            for qid in questions}


def evaluate_batch(adapter, batch, questions, language, budget,
                   continue_on_error=False, on_error=None):
    """Keep per-case failures isolated in diagnostic mode (batch size one)."""
    if continue_on_error and len(batch) != 1:
        raise ValueError("continue-on-error requires batch size 1")
    started = time.perf_counter()
    stage = "audit"
    try:
        audits = [adapter.audit(c["state"], questions, budget) for c in batch]
        for audit in audits:
            for qid in questions:
                if not isinstance(audit[qid], dict):
                    raise ValueError("Invalid input audit")
            if continue_on_error:
                json.dumps(audit, allow_nan=False)
        stage = "synchronize_before"
        adapter.synchronize()
    except Exception as exc:
        if not continue_on_error:
            raise
        if on_error:
            on_error(stage, exc)
        return ([None] * len(batch), [failed_audit(questions) for _ in batch],
                type(exc).__name__, time.perf_counter() - started)
    started = time.perf_counter()
    error = None
    try:
        outputs = adapter.predict([request(c)["state"] for c in batch], questions,
                                  language, budget, len(batch))
        if len(outputs) != len(batch):
            raise ValueError("Wrong output count")
    except Exception as exc:
        error = type(exc).__name__
        outputs = [None] * len(batch)
        if on_error:
            on_error("predict", exc)
    try:
        adapter.synchronize()
    except Exception as exc:
        if not continue_on_error:
            raise
        error = error or type(exc).__name__
        outputs = [None] * len(batch)
        if on_error:
            on_error("synchronize_after", exc)
    return outputs, audits, error, time.perf_counter() - started


def log_failure(path, model, suite, batch, stage, exc, qid=None):
    record = dict(model=model, suite=suite, case_ids=[c["id"] for c in batch],
                  qid=qid, stage=stage, error=type(exc).__name__, message=str(exc),
                  traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("CASE_ERROR", model, suite, record["case_ids"], stage, record["error"], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint")
    p.add_argument("--adapter", default="system1bench.laya_adapter:LayaAdapter")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output", default="results")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Record audit/GPU/decode failures and continue; requires batch size 1")
    args = p.parse_args()
    if args.continue_on_error and args.batch_size != 1:
        p.error("--continue-on-error requires --batch-size 1")
    frozen = read(DATA / "frozen.json")
    if sha(DATA / "frozen.json") != read(ROOT / "protocol_manifest.json")["prepared_sha256"]:
        raise ValueError("Prepared data is not bound to the published manifest")
    module, cls = args.adapter.split(":")
    adapter = getattr(importlib.import_module(module), cls)(args.model, args.checkpoint)
    code = {str(f.relative_to(ROOT)): sha(f) for f in sorted((ROOT / "system1bench").glob("*.py")) if f.name not in ["report.py", "metrics.py"]}
    signature = dict(protocol_sha256=sha(ROOT / "protocol_manifest.json"), prepared_sha256=sha(DATA / "frozen.json"),
                     source_code_sha256=code, model=adapter.metadata, batch_size=args.batch_size, budget=frozen["budget"])
    if args.continue_on_error:
        signature["continue_on_error"] = True
    out = ROOT / args.output / args.model
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "metadata.json"
    if meta_path.exists():
        meta = read(meta_path)
        if meta["signature"] != signature:
            raise ValueError("Metadata signature changed; use a new output directory")
    else:
        meta = dict(signature=signature, status="RUNNING", started_at=datetime.now(timezone.utc).isoformat(), suites={})
        write(meta_path, meta)
    was_complete = meta["status"] == "DONE"
    if was_complete:
        validate_complete_files(out, frozen["suites"], meta)
    for suite in frozen["suites"]:
        name = suite["name"]
        path = out / (name + ".json.gz")
        if path.exists():
            if name in meta["suites"] and sha(path) != meta["suites"][name]["sha256"]:
                raise ValueError("Saved output hash mismatch")
            saved = read_run(path)
            validate_saved(saved, suite, signature)
            print("VERIFIED", args.model, name, flush=True)
        else:
            groups = defaultdict(list)
            for c in suite["cases"]:
                groups[(digest(c["questions"]), c["language"])].append(c)
            records, batches = [], []
            for (_, lang), cases in groups.items():
                q = cases[0]["questions"]
                for start in range(0, len(cases), args.batch_size):
                    batch = cases[start:start + args.batch_size]
                    def record_error(stage, exc, qid=None):
                        log_failure(out / "errors.jsonl", args.model, name, batch, stage, exc, qid)

                    outputs, audits, error, seconds = evaluate_batch(
                        adapter, batch, q, lang, frozen["budget"], args.continue_on_error,
                        record_error if args.continue_on_error else None)
                    bid = len(batches)
                    batches.append(dict(id=bid, requests=len(batch), decisions=sum(len(c["questions"]) for c in batch), seconds=seconds))
                    for c, output, audit in zip(batch, outputs, audits):
                        output_error = error
                        try:
                            answers = output.get("answers", {}) if output is not None else {}
                            if not isinstance(answers, dict):
                                raise ValueError("Answers must be an object")
                        except Exception as exc:
                            if not args.continue_on_error:
                                raise
                            answers = {}
                            output_error = output_error or type(exc).__name__
                            record_error("output", exc)
                        for qid, question in q.items():
                            err = output_error
                            answer = answers.get(qid)
                            pred, probabilities = None, None
                            stage = "decode"
                            try:
                                if args.continue_on_error:
                                    stage = "serialize_answer"
                                    json.dumps(answer, allow_nan=False)
                                    stage = "decode"
                                pred, probabilities = decode(question, answer)
                            except Exception as e:
                                err = err or type(e).__name__
                                if args.continue_on_error and output_error is None:
                                    record_error(stage, e, qid)
                                if args.continue_on_error and stage == "serialize_answer":
                                    answer = None
                            records.append(dict(id=c["id"], qid=qid, group=c["group"], family=c["family"], language=c["language"],
                                                request_sha256=c["request_sha256"], questions_sha256=digest(q), type=question["type"],
                                                labels=labels(question), gold=str(c["gold"][qid]["label"]),
                                                answer=answer, prediction=pred, probabilities=probabilities, error=err,
                                                audit=audit[qid], batch_id=bid, length_target=c.get("length_target"), position=c.get("position")))
            saved = dict(signature=signature, suite={k: v for k, v in suite.items() if k != "cases"}, rows=records, batches=batches,
                         completed_at=datetime.now(timezone.utc).isoformat())
            validate_saved(saved, suite, signature)
            save_run(path, saved)
        meta["suites"][name] = dict(sha256=sha(path), requests=len(suite["cases"]), decisions=len(saved["rows"]),
                                    errors=sum(r["error"] is not None for r in saved["rows"]),
                                    inference_seconds=sum(b["seconds"] for b in saved["batches"]))
        if not was_complete:
            write(meta_path, meta)
        print("DONE", args.model, name, meta["suites"][name], flush=True)
    if not was_complete:
        meta.update(status="DONE", finished_at=datetime.now(timezone.utc).isoformat())
        write(meta_path, meta)


if __name__ == "__main__":
    main()
