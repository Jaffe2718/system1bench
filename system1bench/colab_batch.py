"""Run T4 Laya models in isolated processes and retain every failure report."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from .common import ROOT, read, write


def run_command(args, log_path):
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(list(map(str, args)), cwd=ROOT,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
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


def export_tables(output, summary):
    with (output / "run_status.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["model", "suite", "status", "decisions", "valid_decisions", "errors"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model, result in summary["models"].items():
            for name, suite in result.get("suites", {}).items():
                writer.writerow(dict(model=model, suite=name,
                                     status="FAILED_CASES" if suite["errors"] else "OK",
                                     decisions=suite["decisions"],
                                     valid_decisions=suite["decisions"] - suite["errors"],
                                     errors=suite["errors"]))
            if not result.get("suites"):
                writer.writerow(dict(model=model, suite="", status=result["status"]))
            for name in summary.get("expected_suites", []):
                if name not in result.get("suites", {}):
                    writer.writerow(dict(model=model, suite=name, status="NOT_RUN"))
    with (output / "failures.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["model", "suite", "case_id", "qid", "stage", "error", "message"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model, result in summary["models"].items():
            path = output / "runs" / model / model / "errors.jsonl"
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    record = json.loads(line)
                    for cid in record["case_ids"]:
                        writer.writerow({key: (cid if key == "case_id" else record.get(key))
                                         for key in fields})
            if result.get("error"):
                writer.writerow(dict(model=model, stage="model", error=result["error"],
                                     message=result.get("message", "")))
            for stage, report in result.get("reports", {}).items():
                if report["status"] != "OK":
                    writer.writerow(dict(model=model, stage=stage, error="ReportFailed",
                                         message=report["log"]))


def run_models(models, output, execute=run_command):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "model-logs"
    logs.mkdir(exist_ok=True)
    manifest = read(ROOT / "protocol_manifest.json")
    summary = dict(status="RUNNING", models={},
                   expected_suites=[s["name"] for s in manifest["suites"]])
    for model in models:
        result = dict(status="RUNNING", reports={}, suites={})
        summary["models"][model] = result
        write(output / "run_status.json", summary)
        model_root = output / "runs" / model
        log = logs / (model + ".log")
        result["log"] = str(log)
        try:
            print("START_MODEL", model, flush=True)
            code = execute([sys.executable, "-u", "-m", "system1bench.run",
                            "--model", model, "--batch-size", "1", "--continue-on-error",
                            "--output", str(model_root)], log)
            result["exit_code"] = code
            meta_path = model_root / model / "metadata.json"
            if meta_path.exists():
                meta = read(meta_path)
                result["suites"] = meta.get("suites", {})
            else:
                meta = {}
            if code != 0 or meta.get("status") != "DONE":
                tail = "\n".join(log.read_text(encoding="utf-8").splitlines()[-40:]) if log.exists() else ""
                result.update(status="FAILED", error="ModelRunFailed",
                              message=f"Exit code {code}; see {log}\n{tail}")
            else:
                count = sum(s["errors"] for s in result["suites"].values())
                result.update(status="DONE_WITH_ERRORS" if count else "OK", errors=count)
                # Each model has its own report root: an incomplete peer cannot block it.
                for stage in ("metrics", "report"):
                    report_log = logs / f"{model}-{stage}.log"
                    try:
                        rc = execute([sys.executable, "-u", "-m", "system1bench." + stage,
                                      "--results", str(model_root)], report_log)
                        result["reports"][stage] = dict(status="OK" if rc == 0 else "FAILED",
                                                        exit_code=rc, log=str(report_log))
                    except Exception as exc:
                        result["reports"][stage] = dict(status="FAILED", log=str(report_log),
                                                        error=type(exc).__name__, message=str(exc))
        except Exception as exc:
            result.update(status="FAILED", error=type(exc).__name__, message=str(exc))
        write(output / "run_status.json", summary)
        export_tables(output, summary)
        print("MODEL_RESULT", model, result["status"], flush=True)
    summary["status"] = "OK" if all(
        r["status"] == "OK" and all(s["status"] == "OK" for s in r["reports"].values())
        for r in summary["models"].values()) else "FINISHED_WITH_ERRORS"
    write(output / "run_status.json", summary)
    export_tables(output, summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=["english", "multilingual"],
                        default=["english", "multilingual"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = run_models(list(dict.fromkeys(args.models)), args.output)
    print("BATCH_FINISHED", summary["status"], args.output, flush=True)


if __name__ == "__main__":
    main()
