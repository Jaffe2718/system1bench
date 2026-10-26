"""Fetch only SHA-pinned public sources; raw text remains under ignored data/."""
import gzip
import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from .common import DATA, ROOT, read, required_external_files, sha, write


def fetch_url(url, target, expected):
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                target.write_bytes(response.read())
        except Exception as exc:
            raise RuntimeError(f"Download failed: {url} ({type(exc).__name__}: {exc})") from exc
    if sha(target) != expected:
        raise ValueError(f"Source hash mismatch: {target.name}")


def base(spec):
    target = DATA / spec["original_file"]
    if not target.exists():
        cached = hf_hub_download(spec["repo"], spec["source_file"], repo_type="dataset", revision=spec["revision"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, target)
    if sha(target) != spec["original_sha256"]:
        raise ValueError("Source hash mismatch")
    if str(target).endswith(".parquet"):
        rows = pq.read_table(target).to_pylist()
    else:
        opener = gzip.open if str(target).endswith(".gz") else open
        with opener(target, "rt", encoding="utf-8") as f:
            raw = f.read()
        try:
            rows = json.loads(raw)
        except json.JSONDecodeError:
            rows = [json.loads(s) for s in raw.splitlines() if s.strip()]
    dest = DATA / spec["rows_file"]
    dest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    if sha(dest) != spec["rows_sha256"] or len(rows) != spec["count"]:
        raise ValueError("Normalized source mismatch")
    return spec["suite"]


def main():
    specs = read(ROOT / "sources.json")
    with ThreadPoolExecutor(4) as pool:
        for name in pool.map(base, specs.values()):
            print("Verified", name, flush=True)
    write(DATA / "dataset_manifest.json", specs)
    ontology = read(ROOT / "massive_ontology.json")
    cached = hf_hub_download(ontology["source_repo"], ontology["source_file"], repo_type="dataset", revision=ontology["revision"])
    if sha(cached) != ontology["source_sha256"]:
        raise ValueError("Ontology source mismatch")
    with gzip.open(cached, "rt") as f:
        actual = sorted({json.loads(s)["label_text"] for s in f if s.strip()})
    if actual != ontology["labels"]:
        raise ValueError("Ontology mismatch")
    write(DATA / "datasets/massive_ontology/labels.json", ontology)
    for name, spec in read(ROOT / "external_sources.json").items():
        files = required_external_files(name, spec)
        if len(files) != len(spec["files"]):
            print("Not fetched: clinc/api_metadata.json (historical API receipt; not a dataset file or evaluation input)", flush=True)
        for file in files:
            url = (f'https://raw.githubusercontent.com/{spec["repo"]}/{spec["revision"]}/{file["path"]}'
                   if spec.get("provider", "github") == "github" else
                   f'https://huggingface.co/datasets/{spec["repo"]}/resolve/{spec["revision"]}/{file["path"]}')
            fetch_url(url, DATA / "external" / name / file["path"], file["sha256"])
        print("Verified", name, flush=True)


if __name__ == "__main__":
    main()
