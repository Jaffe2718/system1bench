"""Deterministic serialization, source verification and answer validation."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def required_external_files(name, spec):
    """Exclude one archived API receipt that is not an upstream dataset file.

    CLINC's api_metadata.json is a historical API-response snapshot, not a
    file in the pinned Hub revision. Preparation uses README.md for label
    order and the pinned parquet for examples; it never reads this receipt.
    Keep the historical registry bytes and all actual input hashes unchanged.
    """
    return [f for f in spec["files"]
            if (name, f["path"]) != ("clinc", "api_metadata.json")]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    # Preserve dictionary insertion order: option order is experimental input.
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def labels(q):
    if q["type"] == "choice":
        return list(q["criteria"])
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    if q["type"] == "noul":
        return ["false", "true"]
    raise ValueError(q["type"])


def decode(q, answer):
    ls = labels(q)
    if q["type"] == "noul":
        p = float(answer["noul"])
        raw = [1 - p, p]
        pred = "true" if p >= 0.5 else "false"
    else:
        if set(answer["probabilities"]) != set(ls):
            raise ValueError("probability label mismatch")
        raw = [float(answer["probabilities"][k]) for k in ls]
        pred = answer["choice"] if q["type"] == "choice" else ls[max(range(len(ls)), key=raw.__getitem__)]
    if pred not in ls or not all(math.isfinite(p) and 0 <= p <= 1 for p in raw):
        raise ValueError("invalid prediction")
    if q["type"] == "choice" and raw[ls.index(pred)] != max(raw):
        raise ValueError("choice contradicts the probability argmax")
    if abs(sum(raw) - 1) > 0.02:  # up to 151 four-decimal rounded entries
        raise ValueError("invalid probability mass")
    if q["type"] == "score":
        score = float(answer["score"])
        if not math.isfinite(score) or not 0 <= score <= len(ls) - 1:
            raise ValueError("invalid expected score")
    return pred, [p / sum(raw) for p in raw]


def request(case):
    """Whitelist the inference payload; never forward source gold or metadata."""
    return {"state": case["state"], "questions": case["questions"]}
