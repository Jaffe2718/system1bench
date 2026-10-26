"""Build an isolated, source-pinned option-count bank. No model calls."""
import argparse
import collections
import hashlib
import json
import random
import tarfile
import unicodedata
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEED = 20261026
LEGACY_SEED = 20260927
NATIVE = {"boolq": 2, "copa": 2, "xnli_en": 3, "socialiqa": 3,
          "ag_news": 4, "arc": 4, "sst5": 5, "commonsenseqa": 5}
PAIRED = ("commonsenseqa", "emotion", "banking77")
LEGACY = ("boolq", "xnli_en", "ag_news", "sst5", "emotion", "banking77")


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def text_sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def group_hash(text):
    return digest(normalized(text))


def rng(*parts):
    return random.Random(int(digest([SEED, *parts]), 16))


def dump(path, value):
    Path(path).write_bytes(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n")


def source_rows(spec, cache, offline=False):
    path = cache / spec["cache_file"]
    if not path.exists():
        if offline:
            raise FileNotFoundError(path)
        print("DOWNLOAD", spec["name"], flush=True)
        with urllib.request.urlopen(spec["url"], timeout=120) as response:
            raw = response.read()
        if hashlib.sha256(raw).hexdigest() != spec["sha256"]:
            raise ValueError("Downloaded source hash mismatch: " + spec["name"])
        path.write_bytes(raw)
    if file_sha(path) != spec["sha256"]:
        raise ValueError("Cached source hash mismatch: " + spec["name"])
    if spec["kind"] == "parquet":
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    if spec["kind"] == "jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if spec["kind"] == "tgz":
        # Read one named member, never extract arbitrary archive paths.
        with tarfile.open(path) as archive:
            return [json.loads(line) for line in archive.extractfile(spec["member"]) if line.strip()]
    raise ValueError("Unknown source format")


def base_item(source, source_id, state, instruction, options, gold, group_text):
    return dict(source=source, source_id=str(source_id),
                base_id=f"h3:{source}:{source_id}", state=state, instruction=instruction,
                options=options, gold_indices=gold, source_group_sha256=group_hash(group_text))


def convert(source, rows, labels=None):
    result = []
    if source == "multirc":
        grouped = collections.defaultdict(list)
        for row in rows:
            grouped[(row["idx"]["paragraph"], row["idx"]["question"])].append(row)
        for (pid, qid), group in sorted(grouped.items()):
            group.sort(key=lambda r: r["idx"]["answer"])
            if any(r["label"] not in (0, 1) for r in group):
                continue
            row = group[0]
            result.append(base_item(source, f"{pid}:{qid}",
                dict(passage=row["paragraph"], question=row["question"]),
                "Select all answers supported by the passage. More than one answer may be correct.",
                [r["answer"] for r in group], [i for i, r in enumerate(group) if r["label"] == 1], row["paragraph"]))
        return result
    for index, row in enumerate(rows):
        sid = index
        if source == "boolq":
            state = dict(passage=row["passage"], question=row["question"])
            instruction = "Based on the passage, is the answer to the question yes? Choose one supplied answer."
            options, gold, context = ["No", "Yes"], [int(row["answer"])], row["passage"]
        elif source == "xnli_en":
            state = dict(premise=row["premise"], hypothesis=row["hypothesis"])
            instruction = "What is the relationship between the premise and hypothesis?"
            options = ["Entailment: the premise implies the hypothesis is true.",
                       "Neutral: the premise neither implies nor contradicts the hypothesis.",
                       "Contradiction: the premise implies the hypothesis is false."]
            gold, context = [int(row["label"])], row["premise"]
        elif source == "ag_news":
            state, instruction = dict(article=row["text"]), "Which supplied topic best describes the article?"
            options = ["World news and international politics", "Sports", "Business and economy", "Science and technology"]
            gold, context = [int(row["label"])], row["text"]
        elif source == "sst5":
            state, instruction = dict(text=row["text"]), "Which supplied label best describes the sentiment of the text?"
            options = ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]
            gold, context = [int(row["label"])], row["text"]
        elif source in ("emotion", "banking77"):
            state = dict(text=row["text"])
            instruction = ("Which supplied emotion is most strongly expressed in the text?" if source == "emotion" else
                           "Which supplied banking intent best describes the user's message?")
            options = labels[source]
            target = options[int(row["label"])] if source == "emotion" else row["label_text"].replace("_", " ")
            gold, context = [options.index(target)], row["text"]
        elif source == "copa":
            state = dict(premise=row["premise"])
            instruction = "Which supplied option is the most plausible " + row["question"] + " of the premise?"
            options = [row["choice1"], row["choice2"]]
            sid, gold, context = row["idx"], [int(row["label"])], row["premise"]
        elif source == "socialiqa":
            state = dict(context=row["context"], question=row["question"])
            instruction = "Given the social situation, select the single best answer to the question."
            options = [row["answerA"], row["answerB"], row["answerC"]]
            gold, context = ["ABC".index(row["correct"])], row["context"]
        elif source in ("arc", "commonsenseqa"):
            state = dict(question=row["question"])
            instruction = "Select the single best answer to the question from the supplied candidates."
            options = row["choices"]["text"]
            sid, gold, context = row["id"], [row["choices"]["label"].index(row["answerKey"])], row["question"]
        else:
            raise ValueError(source)
        result.append(base_item(source, sid, state, instruction, list(options), gold, context))
    return result


def valid_base(item):
    texts = item["options"]
    return (len(texts) >= 2 and all(isinstance(t, str) and t.strip() for t in texts)
            and len({normalized(t) for t in texts}) == len(texts)
            and len(set(item["gold_indices"])) == len(item["gold_indices"])
            and all(0 <= i < len(texts) for i in item["gold_indices"]))


def sample(items, count, blocked_ids, blocked_groups, reserved, tag, multi=False):
    candidates = [b for b in items if valid_base(b)
                  and b["base_id"] not in blocked_ids
                  and f"{b['source']}:{b['source_id']}" not in blocked_ids
                  and b["source_group_sha256"] not in blocked_groups | reserved]
    buckets = collections.defaultdict(list)
    for b in candidates:
        stratum = len(b["options"]) if multi else b["gold_indices"][0]
        buckets[stratum].append(b)
    for values in buckets.values():
        values.sort(key=lambda b: digest([SEED, tag, b["base_id"]]))
    selected, selected_groups = [], set()
    while len(selected) < count:
        changed = False
        for key in sorted(buckets):
            while buckets[key]:
                item = buckets[key].pop(0)
                if multi or item["source_group_sha256"] not in selected_groups:
                    selected.append(item)
                    selected_groups.add(item["source_group_sha256"])
                    changed = True
                    break
            if len(selected) == count:
                break
        if not changed:
            raise ValueError(f"Insufficient eligible independent groups for {tag}: {len(selected)}/{count}")
    reserved.update(selected_groups)
    return selected


def splits(items, tag):
    groups = sorted({b["source_group_sha256"] for b in items}, key=lambda g: digest([SEED, tag, g]))
    development = set(groups[:max(1, round(len(groups) * .2))])
    return {b["base_id"]: "development" if b["source_group_sha256"] in development else "confirmation" for b in items}


def pool_for(item, size, tag):
    selected = list(item["gold_indices"])
    wrong = [i for i in range(len(item["options"])) if i not in selected]
    rng(item["base_id"], tag, "wrong").shuffle(wrong)
    selected += wrong[:size - len(selected)]
    rng(item["base_id"], tag, "ids").shuffle(selected)
    pool = [{"id": chr(65 + i), "text": item["options"][index]} for i, index in enumerate(selected)]
    gold_ids = [chr(65 + i) for i, index in enumerate(selected) if index in item["gold_indices"]]
    return pool, gold_ids


def record(item, panel, split, pool, gold_ids, order, aliases):
    options = {c["id"]: c["text"] for c in pool}
    ordered = [{"id": cid, "text": options[cid]} for cid in order]
    if panel == "multi_select":
        state = dict(item["state"], candidates=ordered)
        questions = {cid: dict(type="noul", instructions=
            f"Is candidate {cid} a correct answer to the question, based only on the passage? Judge independently; multiple candidates may be correct.") for cid in order}
        gold = {cid: {"label": "true" if cid in gold_ids else "false"} for cid in order}
    else:
        state = item["state"]
        questions = {"answer": dict(type="choice", instructions=item["instruction"],
                                    criteria={cid: options[cid] for cid in order})}
        gold = {"answer": {"label": gold_ids[0]}}
    request = dict(state=state, questions=questions)
    request_sha = digest(request)
    return dict(id=f"{item['base_id']}:{panel}:{request_sha[:16]}", base_id=item["base_id"],
                source=item["source"], source_id=item["source_id"],
                group=item["source_group_sha256"], panel=panel, split=split,
                k=len(order), language="en", conditions=aliases,
                candidates=ordered, gold_ids=gold_ids, gold=gold,
                request_sha256=request_sha, **request)


def single_records(items, panel, split_map):
    records, mothers = [], []
    for index, item in enumerate(items):
        size = 5 if panel == "paired" else len(item["options"])
        pool, gold_ids = pool_for(item, size, panel)
        gold_id = gold_ids[0]
        mothers.append(dict(base_id=item["base_id"], source=item["source"], source_id=item["source_id"],
                            group=item["source_group_sha256"], panel=panel, split=split_map[item["base_id"]],
                            state=item["state"], instruction=item["instruction"], pool=pool, gold_ids=gold_ids))
        seen = {}
        wrong = [c["id"] for c in pool if c["id"] != gold_id]
        rng(item["base_id"], "paths").shuffle(wrong)
        for chain in range(2 if panel == "paired" else 1):
            path = wrong if chain == 0 else wrong[::-1]
            for k in (range(2, 6) if panel == "paired" else [size]):
                selected = path[:k - 1]
                for order_index in range(2):
                    distractors = selected[:]
                    rng(item["base_id"], chain, k, order_index).shuffle(distractors)
                    position = (index + chain + order_index) % k
                    order = distractors[:position] + [gold_id] + distractors[position:]
                    alias = dict(chain=chain, order=order_index, k=k)
                    row = record(item, panel, split_map[item["base_id"]], pool, gold_ids, order, [alias])
                    if row["request_sha256"] in seen:
                        seen[row["request_sha256"]]["conditions"].append(alias)
                    else:
                        seen[row["request_sha256"]] = row
                        records.append(row)
    return records, mothers


def multi_records(items, split_map):
    records = []
    for item in items:
        pool, gold_ids = pool_for(item, len(item["options"]), "multi_select")
        order = [c["id"] for c in pool]
        for rotation in range(2):
            display = order[rotation:] + order[:rotation]
            records.append(record(item, "multi_select", split_map[item["base_id"]], pool, gold_ids,
                                  display, [dict(order=rotation, k=len(display))]))
    return records


def audit_bank(records, mothers, excluded_ids, excluded_groups):
    seen_ids, seen_requests = set(), set()
    group_splits, group_panels = {}, {}
    by_base = collections.defaultdict(list)
    for row in records:
        assert row["id"] not in seen_ids and row["request_sha256"] not in seen_requests
        seen_ids.add(row["id"])
        seen_requests.add(row["request_sha256"])
        assert row["k"] in (2, 3, 4, 5)
        ids = [c["id"] for c in row["candidates"]]
        assert len(ids) == len(set(ids)) == row["k"]
        assert set(row["gold_ids"]) <= set(ids)
        assert row["request_sha256"] == digest(dict(state=row["state"], questions=row["questions"]))
        assert row["base_id"] not in excluded_ids
        assert f"{row['source']}:{row['source_id']}" not in excluded_ids
        assert row["group"] not in excluded_groups
        assert group_splits.setdefault(row["group"], row["split"]) == row["split"]
        assert group_panels.setdefault(row["group"], row["panel"]) == row["panel"]
        if row["panel"] == "multi_select":
            assert 2 <= len(row["gold_ids"]) < row["k"]
            assert set(row["questions"]) == set(ids) == set(row["gold"])
            assert all(q["type"] == "noul" for q in row["questions"].values())
            assert {cid for cid, g in row["gold"].items() if g["label"] == "true"} == set(row["gold_ids"])
        else:
            assert len(row["gold_ids"]) == 1
            assert row["gold"]["answer"]["label"] == row["gold_ids"][0]
            assert list(row["questions"]["answer"]["criteria"]) == ids
        by_base[row["base_id"]].append(row)
    for mother in mothers:
        rows = by_base[mother["base_id"]]
        pool = {c["id"]: c["text"] for c in mother["pool"]}
        assert all(r["gold_ids"] == mother["gold_ids"] and r["state"] == mother["state"] for r in rows)
        assert all(all(pool[c["id"]] == c["text"] for c in r["candidates"]) for r in rows)
        if mother["panel"] == "paired":
            for chain in range(2):
                previous = set()
                for k in range(2, 6):
                    matched = [r for r in rows if any(a["chain"] == chain and a["k"] == k for a in r["conditions"])]
                    assert matched and {a["order"] for r in matched for a in r["conditions"] if a["chain"] == chain and a["k"] == k} == {0, 1}
                    candidates = set(c["id"] for c in matched[0]["candidates"])
                    assert previous < candidates
                    assert all(set(c["id"] for c in r["candidates"]) == candidates for r in matched)
                    previous = candidates
    return dict(status="PASS", requests=len(records), decisions=sum(len(r["questions"]) for r in records),
                unique_bases=len(by_base), source_groups=len(group_splits),
                panel_requests=dict(collections.Counter(r["panel"] for r in records)),
                panel_k_requests=dict(collections.Counter(f"{r['panel']}/K={r['k']}" for r in records)),
                split_requests=dict(collections.Counter(r["split"] for r in records)),
                excluded_source_ids=len(excluded_ids), excluded_text_groups=len(excluded_groups),
                legacy_overlap=0, cross_panel_group_overlap=0, development_confirmation_group_overlap=0)


def prepare(cache, output, offline=False, exclude_files=()):
    if output.exists():
        raise FileExistsError("Refusing to overwrite a bank; use --verify or a new output directory")
    catalog = json.loads((HERE / "sources.json").read_text(encoding="utf-8"))
    cache.mkdir(parents=True, exist_ok=True)
    raw = {s["name"]: source_rows(s, cache, offline) for s in catalog["sources"]}
    labels = {"emotion": catalog["emotion_label_names"],
              "banking77": sorted({r["label_text"].replace("_", " ") for r in raw["banking77"]})}
    bases = {name: convert(name, rows, labels) for name, rows in raw.items()}
    excluded_ids, excluded_groups = set(), set()
    for source in LEGACY:
        old = random.Random(LEGACY_SEED).sample(range(len(raw[source])), min(1000, len(raw[source])))
        for index in old:
            excluded_ids.add(f"{source}:{index}")
            excluded_groups.add(bases[source][index]["source_group_sha256"])
    external_receipts = []
    for filename in exclude_files:
        payload = json.loads(Path(filename).read_text(encoding="utf-8"))
        if not payload.get("source_ids") and not payload.get("group_sha256"):
            raise ValueError("An empty example is not a final exclusion receipt: " + str(filename))
        excluded_ids.update(payload.get("source_ids", []))
        excluded_groups.update(payload.get("group_sha256", []))
        external_receipts.append(dict(file=Path(filename).name, sha256=file_sha(filename)))
    reserved, records, mothers = set(), [], []
    for source, k in NATIVE.items():
        candidates = [b for b in bases[source] if len(b["options"]) == k and len(b["gold_indices"]) == 1]
        selected = sample(candidates, 100, excluded_ids, excluded_groups, reserved, "native/" + source)
        rows, parents = single_records(selected, "native", splits(selected, "native/" + source))
        records.extend(rows)
        mothers.extend(parents)
    for source in PAIRED:
        selected = sample(bases[source], 200, excluded_ids, excluded_groups, reserved, "paired/" + source)
        rows, parents = single_records(selected, "paired", splits(selected, "paired/" + source))
        records.extend(rows)
        mothers.extend(parents)
    multi = [b for b in bases["multirc"] if len(b["options"]) in (3, 4, 5) and 2 <= len(b["gold_indices"]) < len(b["options"])]
    selected = sample(multi, 200, excluded_ids, excluded_groups, reserved, "multi_select", multi=True)
    records.extend(multi_records(selected, splits(selected, "multi_select")))
    audit = audit_bank(records, mothers, excluded_ids, excluded_groups)
    audit["hypothesis1_2_final_exclusions"] = "provided; coverage depends on supplied lists" if external_receipts else "NOT_PROVIDED"
    audit["external_exclusion_receipts"] = external_receipts
    audit["human_semantic_review"] = "PENDING; no new distractor text was generated"
    pilot = set()
    for source in PAIRED:
        development = [m for m in mothers if m["panel"] == "paired" and m["source"] == source and m["split"] == "development"]
        pilot.update(m["base_id"] for m in sorted(development, key=lambda m: digest([SEED, "pilot", m["base_id"]]))[:20])
    assert len(pilot) == 60
    output.mkdir(parents=True)
    files = {}
    for panel in ("native", "paired", "multi_select"):
        rows = [r for r in records if r["panel"] == panel]
        path = output / (panel + ".jsonl")
        path.write_bytes(b"".join(encode(r) + b"\n" for r in rows))
        files[path.name] = dict(sha256=file_sha(path), requests=len(rows),
                                decisions=sum(len(r["questions"]) for r in rows),
                                bases=len({r["base_id"] for r in rows}))
    path = output / "mothers.jsonl"
    path.write_bytes(b"".join(encode(m) + b"\n" for m in mothers))
    files[path.name] = dict(sha256=file_sha(path), rows=len(mothers))
    dump(output / "pilot_base_ids.json", sorted(pilot))
    dump(output / "exclusions.json", dict(source_ids=sorted(excluded_ids), group_sha256=sorted(excluded_groups)))
    dump(output / "audit.json", audit)
    for filename in ("pilot_base_ids.json", "exclusions.json", "audit.json"):
        files[filename] = dict(sha256=file_sha(output / filename))
    manifest = dict(version="hypothesis3-v1", seed=SEED, language="en", model_calls=0,
                    status="BUILT_FOR_REVIEW", source_catalog_sha256=text_sha(HERE / "sources.json"),
                    builder_sha256=text_sha(__file__), files=files,
                    isolation="Independent bank only; original frozen protocol and runners unchanged",
                    sources=catalog["sources"])
    dump(output / "manifest.json", manifest)
    return manifest, audit


def verify(output):
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_catalog_sha256"] == text_sha(HERE / "sources.json")
    assert manifest["builder_sha256"] == text_sha(__file__)
    for filename, receipt in manifest["files"].items():
        assert file_sha(output / filename) == receipt["sha256"], filename
    records = [json.loads(line) for panel in ("native", "paired", "multi_select")
               for line in (output / (panel + ".jsonl")).read_text(encoding="utf-8").splitlines()]
    mothers = [json.loads(line) for line in (output / "mothers.jsonl").read_text(encoding="utf-8").splitlines()]
    exclusions = json.loads((output / "exclusions.json").read_text(encoding="utf-8"))
    return audit_bank(records, mothers, set(exclusions["source_ids"]), set(exclusions["group_sha256"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/hypothesis3_v1/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data/hypothesis3_v1/bank")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.verify:
        audit = verify(args.output)
    else:
        _, audit = prepare(args.cache, args.output, args.offline, args.exclude)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
