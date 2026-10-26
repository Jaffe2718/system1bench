"""CPU-only, versioned evaluation of H3 directories or downloaded result ZIPs."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np

HERE = Path(__file__).resolve().parent
VERSION = "hypothesis3-evaluation-v1"
IDENTITY = ("base_id", "source", "source_id", "group", "panel", "split", "k", "conditions", "gold_ids")


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Files:
    def __init__(self, path, prefix=""):
        self.path, self.prefix = Path(path), prefix
        self.zip = zipfile.ZipFile(path) if self.path.is_file() else None

    def read(self, name):
        return self.zip.read(self.prefix + name) if self.zip else (self.path / self.prefix / name).read_bytes()

    def exists(self, name):
        return self.prefix + name in self.zip.namelist() if self.zip else (self.path / self.prefix / name).is_file()

    def json(self, name):
        return json.loads(self.read(name))

    def lines(self, name):
        return [json.loads(line) for line in self.read(name).decode("utf-8").splitlines()]

    def models(self):
        names = self.zip.namelist() if self.zip else [p.as_posix() for p in self.path.glob("*/metadata.json")]
        return {n.split("/")[-2] for n in names if n.endswith("/metadata.json")}

    def close(self):
        if self.zip:
            self.zip.close()


def bootstrap(values, info, repetitions, seed):
    """Source-stratified cluster bootstrap; aligned columns share all draws."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    rng = np.random.default_rng(seed)
    result = np.zeros((repetitions, values.shape[1]))
    for source in sorted({i[0] for i in info}):
        ids = [i for i, key in enumerate(info) if key[0] == source]
        groups = sorted({info[i][1] for i in ids})
        sums, counts = [], []
        for group in groups:
            cluster = [i for i in ids if info[i][1] == group]
            sums.append(values[cluster].sum(axis=0))
            counts.append(len(cluster))
        sums, counts = np.array(sums), np.array(counts)
        for start in range(0, repetitions, 200):
            ix = rng.integers(0, len(groups), (min(200, repetitions-start), len(groups)))
            result[start:start+len(ix)] += sums[ix].sum(axis=1) / counts[ix].sum(axis=1)[:, None] * len(ids)/len(info)
    return result


def ci(values):
    return [float(x) for x in np.quantile(values, [.025, .975])]


def validate_predictions(results, bank, manifest_sha, model, profile):
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", model):
        raise ValueError("Unsafe model identifier")
    meta = results.json(model + "/metadata.json")
    signature = meta["signature"]
    if signature["bank_manifest_sha256"] != manifest_sha or signature["profile"] != profile:
        raise ValueError("Model bank/profile fingerprint mismatch")
    raw = results.read(model + "/predictions.jsonl")
    if meta.get("status") != "DONE":
        raise ValueError("Incomplete model metadata")
    if sha(raw) != meta["predictions_sha256"]:
        raise ValueError("Prediction journal SHA256 mismatch")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    saved = {r["id"]: r for r in rows}
    if len(saved) != len(rows) or set(saved) != set(bank):
        raise ValueError("Missing, extra or duplicate prediction IDs; model is not fully evaluable")
    valid = {}
    for rid, original in bank.items():
        row = saved[rid]
        if row["request_sha256"] != original["request_sha256"] or any(row[k] != original[k] for k in IDENTITY):
            raise ValueError("Prediction identity/reference mismatch: " + rid)
        ok = (row.get("error") is None and row.get("decision_errors") == 0
              and row.get("complete_input") is True
              and set(row.get("answers", {})) == set(original["questions"])
              and set(row.get("audit", {})) == set(original["questions"])
              and all(a.get("complete") is True for a in row.get("audit", {}).values()))
        if ok:
            for qid, question in original["questions"].items():
                answer = row["answers"][qid]
                probs = answer["probabilities"]
                labels = list(question["criteria"]) if question["type"] == "choice" else ["false", "true"]
                if (answer.get("error") is not None or answer["labels"] != labels or len(probs) != len(labels)
                        or not np.isfinite(probs).all() or min(probs) < 0 or max(probs) > 1
                        or abs(sum(probs)-1) > 1e-8):
                    raise ValueError("Invalid saved probability contract: " + rid)
                native = answer["answer"]
                if question["type"] == "choice":
                    choice = native["choice"]
                    if choice not in labels or probs[labels.index(choice)] != max(probs):
                        raise ValueError("Choice disagrees with argmax: " + rid)
                    if set(native["probabilities"]) != set(labels):
                        raise ValueError("Native probability labels differ")
                    raw_values = list(native["probabilities"].values())
                    mass = sum(raw_values)
                    if (not np.isfinite(raw_values).all() or min(raw_values) < 0 or max(raw_values) > 1
                            or mass <= 0 or abs(mass-1) > .02):
                        raise ValueError("Invalid native probability distribution")
                    if any(abs(native["probabilities"][k]/mass-p) > 1e-8 for k,p in zip(labels,probs)):
                        raise ValueError("Native and saved probabilities disagree")
                    if answer["prediction"] != choice or row["predicted_ids"] != [choice]:
                        raise ValueError("Saved choice disagrees with raw answer")
                else:
                    p = float(native["noul"])
                    if not 0 <= p <= 1 or abs(p-probs[1]) > 1e-8 or answer["prediction"] != ("true" if p >= .5 else "false"):
                        raise ValueError("Invalid noul probability/threshold")
            if original["panel"] == "multi_select":
                expected = {qid for qid,a in row["answers"].items() if a["answer"]["noul"] >= .5}
                if set(row["predicted_ids"]) != expected or len(row["predicted_ids"]) != len(expected):
                    raise ValueError("Saved selected set disagrees with threshold")
        valid[rid] = ok
    return saved, valid, meta


def single_scores(model, bank, saved, valid, repetitions, seed):
    mothers = collections.defaultdict(list)
    for rid, row in bank.items():
        if row["panel"] == "multi_select":
            continue
        aliases = row["conditions"]
        correct = valid[rid] and set(saved[rid]["predicted_ids"]) == set(row["gold_ids"])
        mothers[row["panel"], row["base_id"], row["k"]].extend([(float(correct), float(valid[rid]))]*len(aliases))
    info = {r["base_id"]: r for r in bank.values()}
    scores, contrasts = [], []
    for panel in ("paired", "native"):
        sources = sorted({info[b]["source"] for p,b,k in mothers if p == panel})
        for source in (["ALL"] if panel == "paired" else []) + sources:
            for split in ("ALL", "development", "confirmation"):
                bases = sorted({b for p,b,k in mothers if p == panel
                                and (source == "ALL" or info[b]["source"] == source)
                                and (split == "ALL" or info[b]["split"] == split)})
                if not bases:
                    continue
                ks = sorted({k for p,b,k in mothers if p == panel and b in bases})
                if any((panel,b,k) not in mothers for b in bases for k in ks):
                    raise ValueError("Unbalanced within-source K coverage")
                vals = np.array([[np.mean([v[0] for v in mothers[panel,b,k]]) for k in ks] for b in bases])
                keys = [(info[b]["source"], info[b]["group"]) for b in bases]
                draws = bootstrap(vals, keys, repetitions, seed)
                for j,k in enumerate(ks):
                    accuracy = float(vals[:,j].mean())
                    low,high = ci(draws[:,j])
                    sl,sh = ci((draws[:,j]-1/k)/(1-1/k)*100)
                    scores.append(dict(model=model,panel=panel,source=source,split=split,k=k,mothers=len(bases),
                        context_groups=len(set(keys)),design_conditions=sum(len(mothers[panel,b,k]) for b in bases),
                        valid_condition_fraction=float(np.mean([np.mean([v[1] for v in mothers[panel,b,k]]) for b in bases])),
                        accuracy=accuracy,accuracy_ci_low=low,accuracy_ci_high=high,chance_baseline=1/k,
                        chance_adjusted_score=100*(accuracy-1/k)/(1-1/k),score_ci_low=sl,score_ci_high=sh))
                if panel == "paired" and ks == [2,3,4,5]:
                    d=(vals[:,3]-.2)/.8-(vals[:,0]-.5)/.5
                    lo,hi=ci(((draws[:,3]-.2)/.8-(draws[:,0]-.5)/.5)*100)
                    contrasts.append(dict(model=model,source=source,split=split,comparison="K5_minus_K2",
                                          score_difference=float(d.mean()*100),ci_low=lo,ci_high=hi,mothers=len(bases)))
    return scores, contrasts


def multi_scores(model, bank, saved, valid, repetitions, seed):
    mothers = collections.defaultdict(list)
    info = {}
    for rid,r in bank.items():
        if r["panel"] != "multi_select":
            continue
        info[r["base_id"]] = r
        gold = set(r["gold_ids"])
        pred = set(saved[rid]["predicted_ids"]) if valid[rid] else set()
        tp,fp,fn = len(gold & pred),len(pred-gold),len(gold-pred)
        em = float(pred == gold) if valid[rid] else 0.
        coin=2.**(-r["k"])
        mothers[r["base_id"]].append([em,2*tp/(2*tp+fp+fn),tp,fp,fn,
            float(len(pred)==r["k"] and valid[rid]),float(not pred and valid[rid]),len(pred),float(valid[rid]),
            (em-coin)/(1-coin)*100,2*len(gold)/(r["k"]+len(gold)),len(gold),coin])
    scores=[]
    for split in ("ALL","development","confirmation"):
        for k in ("ALL",3,4,5):
            bases=sorted(b for b in mothers if (split=="ALL" or info[b]["split"]==split)
                         and (k=="ALL" or info[b]["k"]==k))
            if not bases:
                continue
            values=np.array([np.mean(mothers[b],axis=0) for b in bases])
            keys=[(info[b]["source"],info[b]["group"]) for b in bases]
            mean=values.mean(axis=0)
            draws=bootstrap(values,keys,repetitions,seed)
            el,eh=ci(draws[:,0]);fl,fh=ci(draws[:,1]);cl,ch=ci(draws[:,9])
            tp,fp,fn=values[:,2:5].sum(axis=0)
            scores.append(dict(model=model,panel="multi_select",split=split,k=k,mothers=len(bases),
                paragraph_groups=len(set(keys)),valid_condition_fraction=float(mean[8]),
                exact_match=float(mean[0]),exact_match_ci_low=el,exact_match_ci_high=eh,
                macro_set_f1=float(mean[1]),f1_ci_low=fl,f1_ci_high=fh,
                micro_f1=float(2*tp/(2*tp+fp+fn)),precision=float(tp/(tp+fp)) if tp+fp else 0.,
                recall=float(tp/(tp+fn)),all_selected_rate=float(mean[5]),empty_selected_rate=float(mean[6]),
                mean_selected=float(mean[7]),mean_gold=float(mean[11]),
                fair_coin_exact_match_baseline=float(mean[12]),fair_coin_adjusted_em_score=float(mean[9]),
                fair_coin_score_ci_low=cl,fair_coin_score_ci_high=ch,all_selected_baseline_f1=float(mean[10])))
    return scores


def write_table(path, rows):
    with path.open("w",encoding="utf-8",newline="") as stream:
        if not rows:
            stream.write("model,status\n")
            return
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def evaluate(results_path, output, bank_path=None, repetitions=2000, seed=20261026):
    if repetitions < 100:
        raise ValueError("Use at least 100 bootstrap repetitions")
    results=Files(results_path)
    bankfiles=Files(bank_path) if bank_path else Files(results_path,"question_bank/")
    try:
        manifest=bankfiles.json("manifest.json")
        for name,receipt in manifest["files"].items():
            if sha(bankfiles.read(name)) != receipt["sha256"]:
                raise ValueError("Question bank hash mismatch: " + name)
        bankrows=[r for panel in ("native","paired","multi_select") for r in bankfiles.lines(panel+".jsonl")]
        for r in bankrows:
            payload=json.dumps(dict(state=r["state"],questions=r["questions"]),ensure_ascii=False,separators=(",",":"))
            if sha(payload.encode()) != r["request_sha256"]:
                raise ValueError("Request hash mismatch: " + r["id"])
        run_status=results.json("run_status.json") if results.exists("run_status.json") else {}
        profile=run_status.get("profile","full")
        if profile not in ("full","pilot"):
            raise ValueError("Unknown collection profile")
        if profile == "pilot":
            pilot=set(bankfiles.json("pilot_base_ids.json"))
            bankrows=[r for r in bankrows if r["panel"]=="paired" and r["base_id"] in pilot]
        bank={r["id"]:r for r in bankrows}
        if len(bank)!=len(bankrows):
            raise ValueError("Duplicate bank request IDs")
        manifest_sha=sha(bankfiles.read("manifest.json"))
        cohort=json.loads((HERE/"model_cohort.json").read_text(encoding="utf-8"))
        models=set(cohort["models"])|results.models()|set(run_status.get("models",{}))
        coverage,singles,multi,contrasts=[],[],[],[]
        for model in sorted(models):
            entry=dict(model=model,status="NOT_RUN",expected_requests=len(bank),saved_requests=0,invalid_requests=0,reason="",precision="",quantized="",adapter="")
            coverage.append(entry)
            if model=="jev-1.13.0" and not results.exists(model+"/metadata.json"):
                entry.update(status="SKIPPED_API",reason="Jev API excluded by user; NanoJev remains in cohort")
                continue
            if not results.exists(model+"/metadata.json"):
                collected = run_status.get("models",{}).get(model,{})
                state = collected.get("status", "")
                if state == "FAILED" or state.startswith("BLOCKED_") or state == "SKIPPED_API":
                    entry.update(status=state,reason=collected.get("reason","Collection process failed; no complete metadata"))
                continue
            try:
                saved,valid,meta=validate_predictions(results,bank,manifest_sha,model,profile)
                rows,differences=single_scores(model,bank,saved,valid,repetitions,seed)
                mult=multi_scores(model,bank,saved,valid,repetitions,seed)
                singles.extend(rows);contrasts.extend(differences);multi.extend(mult)
                invalid=sum(not v for v in valid.values())
                runtime=meta["signature"].get("model", {})
                entry.update(precision=str(runtime.get("dtype","unspecified")),
                             quantized=runtime.get("quantized",False) if "dtype" in runtime else "",
                             adapter=runtime.get("adapter","unspecified"))
                entry.update(status="COMPLETE_WITH_INVALID" if invalid else "COMPLETE",
                             saved_requests=len(saved),invalid_requests=invalid)
            except Exception as exc:
                entry.update(status="UNEVALUABLE",reason=type(exc).__name__+": "+str(exc))
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        protocol=dict(version=VERSION,flow="configure-run-evaluate-download",evaluation_performed=True,
            exploratory=True,development_and_confirmation_reported_separately=True,bootstrap_repetitions=repetitions,
            bootstrap_seed=seed,ci="pointwise 95% source-stratified context-cluster percentile; not multiplicity adjusted",
            single_primary="100*(accuracy-1/K)/(1-1/K); equal mothers, alias-weighted within mother/K",
            multiselect="p(true)>=0.5; exact match and macro set F1; fair-coin EM baseline 2^-K, all-selected F1",
            failure_policy="Complete model journals only. Recorded case errors/incomplete audits score zero, with valid coverage reported. Missing models have no scores.",
            interpretation="No pooled single/multiselect rank, no automatic optimal K, no pure K causal claim.",
            model_scope="All established models; Jev API skipped by default; missing adapters/runs remain NOT_RUN.",
            evaluator_sha256=sha(Path(__file__).read_bytes().replace(b"\r\n",b"\n")),
            cohort_sha256=sha((HERE/"model_cohort.json").read_bytes().replace(b"\r\n",b"\n")),
            bank_manifest_sha256=manifest_sha,profile=profile,numpy=np.__version__)
        summary=dict(protocol=protocol,coverage=coverage,single_choice=singles,multi_select=multi,contrasts=contrasts)
        (output/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        (output/"protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        write_table(output/"model_coverage.csv",coverage)
        write_table(output/"paired_scores.csv",[r for r in singles if r["panel"]=="paired"])
        write_table(output/"native_scores.csv",[r for r in singles if r["panel"]=="native"])
        write_table(output/"multi_select_scores.csv",multi)
        write_table(output/"paired_contrasts.csv",contrasts)
        report=render_report(summary)
        (output/"REPORT.md").write_text(report,encoding="utf-8")
        status=dict(flow=protocol["flow"],evaluation_performed=True,evaluation_version=VERSION,
                    models_evaluated=sum(e["status"].startswith("COMPLETE") for e in coverage),
                    models_not_run=sum(e["status"]=="NOT_RUN" for e in coverage),
                    models_unevaluable=sum(e["status"] in ("UNEVALUABLE","FAILED") for e in coverage),
                    models_blocked=sum(e["status"].startswith("BLOCKED_") for e in coverage),
                    collection_status=run_status.get("status","UNKNOWN"),summary_sha256=sha((output/"summary.json").read_bytes()))
        (output/"status.json").write_text(json.dumps(status,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        return summary
    finally:
        results.close();bankfiles.close()


def render_report(summary):
    protocol=summary["protocol"]
    lines=["# Hypothesis 3 评价", "", "配置 → 跑题 → 评价 → 下载。以下为探索性结果，缺失模型不填 0 分。", "",
           "单选主指标：100 × (准确率 − 1/K) / (1 − 1/K)。随机猜测为 0 分，全对 100 分，允许负分。",
           "先对每道母题内的路径/顺序等权平均，再汇总母题；去重请求按条件别名恢复权重。", "",
           "## 同题配对单选", "", "表内是校正分和逐点 95% 聚类 bootstrap 区间。", "",
           "| 模型 | 子集 | K=2 | K=3 | K=4 | K=5 |", "|---|---|---:|---:|---:|---:|"]
    rows=[r for r in summary["single_choice"] if r["panel"]=="paired" and r["source"]=="ALL"]
    for model,split in sorted({(r["model"],r["split"]) for r in rows}):
        values={r["k"]:r for r in rows if r["model"]==model and r["split"]==split}
        formatted=[f"{values[k]['chance_adjusted_score']:.2f} [{values[k]['score_ci_low']:.2f}, {values[k]['score_ci_high']:.2f}]" if k in values else "—" for k in (2,3,4,5)]
        lines.append("| "+model+" | "+split+" | "+" | ".join(formatted)+" |")
    lines += ["", "来源分层、原始准确率和 K5−K2 配对差值见 CSV；原生单选按来源分别报告，不用于跨来源推断 K 效应。", "",
              "## 不定项多选", "", "固定 p(true) ≥ 0.5；单选和多选分数分别报告。", "",
              "| 模型 | 母题 | 整题全对 | 宏平均集合 F1 | 全选率 | 全选基线 F1 | 去随机基线 EM 分 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary["multi_select"]:
        if r["split"]=="ALL" and r["k"]=="ALL":
            lines.append(f"| {r['model']} | {r['mothers']} | {100*r['exact_match']:.2f}% | {100*r['macro_set_f1']:.2f}% | {100*r['all_selected_rate']:.2f}% | {100*r['all_selected_baseline_f1']:.2f}% | {r['fair_coin_adjusted_em_score']:.2f} |")
    lines += ["", "多选随机策略明确为每个候选独立以 0.5 概率勾选，整题基线为 2^(−K)，不是 1/K。",
              "去基线 EM 在每题内计算再平均；不与单选校正分合并排名。不同 K 的多选题不是同一批母题。", "",
              "## 模型覆盖", "", "| 模型 | 状态 | 精度 | 量化 | 无效请求 | 原因 |", "|---|---|---|---|---:|---|"]
    for r in summary["coverage"]:
        reason=r["reason"].replace("|","/").replace("\n"," ")
        lines.append(f"| {r['model']} | {r['status']} | {r['precision']} | {r['quantized']} | {r['invalid_requests']} | {reason} |")
    lines += ["", "完整日志中已记录的失败/不完整输入按错误计分；缺失或损坏的模型日志不评分，不丢弃后只报告成功题。",
              f"区间采用 {protocol['bootstrap_repetitions']} 次按来源分层、按上下文簇重采样；同母题各 K 保持配对。",
              "区间未作多重比较校正；本方法在已有 Laya 结果之后制定，该批仍属于探索性分析。",
              "选项扩充同时改变候选语义、难度、长度和顺序；不自动宣布最优 K，不作训练机制归因。",
              "人工语义复核、协作者最终 H1/H2 排除清单仍待补齐；公开题训练暴露不能排除。", ""]
    return "\n".join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,required=True,help="Collection directory or downloaded ZIP")
    parser.add_argument("--bank",type=Path,help="Bank directory; defaults to question_bank/ inside results")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--bootstrap",type=int,default=2000)
    parser.add_argument("--seed",type=int,default=20261026)
    args=parser.parse_args()
    result=evaluate(args.results,args.output,args.bank,args.bootstrap,args.seed)
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(render_report(result),flush=True)


if __name__=="__main__":
    main()
