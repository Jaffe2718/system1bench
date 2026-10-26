# Hypothesis 3: independent option-count question bank

[Open the dedicated T4 notebook](https://colab.research.google.com/github/Jaffe2718/system1bench/blob/main/notebooks/hypothesis3_t4.ipynb)

Current workflow: **configure -> run questions -> evaluate -> download**.
CPU evaluation reports chance-adjusted single-choice scores, raw accuracy,
cluster intervals and separate multiselect metrics/baselines. It also reports
the full established model cohort's missing runs. See [EVALUATION.md](EVALUATION.md)
for the versioned rule and reuse of historical result ZIPs; no model reload is needed.

## Frozen bank

| Panel | Mother questions | Unique requests | Purpose |
|---|---:|---:|---|
| Native single choice | 800 | 1,600 | Original 2/3/4/5-option tasks, two display orders |
| Same-question paired | 600 | 9,569 | 2--5 options, two nested paths, two display orders |
| Multiple correct answers | 200 | 400 | MultiRC, 3--5 options, two display orders |
| Total | 1,600 | 11,569 | 12,767 typed decisions per model |

Native sources: 100 each from BoolQ, COPA, XNLI English, SocialIQA, AG News,
ARC-Challenge, SST5 choice and CommonsenseQA. These provide 200 mother questions
per K. Paired sources: 200 each from CommonsenseQA, Emotion and Banking77.
Only 2--5 options are ever sent to the model, including for Banking77.

MultiRC preserves real multiple-correct references. Exactly 200 eligible original
questions are sampled: 67 with K=3, 67 with K=4, and 66 with K=5. Each has at least
two positive options and at least one negative option. All original options are
retained; we do not pad or fabricate options. The API representation asks a noul
question for each candidate while retaining the full candidate list in state.
Raw P(true), native answers and the default 0.5 decoding are exported; later
evaluation can revisit the decision rule.

Identical requests are executed only once. `conditions` retains every assignment
that maps to that request; paired questions have 16 assignments each even though
the 9,600 assignments reduce to 9,569 unique requests. Do not treat repeated
orders/paths as independent mother questions.

## Isolation and review status

This directory, its bank and its notebook are independent of hypotheses 1/2.
Nothing changes `system1bench.prepare`, the old source registry, frozen protocol,
old runners, existing notebooks or published results.

For the six reused sources, the original main benchmark's exact 1,000-row sample
is reconstructed using its pinned source and seed 20260927. All 6,000 source IDs
and all normalized context groups touched by those rows are excluded. Native,
paired and multiselect panels have disjoint context groups; development and
confirmation splits are also context-group disjoint. Normalization is NFKC,
casefolding and whitespace collapse; this is not semantic/fuzzy deduplication.
The historical registry hash is provenance only: all source pins, label names and
sampling rules needed for this bank live here, so adding a source to the old
experiment does not change or block this bank.

The collaborator's final hypothesis-1/2 exclusion lists have not been supplied.
The bank explicitly records `hypothesis1_2_final_exclusions: NOT_PROVIDED` rather
than claiming full deduplication against unavailable future/private datasets.
Use `--exclude exclusion.json` to add source IDs/context hashes and rebuild into
a new directory. The JSON schema is shown in `exclusion.example.json`.

The bank is `BUILT_FOR_REVIEW`: labels are inherited from pinned sources and no
new distractor text is generated. Human semantic review remains pending. Public
benchmark exposure in model training is not ruled out. Data collection may proceed
as an exploratory run; the output carries these limitations.

## Rebuild / verify without a GPU

Use Python 3.12+ and the project's existing `pyarrow` dependency. No datasets
remote code, LLM calls, API token, GPU, or model weights are needed for building.

```bash
python research/hypothesis3_v1/build.py
python research/hypothesis3_v1/build.py --verify
python -m unittest discover -s research/hypothesis3_v1/tests -v
```

Default outputs are under ignored `data/hypothesis3_v1/`. The builder never writes
to the original `data/frozen.json` or `protocol_manifest.json`, and refuses to
overwrite an existing bank. `--offline` rebuilds from verified cached sources.
All upstream files have exact SHA256 pins in the independent `sources.json`.

## T4 collection

The dedicated notebook has exactly four code steps: configure, collect, evaluate, download/package.
It runs on Colab or a physical-server Jupyter kernel, reuses the host's CUDA
PyTorch and compatible dependencies, and invokes pip only for missing/incompatible
requirements. There is no Drive or smoke inference. Each model loads once in a separate process and immediately starts this
bank. The default queue is all 22 established local models, excluding Jev API.
Native families use their official interfaces; Llama/Qwen controls use coded
likelihood baselines. New backend code requires Colab validation. Memory,
backend, disk and HF-access blocks are reported explicitly. See COMPATIBILITY.md.

```bash
python research/hypothesis3_v1/run.py \
  --bank data/hypothesis3_v1/bank \
  --output data/hypothesis3_v1/results/run-001
```

Case errors are recorded and later cases/models continue. Journals flush per
request and can resume under the same bank, code and environment fingerprints.
Changing configuration requires a new output directory. An incomplete/truncated
journal is rejected instead of silently discarded. Keyboard interruption remains
available. `run_status.json`, `run_status.csv` and `failures.csv` describe execution,
not research performance. `predictions.jsonl` retains probabilities, native raw
answers, derived IDs, references, group IDs, condition aliases and token audits.

The download contains the exact bank, source/builder receipts, runner code, runtime
versions, logs and the evaluation directory for reproducible inspection.
Original data terms remain in force; see `NOTICE.md`.


## Adaptive collection

The notebook/CLI defaults to two CPU producers and one CUDA consumer. It starts
with up to 8 encoded question rows and grows after 64 consecutive 0.25-second
samples below 70% RAM/GPU memory. Pressure halves capacity; OOM batches split
and retry. Models still load and run sequentially. Use --scheduler serial for
the prior per-request path.

The fixed bank is unchanged. New outputs include adaptive_events.jsonl and
per-request batch metadata; use a fresh output directory. The notebook retains
configure -> collect -> evaluate -> download, Colab's installed torch, and no smoke/Drive.
See PROTOCOL.md for the complete memory, retry, timing and numerical policy.


## Evaluation without rerunning models

The evaluator supports any model using the H3 journal contract. The default
inference queue now contains all 22 local models. Resource/access/backend
blocks are distinguished from NOT_RUN and FAILED; code admission does not
claim that every new model has completed on T4.
Jev API is skipped by default; NanoJev remains included in the cohort.

Run the historical ZIP directly on CPU:

    python research/hypothesis3_v1/evaluate.py --results hypothesis3-t4-results.zip --output h3-evaluation

The notebook's third cell performs the same evaluation on the current run.
The fourth cell packages raw predictions and evaluation reports/tables together.


## Full cohort and monitor settings

See [COMPATIBILITY.md](COMPATIBILITY.md) for per-family dispatch, lazy pinned
source/checkpoint setup, fp16/NF4 provenance, T4 resource blocks and overrides.
The memory monitor now samples at a target 4 Hz and requires 64 consecutive
qualifying samples (about 16 seconds) before growing a Laya batch. It measures
RAM/GPU memory; other model adapters currently run serial complete requests.
No Windows dependency installation or model-weight download is part of local
verification.


## Reuse a server environment and HF model cache

See [SERVER.md](SERVER.md). In the notebook, set HF_CACHE_DIR to the server's
existing persistent Hub cache, or keep None to respect its existing settings.
MODEL_OVERRIDES can point to already downloaded local model/base directories.
ALLOW_MODEL_DOWNLOADS=False and CLI --offline forbid checkpoint/source fetching;
missing or partial cached files receive BLOCKED_CACHE. --cache-dir selects the
same persistent cache for every child model process, including Laya and Kev bases.

NanoJev no longer unconditionally asks its pinned predictor to import the
optional torch._native private API. The feature-detected flag/reason is recorded,
without changing the installed PyTorch or upstream checkpoint source.
