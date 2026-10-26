# Hypothesis 3: independent option-count question bank

[Open the dedicated T4 notebook](https://colab.research.google.com/github/Jaffe2718/system1bench/blob/main/notebooks/hypothesis3_t4.ipynb)

Current workflow: **configure -> run questions -> download**. Research evaluation
is deliberately deferred. No accuracy, chance-adjusted score, F1, best-K estimate,
or model ranking is produced. Runtime errors and input audits are diagnostics.

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

The dedicated notebook has exactly three code steps: configure, collect, download.
It uses Colab's Python/PyTorch and one dependency install, with no Drive or smoke
inference. Each model loads once in a separate process and immediately starts this
bank. Default models are Laya English and Multilingual; other S1 models require
their own validated adapters and are not implicitly claimed as covered.

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
versions and logs so evaluation can be designed after inspecting the returned data.
Original data terms remain in force; see `NOTICE.md`.
