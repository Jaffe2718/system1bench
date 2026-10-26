# Hypothesis 3 evaluation v1

The notebook workflow is **configure -> collect -> evaluate -> download**.
Evaluation runs on CPU with NumPy, requires no new dependencies, and never loads
models or modifies predictions. It accepts both serial and adaptive journals.

This rule was specified after inspecting the initial Laya run. That run and its
already viewed development/confirmation subsets remain exploratory. Future
model outputs should use this versioned rule without outcome-driven changes.

## Single-choice primary score

For each mother question and K, average correctness over the designed paths and
display orders. Restore deduplicated requests using all their `conditions`
aliases, so the four paired conditions at each K have equal weight. Then average
mothers equally. Report:

    score(K) = 100 * (accuracy(K) - 1/K) / (1 - 1/K)

Uniform guessing has expected score 0; perfect answers score 100. Negative scores
are allowed. This is a chance-adjusted score, not an accuracy percentage. Raw
accuracy is retained as a secondary metric.

The paired panel is primary for cross-K comparisons. Report ALL and each source,
separately for ALL/development/confirmation. Native tasks are reported by source:
different native K values come from different tasks and cannot isolate K.
Do not merge native, paired and multiselect into a single ranking.

Report K5-minus-K2 paired score contrasts. Do not automatically select an optimal
K or declare each adjacent decline significant. Candidate semantics, difficulty,
input length and display order change alongside K. Correct-answer inclusion is
guaranteed; candidate-generator recall is outside the experiment.

## Multiple correct answers

Use the already recorded `p(true) >= 0.5` rule. Do not tune the threshold on the
viewed outcomes. Average the two display orders within each mother first.

- Primary: exact equality of the entire predicted set and reference set.
- Secondary: macro set F1, micro F1, precision, recall, all/empty selection rates,
  and mean selected/reference counts.
- Random baseline: explicitly choose each candidate independently with
  probability 0.5. Exact-set success probability is `2**(-K)`, not `1/K`.
  Compute `100 * (EM - 2**(-K)) / (1 - 2**(-K))` within each question, then average.
- Simple baseline: select every candidate; its set F1 is `2*M/(K+M)` where M is
  the reference-positive count. This reveals F1 inflation from over-selection.

Multiselect K=3/4/5 uses different mother questions, not a within-mother K
intervention. Its adjusted exact-match score is not combined with single-choice
scores. Per-K multiselect results are descriptive.

## Uncertainty and failure handling

Use a source-stratified context-cluster percentile bootstrap with fixed seed
20261026, default 2,000 repetitions (configurable). All K columns for a mother
share the same resample. MultiRC resamples paragraph groups, preserving questions
and display variants within a paragraph. Intervals are pointwise 95% and are
not corrected for multiple comparisons. This is an exploratory summary, not a
preregistered confirmatory significance test.

Evaluate a model only when its journal contains every expected request and has
DONE metadata and a matching journal hash. Check bank-file and payload hashes,
request identities, references, conditions, answer labels, saved/native
probabilities, argmax and threshold consistency before scoring. A malformed
model journal is UNEVALUABLE and does not block valid peers.

Within a complete journal, recorded case errors and incomplete input audits
count as incorrect rather than being dropped. An invalid multiselect request
receives EM/F1 zero. Valid-condition fractions and invalid-request counts are
reported. For partial candidate failures the request is treated as invalid.
Missing model runs have no scores; they are not assigned zero performance.

## Model coverage and provenance

`model_cohort.json` snapshots the 23-model established research cohort from
`research/evaluation_completion_v3/manifest.json`. Jev API is skipped unless
results actually exist; NanoJev remains a local model. Discover any additional
model directories and evaluate them using the same H3 output contract.

The evaluator is model-independent. The collection CLI now queues all 22 local
models through the adapters described in COMPATIBILITY.md. Code compatibility
does not imply successful T4 execution: BLOCKED_RESOURCE/BACKEND/DISK/ACCESS,
FAILED, NOT_RUN and SKIPPED_API stay distinct. Precision and quantization are
included in coverage tables; native and NF4 baseline results must retain their
different execution conditions.

Collector metadata retains `evaluation_performed=False`: it describes the raw
collection stage and remains immutable. Evaluation writes its own protocol,
status and summary with `evaluation_performed=True`, code/cohort hashes, NumPy
version, bootstrap settings and bank/profile fingerprints. Re-evaluating the same
inputs/settings is deterministic. Use a separate evaluation output directory
when retaining previous protocol versions.

## Outputs and reuse of the completed ZIP

`evaluation/` contains `REPORT.md`, `summary.json`, `protocol.json`, `status.json`,
`paired_scores.csv`, `paired_contrasts.csv`, `native_scores.csv`,
`multi_select_scores.csv` and `model_coverage.csv`. Scores use a 0-to-100 scale
(with negatives allowed); accuracy/F1/rates remain fractions in JSON/CSV.

```bash
# Current Colab run, before the download step copies the bank into results:
python research/hypothesis3_v1/evaluate.py \
  --results /content/hypothesis3-results/SESSION \
  --bank /content/hypothesis3-banks/BANK_REVISION \
  --output /content/hypothesis3-results/SESSION/evaluation

# Historical downloaded ZIP; no GPU, retraining or new inference:
python research/hypothesis3_v1/evaluate.py \
  --results hypothesis3-t4-results.zip --output h3-evaluation
```

The frozen bank, builder and receipts are unchanged. Semantic label review and
the collaborator's final H1/H2 exclusion lists are still pending. Public dataset
training exposure is not ruled out.
