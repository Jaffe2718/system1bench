# Hypothesis 3 collection protocol v1

Status: data bank and prediction collection only. **Evaluation is not frozen.**
No score, ranking, optimal K, significance claim or mechanism conclusion is
computed by the collection notebook. A later protocol version will define the
evaluation step before confirmatory interpretation.

## Question bank

The experimental unit is the mother question; MultiRC questions share paragraph
clusters. The initial bank is English only. Native panel sources retain their
original K and reference semantics. Equal counts across native K do not make their
task difficulty equal, so cross-K conclusions cannot be inferred from that panel
alone. Native and paired panels do not share mother/context groups.

Each paired mother retains one source-provided reference and four preselected
incorrect labels/answers. No candidate prose is invented or expanded. A seeded
order of the four distractors and its reverse define two nested addition paths.
K=2/3/4/5 always retain the gold meaning and preserve every retained option's text.
Five neutral IDs are assigned once per mother, independent of correctness, and
remain stable across K/path/order. Display position is counterbalanced within each
source/K across the collection; two display orders are stored. This is a candidate
subset intervention, not a claim to isolate K independently of input length or the
semantic difficulty of added distractors. Correct-answer inclusion is guaranteed,
so the bank does not test candidate-generator recall or a full deployment optimum.

Two source questions with the same normalized paragraph/context are kept in one
data split. Development uses approximately 20% of source groups; the remainder is
confirmation. Both display orders and every K/path variant of a mother share its
split. Sixty development paired mothers (20 per source) are listed as an optional
pilot selection. The notebook defaults to full collection and performs no pilot
or smoke stage automatically. The same frozen inputs must be supplied to all models.

Multiselect uses complete original MultiRC questions with K=3/4/5 and 2 <= number
of positives < K. No gold count is provided to the model. It is a separate task,
not a fifth level of single-choice K. The complete option set accompanies each
candidate-specific noul question. Probability-to-set rules and performance
metrics remain open for later evaluation; native outputs are retained.

## Integrity and failure policy

Only `state` and `questions` reach the adapter. Source IDs, gold, splits, condition
aliases and candidate-pool receipts remain outside the model payload. Failure
records retain case identity, phase, exception and traceback. The runner records
all intended requests, including invalid answers, and does not discard difficult
cases based on model behavior. Failed initializations do not block later models.

All actual candidate texts, source IDs, byte hashes, source revisions and protocol
code are bound in the bank or download. Raw source bytes are SHA256-verified before
parsing. Code and JSON receipt text use canonical LF hashing for Windows/Linux
portability. The frozen original experiments are neither modified nor extended.

## Pending before a confirmatory paper

- Import the collaborator's final hypothesis-1/2 source-ID/context exclusion lists.
- Review ambiguity and distractor quality independently of tested model outcomes.
- Decide per-K comparison, random baselines, probability and multiselect metrics.
- Decide treatment of missing/invalid model outputs and incomplete input audits.
- Specify paired/group-level uncertainty, multiple comparisons and the rule, if
  any, for selecting a preferred K from development data.
- Keep confirmation outcomes out of protocol/model/threshold selection.
