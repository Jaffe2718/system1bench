# H3 full-cohort collection compatibility

The default collection CLI/notebook now queues all 22 established local models,
with Jev API excluded. NanoJev is included. Smaller models run first, in separate
processes. A single model's admission/load/case failure does not stop its peers.
This is code support, not a claim that every new backend has passed Colab T4.

No dependencies are installed in the user's Windows environment. Installation
and checkpoint/source downloads occur only when the notebook or CLI is run on
the host; local tests inject fake engines and use existing libraries. The
notebook now supports Colab and physical-server Jupyter environments.

## Backend selection

| Family | H3 implementation | Precision policy | Scheduler |
|---|---|---|---|
| Laya English/Multilingual | Existing pinned Laya encoder/collator/decoder | Existing validated fp16 AMP path | Adaptive batches |
| Kev 0.8B/4B/9B/27B | Pinned official checkpoint loader and pointer head | fp16 requested through LoadOptions; native head retained | Serial requests |
| NanoJev | Official DecisionPredictor from pinned checkpoint source/scripts | fp32 storage/inference, feature-detected native override deregistration | Serial requests |
| Intern Decision 0.8B/2B/4B | Pinned official schema, HF backend and calibration preset | fp16 on T4; explicit backend dtype | Serial requests |
| StartLux 4B/9B/27B | Official code from pinned model snapshot, native decide API | fp16 if constructor exposes dtype; otherwise explicit backend block on T4 | Serial requests |
| Llama 3.1 8B, Llama 3.2 1B/3B | Full-input constrained answer-code likelihood | fp16 or explicitly recorded NF4 | Serial requests |
| Qwen 3 1.7B/8B, Qwen 3.5 0.8B/4B/9B, Qwen 3.8 27B | Full-input constrained answer-code likelihood, thinking disabled | fp16 or explicitly recorded NF4 | Serial requests |
| Jev API | No API invocation | Skipped | Not scheduled by all |

Native decision models are not replaced by ordinary LM next-token scoring.
Other architectures are not passed into Laya's private batching methods.
The scheduler request and effective scheduler are recorded separately.

## Laya monitoring revision

Default interval is 0.25 seconds (target 4 Hz). Growth requires 64 consecutive
active samples with RAM and effective GPU-memory ratios below 70%, approximately
16 seconds. Each increase is 1.5x, capped at 128 question rows and 32,768 padded
tokens. Both a completed batch and an in-flight long forward count as activity;
otherwise faster monitoring would incorrectly reset on long batches. Queue
backlog is still required. A missed monitoring deadline resets the low counter;
stale samples are not replayed in a burst. OS scheduling cannot guarantee a hard
real-time sampling rate, and actual sample timestamps remain in the journal.

Pressure at 85% or less than 2 GiB free GPU memory halves capacity. OOM triggers
split/retry. Growth cooldown is also 64 samples. These measurements concern GPU
memory, not GPU SM compute utilization. The monitor requires no NVML install.

## Precision, admission and honest failure states

Before weight download, use conservative parameter-size estimates and free GPU
memory to choose a policy or emit `BLOCKED_RESOURCE`. This is a heuristic, not
an OOM guarantee. Native heads have no validated NF4 path here; a large native
model requires enough GPU memory. Ordinary Llama/Qwen baselines may use NF4
double quantization with fp16 compute when their fp16 estimate does not fit.
That changed inference condition is explicit in model metadata and coverage
tables and is not presented as historical full-precision equivalence.

27B models ordinarily fail this T4 admission estimate even for the baseline NF4
path. Larger GPUs can use the same queue with a fitting policy. The code does
not download huge weights just to discover an obvious memory mismatch, and
does not invent a score for blocked runs. Disk insufficiency is `BLOCKED_DISK`;
user model caches/weights are never deleted automatically to make space.

`BLOCKED_BACKEND` describes unsupported precision/native APIs. Gated or
unavailable Hugging Face checkpoints are `BLOCKED_ACCESS`; use authorized HF
credentials if already available, with no GH_TOKEN requirement. Arbitrary
initialization failures remain `FAILED`, with logs for diagnosis.

Llama-3.1-8B's historical registry entry has only local-file identity, not a Hub
commit pin. Without an explicit revision override, the loader resolves the
official repo commit before downloading and records both the historical label
and resolved commit with `historical_match_verified=False`. It does not claim
the newly fetched snapshot matches the previous local files.

## Overrides and source setup

Optional notebook `MODEL_OVERRIDES`, or CLI `--model-config overrides.json`:

```json
{
  "kev_4b": {"path": "/content/checkpoints/kev-4b", "base_path": "/content/checkpoints/qwen-base"},
  "qwen3_8b": {"precision": "nf4"}
}
```

Allowed per-model fields: path, base_path, revision, base_revision and precision
(auto/float16/bfloat16/float32/nf4). The model repository cannot be silently
replaced. Laya accepts only a local checkpoint path, keeping its validated
revision/precision policy. Public pins include the Kev base revisions, and
checkpoint metadata must match those pins before a local base path is used.

The notebook checks `requirements-notebook.txt` and installs only missing or
incompatible entries in one pip call under the host's existing torch version
constraint. No torch/CUDA reinstall is requested. Official source is
loaded lazily after the selected model passes admission: Kev git commit
0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91 and Intern git commit
3572c8a68b5df5dafe02d0e093989ba8ec0183bc. NanoJev/StartLux code comes from the
same pinned HF snapshot as the weights. Reused historical adapter files are
unchanged; their hashes/source copies, native source hashes and checkpoint file
hashes are retained with results.

The notebook remains configure -> run -> evaluate -> download. Evaluation keeps
completed, failed, blocked, not-run and skipped-API rows distinct. Recorded case
errors in a complete journal still count as wrong, per EVALUATION.md.


## Cache-first server notebook

[SERVER.md](SERVER.md) supersedes the original Colab-only environment setup.
The notebook detects compatible installed distributions first, preserves the
host torch version and installs only missing/incompatible requirements. Complete
local model directories and pinned HF snapshots are preferred. Incomplete shard
sets are not called cache hits; online mode fills missing files and strict
offline mode reports BLOCKED_CACHE. Native git-source setup also respects
offline mode. The whole-model disk-space estimate no longer rejects a warm
cache before lookup. Colab temporary-cache lifetime is not made persistent by
this code, and Google Drive is never mounted.

The NanoJev wrapper detects whether torch._native and its deregistration hook
exist before selecting disable_native_triton. With the module absent the flag
is False; with the hook available it is True. No private module is fabricated,
no cached upstream source is edited and no torch upgrade is required.
