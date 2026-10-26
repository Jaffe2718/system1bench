# Colab / physical-server notebook

Use `notebooks/hypothesis3_t4.ipynb` in the collaborator's existing GPU Jupyter
environment. It now uses the same four steps on Colab and a physical server:
configure, collect, evaluate, package/download. No changes to the historical
`notebooks/system1bench_t4.ipynb` are needed.

## Existing environment and persistent cache

Select an existing Python 3.12+ kernel with CUDA PyTorch already installed.
The notebook refuses to install or replace torch/CUDA. Dependency metadata is
checked first: compatible installed versions are reused. Only missing or
incompatible requirements trigger one pip call, constrained to the existing
torch version. It does not require downgrading an installed Transformers 5.17
to 5.16.1. The Laya private batching API still requires laya==0.3.20.

First-cell settings:

```python
REPO_PATH = "/srv/projects/system1bench"  # or None to find it from cwd/parents
WORK_ROOT_PATH = "/srv/experiments/hypothesis3"
HF_CACHE_DIR = "/srv/cache/huggingface/hub"
AUTO_INSTALL_MISSING = False  # inspect/report missing requirements; admin prepares them
ALLOW_MODEL_DOWNLOADS = True
UPDATE_REPO = False
```

`HF_CACHE_DIR=None` preserves the host's HF_HUB_CACHE/HF_HOME/default cache
instead of forcing a new empty directory. HF cache normally lives at
`~/.cache/huggingface/hub`. Prefer the directory that already contains the
server's `models--.../snapshots/` repositories. A new unrelated cache location
cannot reuse files kept elsewhere automatically.

Without explicit paths, an existing repository is found from the current
directory and its parents. Outside Colab, runtime files default to the sibling
`hypothesis3-runtime/` directory. Existing source is not auto-pulled unless
UPDATE_REPO=True; sync the repository before opening the new notebook.

## Checkpoint lookup order

1. An explicit MODEL_OVERRIDES local checkpoint/base directory, if provided.
2. The requested pinned revision in the HF cache, checked without network access.
3. If missing/incomplete and downloads are allowed, normal Hub incremental
   download into that same cache. Existing blobs are reused; force_download is
   never set.

Checking a snapshot directory's existence alone is insufficient: a previous
download may have only fetched config files. The lookup checks required
tokenizer/config/native-source files and weight files, including every shard
listed by a safetensors index. This is a completeness check, not proof against
arbitrary local corruption. Runtime file hashes are retained as before.

Examples for directories outside the HF cache:

```python
MODEL_OVERRIDES = {
    "nanojev": {"path": "/srv/models/NanoJev"},
    "kev_4b": {
        "path": "/srv/models/kev-4b",
        "base_path": "/srv/models/Qwen3.5-4B-Base",
    },
}
```

Use the exact pinned checkpoint/base revisions when comparing with historical
runs. An arbitrary different checkpoint directory is not automatically the
same model; hashes and overrides remain in the result package.

Set ALLOW_MODEL_DOWNLOADS=False for strict offline model collection. Missing
files become BLOCKED_CACHE with a reason; it does not silently go online.
Kev/Intern also need their pinned source checkouts in `.aris/vendor/` already
prepared. Source setup respects offline mode too. Llama-3.1's historical entry
has no public immutable revision, so offline use requires a local path or an
explicit revision override. Data downloads are controlled separately by
ALLOW_DATA_DOWNLOADS; the existing frozen bank/raw-data cache can be reused.

The collector CLI supports the same settings:

```bash
python research/hypothesis3_v1/run.py \
  --bank /srv/experiments/hypothesis3/BANK \
  --output /srv/experiments/hypothesis3/RESULTS \
  --models all --cache-dir /srv/cache/huggingface/hub --offline
```

No user model files are deleted to free disk. The previous conservative
whole-model free-disk check is removed: it could incorrectly reject an already
cached checkpoint. Missing downloads rely on Hub's file-level checks; ENOSPC
is reported as BLOCKED_DISK when encountered during checkpoint preparation.

Colab's default cache is on its temporary VM disk and may disappear after a VM
reset/shutdown. Code cannot reuse files that no longer exist. This notebook does
not mount or write Google Drive. A physical server's persistent disk avoids
that lifetime problem; first-time downloads of distinct uncached weights are
still necessary. Caching does not eliminate GPU loading/inference time.

## NanoJev torch._native compatibility

The pinned NanoJev predictor imports torch._native only when
disable_native_triton=True. Previously the H3 wrapper forced this flag on and
failed on Colab versions without that private module. It now inspects the API:

- Module absent: pass False; there is no such native override registry to undo.
- Deregistration API available: pass True, preserving the explicit ATen path.
- Module exists with an unsupported API: report BLOCKED_BACKEND.

Unexpected dependency-import errors are not swallowed. No fake Torch modules,
upstream cache edits or PyTorch upgrades are used. Effective flag and reason
are recorded in metadata. The original pinned predictor and model tensors stay
unchanged; fp32 inference remains selected for NanoJev.

## Workload and output

H3 has 1,600 mother questions: native 800, paired 600, multiselect 200.
There are 11,569 unique requests and 12,767 decisions per model. The paired panel
alone contributes 9,569 requests after deduplication (9,600 design conditions),
because each mother tests K=2/3/4/5, two nested paths and two display orders.
All 22 local models therefore budget 254,518 requests before blocked runs.
The notebook now displays the mother/request/decision budget before collection.

For comparison, the original 36-suite `protocol_manifest.json` totals 22,934
requests and 26,450 decisions per model, including repeated/control conditions.
The later shared model-expansion panel has 2,601 requests / 4,905 decisions.
These request totals are not counts of independent source questions.

On Colab the final cell invokes browser download. On a server it writes a ZIP
to the runtime directory and displays a normal Jupyter file link, also printing
the full path. Raw predictions, evaluation tables, bank, code and logs remain
in the output directory. No Colab modules are imported on the server path.

See [HF cache documentation](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)
and [cache environment variables](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).
