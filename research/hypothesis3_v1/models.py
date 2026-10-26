"""H3 model dispatch and cache-first checkpoint setup; imports never load weights."""
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COHORT = json.loads((HERE / "model_cohort.json").read_text(encoding="utf-8"))
LAYA = {"english", "multilingual"}
NATIVE = {n for n in COHORT["models"] if n.startswith(("kev_", "intern_", "startlux_"))} | {"nanojev"}
VENDORS = {
    "kev": ("https://github.com/jaredpalmer/kev.git", "0fe8fc97c2bcc247fa3efb6e5c32af4e99770e91", "kev_source"),
    "intern": ("https://github.com/InternLM/Intern-Decision.git", "3572c8a68b5df5dafe02d0e093989ba8ec0183bc", "intern_source"),
}
PARAMETERS_B = {
    "kev_08b": .9, "kev_4b": 5., "kev_9b": 10., "kev_27b": 28., "nanojev": .6,
    "intern_08b": .9, "intern_2b": 2.2, "intern_4b": 5.,
    "startlux_4b": 5., "startlux_9b": 10., "startlux_27b": 28.,
    "qwen35_08b": .9, "qwen35_4b": 5., "qwen35_9b": 10., "qwen38_27b": 28.,
    "qwen3_17b": 1.7, "qwen3_8b": 8.2, "llama32_1b_public": 1.3,
    "llama32_3b_public": 3.3, "llama31_8b_instruct": 8.1,
}
STORAGE = dict(cache_dir=None, offline=False)


def configure_storage(cache_dir=None, offline=None):
    """Call before importing HF libraries; also pass cache_dir explicitly."""
    use_offline = (os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")
                   if offline is None else bool(offline))
    STORAGE.update(cache_dir=str(Path(cache_dir).expanduser().resolve()) if cache_dir else None,
                   offline=use_offline)
    if cache_dir:
        os.environ["HF_HUB_CACHE"] = STORAGE["cache_dir"]
    if STORAGE["offline"]:
        os.environ["HF_HUB_OFFLINE"] = "1"
    elif offline is not None:
        os.environ.pop("HF_HUB_OFFLINE", None)


def snapshot_complete(path, name):
    """A snapshot directory alone may contain only a previous config download."""
    root = Path(path)
    def all_files(names):
        return all((root / n).is_file() for n in names)
    def tokenizer(prefix=""):
        return all_files([prefix+"tokenizer_config.json"]) and any(
            (root / (prefix+n)).is_file() for n in ("tokenizer.json", "tokenizer.model", "vocab.json"))
    def weights():
        index = root / "model.safetensors.index.json"
        if index.is_file():
            try:
                names = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
                return bool(names) and all(not Path(n).is_absolute() and ".." not in Path(n).parts
                                           and (root/n).is_file() for n in names)
            except (ValueError, KeyError, TypeError):
                return False
        return (root / "model.safetensors").is_file()
    if name in LAYA:
        prefix = "multilingual/" if name == "multilingual" else ""
        return all_files([prefix+n for n in ("rl_agent_config.json", "model.safetensors", "encoder/config.json")]) and tokenizer(prefix+"tokenizer/")
    if name == "nanojev":
        return all_files(["config.json", "best.safetensors", "backbone_config/config.json",
                          "source/scripts/predict_toy_decisions.py", "source/scripts/train_toy_decisions.py"]) and tokenizer("tokenizer/")
    if name.startswith("kev_"):
        return (root/"head.pt").is_file() and (
            all_files(["adapter_config.json", "adapter_model.safetensors"]) or
            (all_files(["config.json"]) and weights()))
    extra = []
    if name.startswith("startlux_"):
        extra = ["decision_config.json", "startlux_decision/model.py", "startlux_decision/jevfmt.py"]
    return all_files(["config.json", *extra]) and tokenizer() and weights()


def cached_snapshot(repo, revision, patterns, name):
    from huggingface_hub import snapshot_download
    options = dict(repo_id=repo, revision=revision, allow_patterns=patterns,
                   cache_dir=STORAGE["cache_dir"])
    try:
        cached = snapshot_download(**options, local_files_only=True)
    except Exception as exc:
        if type(exc).__name__ not in ("LocalEntryNotFoundError", "CacheNotFound", "FileNotFoundError"):
            raise
        cached = None
    if cached and snapshot_complete(cached, name):
        print("MODEL_CACHE_HIT", name, repo, revision, cached, flush=True)
        return cached
    if STORAGE["offline"]:
        raise ModelBlocked("BLOCKED_CACHE", f"Missing/incomplete cached snapshot: {name} {repo}@{revision}; prefetch online or supply a complete local path")
    print("MODEL_CACHE_MISS_OR_PARTIAL", name, repo, revision, flush=True)
    # Hub reuses existing blobs and downloads missing files. Do not reject a
    # warm cache based on the size of an entire model or force redownload.
    try:
        downloaded = snapshot_download(**options, local_files_only=False)
    except OSError as exc:
        if exc.errno == 28:
            raise ModelBlocked("BLOCKED_DISK", "Model cache disk is full; existing model files and results are retained") from exc
        raise
    if not snapshot_complete(downloaded, name):
        raise ModelBlocked("BLOCKED_CACHE", f"Downloaded snapshot lacks required files for {name}: {downloaded}")
    return downloaded


def nano_triton_policy():
    """The pinned predictor imports torch._native only when this flag is true."""
    try:
        native = importlib.import_module("torch._native")
    except ModuleNotFoundError as exc:
        if exc.name != "torch._native":
            raise
        return False, "torch._native absent; no private native override registry to deregister"
    utility = getattr(native, "triton_utils", None)
    if utility is None:
        try:
            utility = importlib.import_module("torch._native.triton_utils")
        except ModuleNotFoundError as exc:
            if exc.name != "torch._native.triton_utils":
                raise
            raise ModelBlocked("BLOCKED_BACKEND", "torch._native exists without a supported triton_utils API") from exc
    if not callable(getattr(utility, "deregister_op_overrides", None)):
        raise ModelBlocked("BLOCKED_BACKEND", "Native Triton override registry has no deregister_op_overrides API")
    return True, "private native override deregistration API available"


def make_nano_predictor(predictor_class, path):
    disable, reason = nano_triton_policy()
    model = predictor_class(path, max_length=8192, precision="fp32", disable_native_triton=disable)
    return model, disable, reason


class ModelBlocked(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def model_names():
    # Smaller models first; each runs in a separate process, never co-resident.
    return sorted((n for n in COHORT["models"] if n != "jev-1.13.0"),
                  key=lambda n: (PARAMETERS_B.get(n, .5), n))


def select_models(names):
    names = model_names() if names == ["all"] else names
    if not names or any(n not in COHORT["models"] for n in names):
        raise ValueError("Unknown model; use --models all or established cohort IDs")
    return list(dict.fromkeys(names))


def load_overrides(path):
    if path is None:
        return {}
    overrides = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {"path", "base_path", "revision", "base_revision", "precision"}
    for name, values in overrides.items():
        if name not in COHORT["models"] or not isinstance(values, dict) or set(values)-allowed:
            raise ValueError("Invalid model override: " + name)
        if values.get("precision", "auto") not in ("auto", "float16", "bfloat16", "float32", "nf4"):
            raise ValueError("Invalid precision override")
    return overrides


def memory_policy(name, free_gib, precision="auto", bf16_supported=False):
    if name == "jev-1.13.0":
        raise ModelBlocked("SKIPPED_API", "Jev API excluded; no API request made")
    native = name in NATIVE
    params = PARAMETERS_B.get(name, .5)
    if precision == "auto":
        precision = "nf4" if not native and params >= 6 and free_gib < params*2/1.073741824+2.5 else "float16"
    if precision == "bfloat16" and not bf16_supported:
        raise ModelBlocked("BLOCKED_BACKEND", "This GPU does not natively support BF16; choose float16")
    if precision == "nf4" and native:
        raise ModelBlocked("BLOCKED_BACKEND", "Native decision heads have no validated NF4 path; not replaced with an LM baseline")
    # NanoJev's official path keeps fp32 storage; count that memory conservatively.
    if name == "nanojev":
        precision = "float32"
    bytes_per_param = .5 if precision == "nf4" else 4 if precision == "float32" else 2
    required = params*bytes_per_param/1.073741824 + 2.5
    if free_gib < required:
        raise ModelBlocked("BLOCKED_RESOURCE", f"{name}: estimated {required:.1f} GiB free needed for {precision}; {free_gib:.1f} available. No weights downloaded; use a larger GPU.")
    return precision


def ensure_vendor(family):
    url, revision, folder = VENDORS[family]
    target = ROOT / ".aris/vendor" / folder
    if not (target / ".git").exists():
        if STORAGE["offline"]:
            raise ModelBlocked("BLOCKED_CACHE", "Pinned native source missing in offline mode: " + str(target))
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "remote", "add", "origin", url], check=True)
    head = subprocess.run(["git", "-C", str(target), "rev-parse", "--verify", "HEAD"],
                          capture_output=True, text=True)
    if head.returncode:
        if STORAGE["offline"]:
            raise ModelBlocked("BLOCKED_CACHE", "Pinned native source checkout incomplete in offline mode: " + str(target))
        # A previous interrupted fetch leaves an initialized repo without HEAD;
        # retry it rather than poisoning every later model of the same family.
        subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", revision], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", revision], check=True)
    actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise ValueError("Native vendor revision differs from pin")
    if subprocess.check_output(["git", "-C", str(target), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise ValueError("Native vendor tree has local modifications")
    sys.path.insert(0, str(target))
    return target


def prepare_checkpoint(name, override):
    from huggingface_hub import HfApi
    pin = dict(COHORT["models"][name])
    pin.update({k:v for k,v in override.items() if k != "precision"})
    if "path" not in pin:
        revision = pin["revision"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            if STORAGE["offline"]:
                raise ModelBlocked("BLOCKED_CACHE", "Offline mode needs an explicit immutable revision or local path for " + name)
            pin["historical_revision"] = revision
            pin["revision"] = HfApi().model_info(pin["repo"]).sha
            pin["historical_match_verified"] = False
        if name in LAYA:
            patterns = (["multilingual/**"] if name == "multilingual" else
                        ["encoder/**", "tokenizer/**", "model.safetensors", "rl_agent_config.json"])
        else:
            patterns = ["*.json", "*.safetensors", "*.pt", "*.model", "*.txt", "*.jinja", "*.py", "LICENSE*", "NOTICE*"]
        pin["path"] = cached_snapshot(pin["repo"], pin["revision"], patterns, name)
    else:
        pin["path"] = str(Path(pin["path"]).expanduser().resolve())
        if not snapshot_complete(pin["path"], name):
            raise ModelBlocked("BLOCKED_CACHE", "Supplied local checkpoint is incomplete for " + name + ": " + pin["path"])
        print("MODEL_LOCAL_PATH", name, pin["path"], flush=True)
    if "base_repo" in pin:
        if "base_path" not in pin:
            pin["base_path"] = cached_snapshot(pin["base_repo"], pin["base_revision"],
                ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"], "base")
        else:
            pin["base_path"] = str(Path(pin["base_path"]).expanduser().resolve())
            if not snapshot_complete(pin["base_path"], "base"):
                raise ModelBlocked("BLOCKED_CACHE", "Supplied local base checkpoint is incomplete: " + pin["base_path"])
    return pin


def wrap_distribution(question, dist):
    from system1bench.decision_models import validate_distribution
    pred, probs = validate_distribution(question, dist)
    answer = dict(probabilities=probs)
    if question["type"] == "choice":
        answer["choice"] = pred
    elif question["type"] == "noul":
        answer["noul"] = probs["true"]
    else:
        answer["score"] = sum(int(k)*v for k,v in probs.items())
    return answer


class NativeBridge:
    """Keep each family's official inference; adapt only the runner envelope."""
    def __init__(self, name, pin, precision):
        import torch
        requested_precision = precision
        dtype = getattr(torch, precision)
        (ROOT / ".aris/family_v3").mkdir(parents=True, exist_ok=True)
        path = Path(pin["path"])
        if name.startswith("kev"):
            vendor = ensure_vendor("kev")
        elif name.startswith("intern"):
            vendor = ensure_vendor("intern")
        elif name == "nanojev":
            vendor = path / "source/scripts"
            if not (vendor / "predict_toy_decisions.py").is_file():
                raise ModelBlocked("BLOCKED_BACKEND", "Pinned NanoJev snapshot lacks source/scripts/predict_toy_decisions.py")
            sys.path.insert(0, str(vendor))
        else:
            vendor = path
            sys.path.insert(0, str(path))
        # These imports reuse frozen encoders, renderers and decoders, not their
        # BF16-only load paths. Source files remain unchanged.
        from research.model_expansion_v3.adapter import get_adapter
        self.engine = engine = get_adapter(name, pin, load=False)
        engine.limit = 8192
        if name.startswith("kev"):
            from kev.checkpoint import LoadOptions
            if engine.checkpoint.meta.base_revision != pin.get("base_revision"):
                raise ValueError("Kev base revision differs from public pin")
            if engine.checkpoint.meta.weights_dtype == "bf16":
                if not torch.cuda.is_bf16_supported():
                    raise ModelBlocked("BLOCKED_BACKEND", "This Kev checkpoint forces BF16 and this GPU lacks native BF16 support")
                dtype, precision = torch.bfloat16, "bfloat16"
            if "base_path" in pin:
                engine.checkpoint.meta.base = str(Path(pin["base_path"]).resolve())
                engine.checkpoint.meta.base_revision = None
            engine.tok, engine.model = engine.checkpoint.load("cuda", LoadOptions(
                dtype=dtype, attn="sdpa", merge=True, backend="torch", cuda_graphs=False, fused=False))
        elif name.startswith("intern"):
            from src.inference.engine import DecisionEngine
            calibration = ROOT / (".aris/intern_4b_calibration.json" if name == "intern_4b" else f".aris/family_v3/{name}_calibration.json")
            engine.model = DecisionEngine(str(path), max_length=8192, calibration_path=str(calibration),
                backend="hf", device="cuda", dtype=precision, attn_implementation="sdpa")
        elif name == "nanojev":
            from predict_toy_decisions import DecisionPredictor
            engine.model, disable, reason = make_nano_predictor(DecisionPredictor, path)
            engine.metadata.update(disable_native_triton=disable, native_triton_compatibility=reason)
        else:
            from startlux_decision.model import StartLuxDecision
            options = dict(device="cuda", max_length=8192, graphs=False, images=False)
            signature = inspect.signature(StartLuxDecision)
            if "dtype" in signature.parameters:
                default = signature.parameters["dtype"].default
                options["dtype"] = precision if isinstance(default, str) else dtype
            else:
                if not torch.cuda.is_bf16_supported():
                    raise ModelBlocked("BLOCKED_BACKEND", "Pinned StartLux constructor exposes no dtype option on this non-BF16 GPU")
                precision = "bfloat16"
            import os
            os.environ["STARTLUX_ALLOW_SLOW"] = "1"
            engine.model = StartLuxDecision(str(path), **options)
        from system1bench.decision_models import fingerprints
        from system1bench.common import sha
        self.metadata = dict(engine.metadata, model=name, h3_native_bridge=True, dtype=precision, requested_dtype=requested_precision,
            quantized=False, scheduler_effective="serial", context_limit=8192,
            checkpoint={k:v for k,v in pin.items() if k not in ("path","base_path")},
            model_files_sha256=fingerprints(path, name == "nanojev"),
            vendor_code_sha256={str(p.relative_to(vendor)):sha(p) for p in sorted(vendor.rglob("*.py")) if "__pycache__" not in p.parts})
        if "base_path" in pin:
            self.metadata["base_files_sha256"] = fingerprints(Path(pin["base_path"]))

    def synchronize(self):
        self.engine.synchronize()

    def audit(self, state, questions, budget):
        self.engine.limit = min(8192, budget["max_len"])
        return self.engine.audit(state, questions)

    def predict(self, states, questions, language, budget, batch_size):
        outputs = []
        for state in states:
            probabilities = self.engine.predict(state, questions)
            outputs.append(dict(answers={q:wrap_distribution(question, probabilities[q]) for q,question in questions.items()}))
            # Avoid retaining one encoded prompt for every completed request.
            cache = getattr(self.engine, "cache", None)
            if cache is not None:
                cache.clear()
        return outputs


class CodedLM:
    """Pinned Llama/Qwen direct-code baseline, with explicit T4 precision."""
    def __init__(self, name, pin, precision):
        import torch
        import transformers
        from transformers import AutoTokenizer, BitsAndBytesConfig
        from system1bench.common import digest
        from system1bench.decision_models import fingerprints
        torch.set_num_threads(4)
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(pin["path"], local_files_only=True, trust_remote_code=False)
        if not self.tok.chat_template:
            raise ValueError("Native instruct chat template required")
        self.code_ids = [self.tok.encode(c, add_special_tokens=False) for c in "ABCDE"]
        if any(len(v) != 1 for v in self.code_ids) or len({v[0] for v in self.code_ids}) != 5:
            raise ValueError("H3 answer codes must be distinct single tokens")
        self.code_ids = [v[0] for v in self.code_ids]
        cls = (transformers.AutoModelForImageTextToText if name.startswith(("qwen35_","qwen38_"))
               else transformers.AutoModelForCausalLM)
        kwargs = dict(local_files_only=True, trust_remote_code=False, attn_implementation="sdpa",
                      dtype=torch.float16 if precision == "nf4" else getattr(torch, precision))
        if precision == "nf4":
            kwargs.update(device_map={"":"cuda:0"}, quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True))
        self.model = cls.from_pretrained(pin["path"], **kwargs)
        if precision != "nf4":
            self.model = self.model.to("cuda")
        self.model.eval()
        self.cache = None
        import platform
        self.metadata = dict(model=name,adapter="h3_constrained_next_token",torch=torch.__version__,
            python=platform.python_version(),gpu=torch.cuda.get_device_name(0),
            transformers=transformers.__version__,dtype=precision,quantized=precision=="nf4",seed=0,
            quantization="NF4 double-quant, fp16 compute" if precision=="nf4" else None,
            checkpoint={k:v for k,v in pin.items() if k not in ("path","base_path")},
            model_files_sha256=fingerprints(Path(pin["path"])),chat_template_sha256=digest(self.tok.chat_template),
            scheduler_effective="serial",enable_thinking=False,text_only=True,truncation=False,context_limit=8192,
            probability_semantics="conditional likelihood of candidate answer codes; not calibrated correctness")

    def encoded(self, state, questions, budget):
        from system1bench.common import digest
        from system1bench.llm_adapter import messages
        key = digest([state,questions,budget])
        if self.cache is not None and self.cache[0] == key:
            return self.cache[1]
        encoded = {}
        for qid,q in questions.items():
            text = self.tok.apply_chat_template(messages(state,q),tokenize=False,
                add_generation_prompt=True,enable_thinking=False)
            ids = self.tok.encode(text,add_special_tokens=False)
            if len(ids)>min(8192,budget["max_len"]):
                raise ValueError("ContextOverflow: full input retained; no truncation")
            encoded[qid]=ids
        self.cache=(key,encoded)
        return encoded

    def audit(self, state, questions, budget):
        from system1bench.common import labels, digest
        return {qid:dict(complete=True,prompt_tokens=len(ids),token_sha256=digest(ids),options=len(labels(questions[qid])))
                for qid,ids in self.encoded(state,questions,budget).items()}

    def synchronize(self):
        self.torch.cuda.synchronize()

    def predict(self, states, questions, language, budget, batch_size):
        from system1bench.common import labels
        from system1bench.llm_adapter import answer_from_logits
        t, outputs = self.torch, []
        with t.inference_mode():
            for state in states:
                answers = {}
                for qid,ids in self.encoded(state,questions,budget).items():
                    inputs=t.tensor([ids],device="cuda")
                    logits=self.model(input_ids=inputs,attention_mask=t.ones_like(inputs),
                        use_cache=False,logits_to_keep=1).logits[0,-1].float()
                    allowed=self.code_ids[:len(labels(questions[qid]))]
                    answers[qid]=answer_from_logits(questions[qid],logits[allowed].cpu().tolist())
                outputs.append(dict(answers=answers))
                self.cache=None
        return outputs


def create_adapter(name, overrides=None):
    if name == "jev-1.13.0":
        raise ModelBlocked("SKIPPED_API", "Jev API excluded; no API call made")
    if name in LAYA:
        from system1bench.laya_adapter import LayaAdapter
        override = (overrides or {}).get(name,{})
        if set(override)-{"path"}:
            raise ValueError("Laya keeps its validated model/precision policy; only local path override supported")
        pin = prepare_checkpoint(name, override)
        return LayaAdapter(name, pin["path"])
    import torch
    override=(overrides or {}).get(name,{})
    free,_=torch.cuda.mem_get_info()
    precision=memory_policy(name,free/2**30,override.get("precision","auto"),torch.cuda.is_bf16_supported())
    try:
        pin=prepare_checkpoint(name,override)
    except Exception as exc:
        if isinstance(exc, OSError) and getattr(exc, "errno", None) == 28:
            raise ModelBlocked("BLOCKED_DISK", "Model cache disk is full; earlier results are retained") from exc
        if type(exc).__name__ in ("GatedRepoError","RepositoryNotFoundError"):
            raise ModelBlocked("BLOCKED_ACCESS", "Hugging Face checkpoint unavailable or requires authorized HF credentials") from exc
        raise
    return NativeBridge(name,pin,precision) if name in NATIVE else CodedLM(name,pin,precision)
