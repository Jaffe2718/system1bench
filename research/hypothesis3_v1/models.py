"""H3 model dispatch and Colab-only checkpoint setup; no imports load weights."""
import inspect
import json
from pathlib import Path
import re
import shutil
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
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "remote", "add", "origin", url], check=True)
    head = subprocess.run(["git", "-C", str(target), "rev-parse", "--verify", "HEAD"],
                          capture_output=True, text=True)
    if head.returncode:
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
    from huggingface_hub import HfApi, snapshot_download
    pin = dict(COHORT["models"][name])
    pin.update({k:v for k,v in override.items() if k != "precision"})
    if "path" not in pin:
        revision = pin["revision"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            # Historical Llama pin is only a local-file-hash receipt. Resolve and
            # explicitly mark the new revision; never claim historical equality.
            pin["historical_revision"] = revision
            pin["revision"] = HfApi().model_info(pin["repo"]).sha
            pin["historical_match_verified"] = False
        pin["path"] = snapshot_download(pin["repo"], revision=pin["revision"],
            allow_patterns=["*.json", "*.safetensors", "*.pt", "*.model", "*.txt", "*.jinja", "*.py", "LICENSE*", "NOTICE*"])
    if not Path(pin["path"]).is_dir():
        raise FileNotFoundError(pin["path"])
    if "base_repo" in pin and "base_path" not in pin:
        pin["base_path"] = snapshot_download(pin["base_repo"], revision=pin["base_revision"],
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
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
            engine.model = DecisionPredictor(path, max_length=8192, precision="fp32", disable_native_triton=True)
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
        return LayaAdapter(name, override.get("path"))
    import torch
    override=(overrides or {}).get(name,{})
    free,_=torch.cuda.mem_get_info()
    precision=memory_policy(name,free/2**30,override.get("precision","auto"),torch.cuda.is_bf16_supported())
    # Check disk before a multi-gigabyte transfer. Metadata/code downloads only
    # start after GPU admission; user weights are never deleted to make room.
    if "path" not in override and shutil.disk_usage(ROOT).free < PARAMETERS_B.get(name,1)*2e9+2e9:
        raise ModelBlocked("BLOCKED_DISK", "Insufficient free disk for pinned checkpoint; retain earlier results")
    try:
        pin=prepare_checkpoint(name,override)
    except Exception as exc:
        if type(exc).__name__ in ("GatedRepoError","RepositoryNotFoundError"):
            raise ModelBlocked("BLOCKED_ACCESS", "Hugging Face checkpoint unavailable or requires authorized HF credentials") from exc
        raise
    return NativeBridge(name,pin,precision) if name in NATIVE else CodedLM(name,pin,precision)
