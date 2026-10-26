"""Local Laya adapter and exact input-construction audit."""
import platform
from pathlib import Path

from .common import digest, sha

MODEL_REPO = "convaiinnovations/laya"
MODEL_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"


class LayaAdapter:
    def __init__(self, model, checkpoint=None):
        import torch
        import transformers
        from huggingface_hub import snapshot_download
        from laya import Agent

        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
        torch.manual_seed(0)
        sub = "multilingual" if model == "multilingual" else ""
        if checkpoint is None:
            patterns = ["multilingual/**"] if sub else ["encoder/**", "tokenizer/**", "model.safetensors", "rl_agent_config.json"]
            checkpoint = snapshot_download(MODEL_REPO, revision=MODEL_REVISION, allow_patterns=patterns)
        model_root = Path(checkpoint) / sub
        files = sorted(p for p in model_root.rglob("*") if p.is_file() and "multilingual" not in p.relative_to(model_root).parts)
        # Computed BEFORE Agent construction / inference; paths are portable relative names.
        hashes = {str(p.relative_to(model_root)): sha(p) for p in files}
        import laya
        source_root = Path(laya.__file__).parent
        sources = {str(p.relative_to(source_root)): sha(p) for p in sorted(source_root.rglob("*.py"))}
        self.agent = Agent(str(checkpoint), subfolder=sub or None, device="cuda")
        self._head_audit_cache = {}
        self.metadata = dict(adapter="laya", model=model, repo=MODEL_REPO, requested_revision=MODEL_REVISION,
                             model_files_sha256=hashes, laya_source_sha256=sources, python=platform.python_version(),
                             torch=torch.__version__, transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0),
                             dtype=str(self.agent.dtype), amp_enabled=self.agent.amp_enabled,
                             temperatures=self.agent.temperature, configuration=self.agent.cfg)

    def synchronize(self):
        import torch
        torch.cuda.synchronize()

    def predict(self, states, questions, language, budget, batch_size):
        return self.agent.predict_batch(states, questions, batch_size=batch_size, lang=language, **budget)

    def audit(self, state, questions, budget):
        from laya.common import build_sequence, render_options, serialize_state

        tok = self.agent.tok
        state_text = serialize_state(state)
        state_ids = tok(state_text.replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"]
        result = {}
        for qid, q in questions.items():
            cache_key = (digest(q), budget["max_len"], budget["head_max_len"])
            if cache_key in self._head_audit_cache:
                base = self._head_audit_cache[cache_key].copy()
                state_shortened = len(state_ids) > budget["max_len"] - base["head_tokens"]
                sanitized = tok.mask_token in state_text or tok.mask_token in str(q)
                base.update(state_tokens=len(state_ids), state_shortened=state_shortened,
                            special_mask_sanitized=sanitized,
                            complete=base["complete"] and not state_shortened and not sanitized)
                result[qid] = base
                continue
            internal = self.agent._to_internal(q)
            empty, markers = build_sequence(tok, "", internal, state_ids=[], **budget)
            original = [tok(" " + s.replace(tok.mask_token, " "), add_special_tokens=False)["input_ids"] for s in render_options(internal)]
            actual = [empty[p + 1:(markers[i + 1] if i + 1 < len(markers) else len(empty) - 2)] for i, p in enumerate(markers)]
            ins = tok("%s question: %s" % (internal["t"], str(internal["ins"]).replace(tok.mask_token, " ")), add_special_tokens=False)["input_ids"]
            actual_ins = empty[1:markers[0] - 1] if markers else []
            state_shortened = len(state_ids) > budget["max_len"] - len(empty)
            shortened = sum(a != b for a, b in zip(actual, original)) + abs(len(actual) - len(original))
            marker_sanitized = tok.mask_token in state_text or tok.mask_token in str(q)
            result[qid] = dict(state_tokens=len(state_ids), head_tokens=len(empty), options=len(original),
                               option_texts_shortened=shortened, unique_encoded_options=len(set(map(tuple, actual))),
                               instruction_shortened=actual_ins != ins, state_shortened=state_shortened,
                               special_mask_sanitized=marker_sanitized,
                               complete=(not shortened and actual_ins == ins and not state_shortened and not marker_sanitized and len(markers) == len(original)))
            base = result[qid].copy()
            base["complete"] = not shortened and actual_ins == ins and len(markers) == len(original)
            self._head_audit_cache[cache_key] = base
        return result
