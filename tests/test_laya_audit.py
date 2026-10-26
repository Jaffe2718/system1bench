"""Audit compatibility with Laya's callable tokenizer API, without a GPU."""
import sys
import types
import unittest
from unittest.mock import Mock, patch

from system1bench.laya_adapter import LayaAdapter


class LayaAuditTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Mock(mask_token="[MASK]")
        self.tokens = {
            "short": [7], "long": [7] * 100, "short ": [7],
            "choice question: pick": [10, 11],
            " a: alpha": [20, 21], " b: beta": [30, 31],
        }
        self.tokenizer.side_effect = lambda text, **kwargs: {"input_ids": self.tokens[text]}
        self.adapter = LayaAdapter.__new__(LayaAdapter)
        self.adapter.agent = types.SimpleNamespace(
            tok=self.tokenizer,
            _to_internal=lambda q: {"t": q["type"], "ins": q["instructions"]},
        )
        self.adapter._head_audit_cache = {}
        self.common = types.ModuleType("laya.common")
        # This surface deliberately has no encode_text, just like PyPI 0.3.20.
        self.common.serialize_state = lambda state: state
        self.common.render_options = lambda q: ["a: alpha", "b: beta"]
        self.common.build_sequence = Mock(return_value=(
            [1, 10, 11, 2, 3, 20, 21, 3, 30, 31, 2, 2], [4, 7]))
        self.questions = {"answer": {"type": "choice", "instructions": "pick"}}
        self.budget = {"max_len": 64, "head_max_len": 32}

    def audit(self, state):
        with patch.dict(sys.modules, {"laya.common": self.common}):
            return self.adapter.audit(state, self.questions, self.budget)["answer"]

    def test_callable_tokenizer_without_encode_text(self):
        result = self.audit("short")
        self.assertTrue(result["complete"])
        self.assertEqual(result["option_texts_shortened"], 0)
        self.assertEqual(result["state_tokens"], 1)

    def test_cached_head_still_checks_state_length_and_mask(self):
        self.audit("short")
        self.assertTrue(self.audit("long")["state_shortened"])
        masked = self.audit("short[MASK]")
        self.assertTrue(masked["special_mask_sanitized"])
        self.assertFalse(masked["complete"])
        self.common.build_sequence.assert_called_once()

    def test_option_shortening_remains_visible(self):
        self.tokens[" a: alpha"] = [20, 21, 22]
        result = self.audit("short")
        self.assertEqual(result["option_texts_shortened"], 1)
        self.assertFalse(result["complete"])


if __name__ == "__main__":
    unittest.main()
