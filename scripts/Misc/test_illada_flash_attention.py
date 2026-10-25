"""Run with Python; CPU tests mock only the FlashAttention kernels, GPU tests use them."""

import copy
import importlib
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F


MODEL_DIR = Path(os.environ.get(
    "ILLADA_MODEL_CODE",
    Path(__file__).resolve().parents[2] / "verl/models/iLLaDA-8B-Instruct",
))
package = types.ModuleType("_illada_attention_test")
package.__path__ = [str(MODEL_DIR)]
sys.modules[package.__name__] = package
modeling = importlib.import_module(f"{package.__name__}.modeling_illada")


def dense_flash_reference(q, k, v, dropout_p=0.0, causal=False):
    assert not causal, "iLLaDA must use bidirectional attention"
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        dropout_p=dropout_p, is_causal=False,
    ).transpose(1, 2)


def packed_flash_reference(q, k, v, cu_seqlens_q, cu_seqlens_k,
                           max_seqlen_q, max_seqlen_k, dropout_p=0.0, causal=False):
    assert torch.equal(cu_seqlens_q, cu_seqlens_k)
    assert max_seqlen_q == max_seqlen_k
    pieces = []
    for start, end in zip(cu_seqlens_q[:-1].tolist(), cu_seqlens_q[1:].tolist()):
        pieces.append(dense_flash_reference(
            q[start:end][None], k[start:end][None], v[start:end][None],
            dropout_p=dropout_p, causal=causal,
        )[0])
    return torch.cat(pieces)


def make_model(backend, dtype=torch.float32):
    config = modeling.ILLaDAConfig(
        vocab_size=64, hidden_size=64, intermediate_size=96,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        pad_token_id=0, attn_implementation=backend, dtype=dtype,
    )
    return modeling.ILLaDAForCausalLM(config).to(dtype=dtype)


def compare_outputs_and_gradients(test, reference, candidate, ids,
                                  reference_kwargs=None, candidate_kwargs=None, atol=1e-6, rtol=1e-4):
    reference.zero_grad(set_to_none=True)
    candidate.zero_grad(set_to_none=True)
    expected = reference(ids, **(reference_kwargs or {})).logits
    actual = candidate(ids, **(candidate_kwargs or reference_kwargs or {})).logits
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    expected.float().square().mean().backward()
    actual.float().square().mean().backward()
    for (name, p), (_, q) in zip(reference.named_parameters(), candidate.named_parameters()):
        test.assertIsNotNone(p.grad, name)
        test.assertIsNotNone(q.grad, name)
        torch.testing.assert_close(q.grad, p.grad, atol=atol, rtol=rtol, msg=name)


def packed_case(device):
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], device=device)
    positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3]], device=device)
    segments = torch.tensor([0, 0, 0, 1, 1, 1, 1], device=device)
    mask = (segments[:, None] == segments[None, :])[None, None]
    reference_kwargs = dict(position_ids=positions, attention_bias=mask)
    packed_kwargs = dict(
        position_ids=positions,
        cu_seqlens=torch.tensor([0, 3, 7], dtype=torch.int32, device=device),
        max_seqlen=4,
    )
    return ids, reference_kwargs, packed_kwargs


class CPUFlashAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)
        # Bypass only the device/package availability check for CPU kernel mocks.
        check = patch.object(modeling.ILLaDAPreTrainedModel, "_flash_attn_2_can_dispatch", return_value=True)
        check.start()
        self.addCleanup(check.stop)
        imports = patch("transformers.modeling_utils.lazy_import_flash_attention")
        imports.start()
        self.addCleanup(imports.stop)
        self.reference = make_model("sdpa")
        self.candidate = make_model("flash_attention_2")
        self.candidate.load_state_dict(self.reference.state_dict())

    def test_unmasked_and_all_valid_mask_use_flash_with_matching_gradients(self):
        self.assertTrue(modeling.ILLaDAPreTrainedModel._supports_flash_attn_2)
        self.assertIsInstance(self.candidate.model.layers[0].self_attn, modeling.ILLaDAFlashAttention2)
        ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        for mask in (None, torch.ones_like(ids)):
            with self.subTest(mask=mask), patch.object(
                modeling, "flash_attn_func", side_effect=dense_flash_reference
            ) as kernel:
                compare_outputs_and_gradients(
                    self, self.reference, self.candidate, ids, dict(attention_mask=mask)
                )
                kernel.assert_called_once()
                self.assertFalse(kernel.call_args.kwargs["causal"])

    def test_padding_and_arbitrary_bias_preserve_sdpa_semantics(self):
        ids = torch.tensor([[0, 0, 3, 4], [5, 6, 0, 0]])
        masks = [
            dict(attention_mask=torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])),
            dict(attention_mask=torch.zeros_like(ids)),
            dict(attention_bias=torch.ones(2, 1, 4, 4, dtype=torch.bool).tril()),
            dict(attention_bias=torch.randn(2, 1, 4, 4)),
        ]
        for kwargs in masks:
            with self.subTest(kwargs=kwargs), patch.object(modeling, "flash_attn_func") as kernel:
                compare_outputs_and_gradients(self, self.reference, self.candidate, ids, kwargs)
                kernel.assert_not_called()

    def test_packed_sequences_match_block_diagonal_attention(self):
        ids, reference_kwargs, packed_kwargs = packed_case("cpu")
        with patch.object(modeling, "flash_attn_varlen_func", side_effect=packed_flash_reference) as kernel:
            compare_outputs_and_gradients(
                self, self.reference, self.candidate, ids, reference_kwargs, packed_kwargs
            )
            kernel.assert_called_once()
            self.assertFalse(kernel.call_args.kwargs["causal"])

    def test_attention_weights_fall_back_to_eager(self):
        reference = make_model("eager")
        reference.load_state_dict(self.reference.state_dict())
        ids = torch.tensor([[1, 2, 3]])
        with patch.object(modeling, "flash_attn_func") as kernel:
            actual = self.candidate(ids, output_attentions=True)
            expected = reference(ids, output_attentions=True)
            torch.testing.assert_close(actual.logits, expected.logits)
            torch.testing.assert_close(actual.attentions[0], expected.attentions[0])
            kernel.assert_not_called()

    def test_dropout_is_disabled_during_evaluation(self):
        attention = self.candidate.model.layers[0].self_attn
        attention.attention_dropout = 0.2
        ids = torch.tensor([[1, 2, 3]])
        with patch.object(modeling, "flash_attn_func", side_effect=dense_flash_reference) as kernel:
            self.candidate.train()(ids)
            self.assertEqual(kernel.call_args.kwargs["dropout_p"], 0.2)
            self.candidate.eval()(ids)
            self.assertEqual(kernel.call_args.kwargs["dropout_p"], 0.0)

    def test_checkpoint_reload_selects_flash_attention(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.reference.save_pretrained(tmp)
            model = modeling.ILLaDAForCausalLM.from_pretrained(
                tmp, attn_implementation="flash_attention_2"
            )
            self.assertIsInstance(model.model.layers[0].self_attn, modeling.ILLaDAFlashAttention2)
            with patch.object(modeling, "flash_attn_func", side_effect=dense_flash_reference) as kernel:
                model(torch.tensor([[1, 2, 3]]))
                kernel.assert_called_once()


@unittest.skipUnless(torch.cuda.is_available() and modeling.flash_attn_func is not None,
                     "requires a ticketed CUDA GPU and flash-attn")
class GPUFlashAttentionTests(unittest.TestCase):
    def test_real_kernels_forward_and_backward(self):
        torch.manual_seed(17)
        reference = make_model("sdpa", torch.bfloat16).cuda()
        candidate = make_model("flash_attention_2", torch.bfloat16).cuda()
        candidate.load_state_dict(copy.deepcopy(reference.state_dict()))
        ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], device="cuda")
        compare_outputs_and_gradients(self, reference, candidate, ids, atol=3e-3, rtol=3e-2)
        ids, reference_kwargs, packed_kwargs = packed_case("cuda")
        compare_outputs_and_gradients(
            self, reference, candidate, ids, reference_kwargs, packed_kwargs,
            atol=3e-3, rtol=3e-2,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
