"""Small, independent regression tests; no checkpoint/reference data required."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from image_encoder import ImageEncoder, MultiHeadAttention, TransformerBlock


def attention_parameters(width=4):
    rng = np.random.default_rng(2026)
    return {
        'a.in_proj_weight': rng.normal(0, .2, (3 * width, width)).astype('float32'),
        'a.in_proj_bias': rng.normal(0, .1, 3 * width).astype('float32'),
        'a.out_proj.weight': rng.normal(0, .2, (width, width)).astype('float32'),
        'a.out_proj.bias': rng.normal(0, .1, width).astype('float32'),
    }


class EncoderTests(unittest.TestCase):
    def test_patch_order_against_explicit_crops(self):
        # Distinct pixels expose both wrong grid order and wrong C/H/W flattening.
        x = np.arange(2 * 3 * 224 * 224, dtype=np.float32).reshape(2, 3, 224, 224)
        encoder = ImageEncoder.__new__(ImageEncoder)
        actual = encoder.patchify(x)
        expected = np.stack([
            np.stack([image[:, r:r+32, c:c+32].ravel()
                      for r in range(0, 224, 32) for c in range(0, 224, 32)])
            for image in x])
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (2, 49, 3072))

    def test_split_heads_against_feature_slices(self):
        attention = MultiHeadAttention(attention_parameters(8), 'a', 2)
        x = np.arange(2 * 3 * 8, dtype=np.float32).reshape(2, 3, 8)
        actual = attention.split_heads(x)
        for b in range(2):
            for h in range(2):
                np.testing.assert_array_equal(actual[b, h], x[b, :, h*4:(h+1)*4])

    def test_attention_against_loop_oracle(self):
        attention = MultiHeadAttention(attention_parameters(), 'a', 2)
        x = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 10
        qkv = x.astype('float64') @ attention.qkv_weight.T + attention.qkv_bias
        expected = np.empty_like(x, dtype='float64')
        # Float64 scalar dot products form an independent, tiny reference.
        for b in range(2):
            merged = np.zeros((3, 4), dtype='float64')
            for head in range(2):
                start = head * 2
                q = qkv[b, :, start:start+2]
                k = qkv[b, :, 4+start:6+start]
                v = qkv[b, :, 8+start:10+start]
                for query in range(3):
                    scores = np.array([np.dot(q[query], k[key]) / np.sqrt(2) for key in range(3)])
                    coeff = np.exp(scores - scores.max()); coeff /= coeff.sum()
                    merged[query, start:start+2] = sum(coeff[key] * v[key] for key in range(3))
            expected[b] = merged @ attention.out_weight.T + attention.out_bias
        np.testing.assert_allclose(attention(x), expected, rtol=2e-5, atol=1e-7)
        self.assertEqual(attention(x).dtype, np.float32)

    def test_causal_mask_blocks_future_information(self):
        attention = MultiHeadAttention(attention_parameters(), 'a', 2)
        x = np.arange(12, dtype=np.float32).reshape(1, 3, 4) / 10
        changed = x.copy(); changed[:, -1] += [1, -3, 4, -2]
        np.testing.assert_array_equal(attention(x, causal=True)[:, :2],
                                      attention(changed, causal=True)[:, :2])
        self.assertFalse(np.allclose(attention(x)[:, :2], attention(changed)[:, :2]))

    def test_zero_updates_preserve_both_residual_paths(self):
        w = {k.replace('a.', 'b.attn.'): np.zeros_like(v) for k, v in attention_parameters().items()}
        for norm in ['ln_1', 'ln_2']:
            w['b.' + norm + '.weight'] = np.ones(4, dtype='float32')
            w['b.' + norm + '.bias'] = np.zeros(4, dtype='float32')
        w.update({'b.mlp.c_fc.weight': np.zeros((7, 4), dtype='float32'),
                  'b.mlp.c_fc.bias': np.zeros(7, dtype='float32'),
                  'b.mlp.c_proj.weight': np.zeros((4, 7), dtype='float32'),
                  'b.mlp.c_proj.bias': np.zeros(4, dtype='float32')})
        x = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 10
        np.testing.assert_array_equal(TransformerBlock(w, 'b', 2)(x), x)

    def test_four_dimensional_key_transpose_regression(self):
        q = np.ones((2, 12, 50, 64), dtype='float32')
        k = q.copy()
        self.assertEqual(k.T.shape, (64, 50, 12, 2))
        with self.assertRaises(ValueError):
            _ = q @ k.T
        self.assertEqual((q @ k.swapaxes(-1, -2)).shape, (2, 12, 50, 50))


if __name__ == '__main__':
    unittest.main(verbosity=2)
