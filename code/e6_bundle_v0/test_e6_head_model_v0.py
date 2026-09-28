"""Pure/synthetic E6 component checks, including one fixed toy optimization.

No real EO inputs, E5 code/results, model checkpoints, GPU, or experiment runner.
The toy is a software sanity check, not evidence for EO task performance.
"""
import inspect
import math
import unittest

import torch

from e6_head_model_v0 import (AttentionBlock, SpatialPairHead, encode_metadata,
                              full_transform, make_head, spatial_positions, state_hash)


class E6HeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def sample(self, n=2):
        generator = torch.Generator(device='cpu').manual_seed(19)
        pairs = torch.randn(n, 2, 64, 768, generator=generator, dtype=torch.float32)
        phenomena = ['flood']*n
        dates = [('2020-01-01', '2020-02-02')]*n
        return pairs, phenomena, dates

    def test_full_transform_shape_types_delta_immutability_and_reverse(self):
        pairs, _, _ = self.sample()
        original = pairs.clone()
        tokens, types = full_transform(pairs)
        self.assertEqual(tuple(tokens.shape), (2, 192, 768))
        self.assertEqual(types.tolist(), [0]*64+[1]*64+[3]*64)
        self.assertEqual(types.dtype, torch.int64)
        torch.testing.assert_close(tokens[:, :64], pairs[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(tokens[:, 64:128], pairs[:, 1], rtol=0, atol=0)
        torch.testing.assert_close(tokens[:, 128:], pairs[:, 1]-pairs[:, 0], rtol=0, atol=0)
        reverse, _ = full_transform(pairs.flip(1))
        torch.testing.assert_close(reverse[:, 128:], -tokens[:, 128:], rtol=0, atol=0)
        single, _ = full_transform(pairs[0])
        torch.testing.assert_close(single, tokens[0], rtol=0, atol=0)
        tokens.zero_()
        torch.testing.assert_close(pairs, original, rtol=0, atol=0)
        constant, _ = full_transform(torch.ones(2, 64, 768))
        self.assertTrue(torch.equal(constant[128:], torch.zeros(64, 768)))

    def test_transform_rejects_bad_shape_dtype_nonfinite_and_difference_overflow(self):
        pair = self.sample(1)[0][0]
        bad = [pair.double(), pair[:, :63], torch.empty(0, 2, 64, 768), pair[None, None]]
        for value in (float('nan'), float('inf')):
            corrupt = pair.clone(); corrupt[0, 0, 0] = value; bad.append(corrupt)
        overflow = torch.empty_like(pair)
        overflow[0].fill_(-torch.finfo(torch.float32).max)
        overflow[1].fill_(torch.finfo(torch.float32).max)
        bad.append(overflow)
        for value in bad:
            with self.subTest(shape=tuple(value.shape), dtype=value.dtype):
                with self.assertRaises(ValueError):
                    full_transform(value)

    def test_metadata_exact_features_and_strict_dates(self):
        result = encode_metadata(['flood', 'landslide'],
                                 [('2024-02-29', '2024-03-01'), ('1999-12-31', '2000-01-01')])
        expected = torch.tensor([[1, 0, .24, 1/11, 28/30, .24, 2/11, 0],
                                 [0, 1, -.01, 1, 1, 0, 0, 0]], dtype=torch.float32)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        cases = [(['flood'], [('2021-02-29', '2021-03-01')]),
                 (['flood'], [('2020-01-01T00:00:00', '2020-02-01')]),
                 (['flood'], [('2020-01-01', '2020-01-01')]),
                 (['flood'], [('2020-02-01', '2020-01-01')]),
                 (['flood'], [('2020-1-01', '2020-02-01')]),
                 (['flood'], [('0000-01-01', '2020-02-01')]),
                 (['unknown'], [('2020-01-01', '2020-02-01')]),
                 (['flood'], []), ([], []), ('flood', [('2020-01-01', '2020-02-01')]),
                 ([{'id': 'secret', 'answer': 'yes'}], [('2020-01-01', '2020-02-01')])]
        for phenomena, dates in cases:
            with self.subTest(phenomena=phenomena, dates=dates):
                with self.assertRaises(ValueError):
                    encode_metadata(phenomena, dates)

    def test_positions_preserve_grid_and_block_identity(self):
        pos = spatial_positions()
        self.assertEqual(tuple(pos.shape), (192, 128))
        torch.testing.assert_close(pos[:64], pos[64:128], rtol=0, atol=0)
        torch.testing.assert_close(pos[:64], pos[128:], rtol=0, atol=0)
        torch.testing.assert_close(pos[0], torch.tensor([0., 1.]*64), rtol=0, atol=0)
        self.assertAlmostEqual(float(pos[1, 64]), math.sin(1), places=6)
        self.assertAlmostEqual(float(pos[8, 0]), math.sin(1), places=6)
        self.assertEqual(float(pos[1, 0]), 0)
        self.assertEqual(float(pos[8, 64]), 0)
        self.assertAlmostEqual(float(pos[63, 62]), math.sin(7/10000**(31/32)), places=7)

    def test_architecture_seed_hash_independent_blocks_and_rng_scope(self):
        before = torch.random.get_rng_state().clone()
        first, again, other = make_head(7), make_head(7), make_head(8)
        torch.testing.assert_close(torch.random.get_rng_state(), before, rtol=0, atol=0)
        self.assertEqual(state_hash(first), state_hash(again))
        self.assertNotEqual(state_hash(first), state_hash(other))
        self.assertEqual(sum(p.numel() for p in first.parameters()), 367361)
        self.assertTrue(all(p.device.type == 'cpu' and p.dtype == torch.float32 for p in first.parameters()))
        self.assertTrue(torch.equal(first.cls, torch.zeros(128)))
        self.assertEqual(len(first.blocks), 2)
        self.assertEqual(first.blocks[0].ff_in.out_features, 256)
        self.assertNotEqual(first.blocks[0].qkv.weight.data_ptr(), first.blocks[1].qkv.weight.data_ptr())
        self.assertFalse(torch.equal(first.blocks[0].qkv.weight, first.blocks[1].qkv.weight))
        for layer in first.modules():
            if isinstance(layer, torch.nn.LayerNorm):
                self.assertEqual(layer.eps, 1e-5)
            if isinstance(layer, torch.nn.GELU):
                self.assertEqual(layer.approximate, 'none')
            self.assertNotIsInstance(layer, torch.nn.Dropout)
        with torch.no_grad():
            other.classifier.weight[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            state_hash(other)

    def test_label_id_free_inference_and_no_mutation_batch_consistency(self):
        head = make_head(11)
        pairs, phenomena, dates = self.sample()
        original = pairs.clone()
        before = state_hash(head)
        self.assertEqual(set(inspect.signature(head.predict).parameters), {'pairs', 'phenomena', 'dates'})
        self.assertEqual(set(inspect.signature(head.forward).parameters), {'pairs', 'metadata'})
        with self.assertRaises(TypeError):
            head.predict(pairs, phenomena, dates, labels=torch.ones(2))
        result = head.predict(pairs, phenomena, dates)
        self.assertTrue(head.training)
        self.assertFalse(result['logits'].requires_grad)
        self.assertEqual(before, state_hash(head))
        torch.testing.assert_close(pairs, original, rtol=0, atol=0)
        single = head.predict(pairs[:1], phenomena[:1], dates[:1])
        torch.testing.assert_close(result['logits'][:1], single['logits'], rtol=1e-5, atol=5e-6)
        torch.testing.assert_close(result['probabilities'], torch.sigmoid(result['logits']))
        self.assertTrue(torch.equal(result['predictions_yes'], result['logits'] >= 0))

    def test_input_metadata_and_model_replacements_are_not_silently_ignored(self):
        head = make_head(11)
        pairs, phenomena, dates = self.sample()
        base = head.predict(pairs, phenomena, dates)['logits']
        alternatives = [head.predict(pairs.flip(1), phenomena, dates)['logits'],
                        head.predict(torch.zeros_like(pairs), phenomena, dates)['logits'],
                        head.predict(pairs, phenomena, [('2016-02-29', '2024-04-01')]*2)['logits'],
                        head.predict(pairs, ['landslide']*2, dates)['logits'],
                        make_head(12).predict(pairs, phenomena, dates)['logits']]
        for i, alternative in enumerate(alternatives):
            with self.subTest(replacement=i):
                self.assertGreater(float((base-alternative).abs().max()), 1e-6)
        # This only checks information flow. It does not assign physical labels
        # to swapped dates/frames, nor require any specific semantic prediction.
        duplicate = pairs[:1].repeat(2, 1, 1, 1)
        same = head.predict(duplicate, phenomena, dates)['logits']
        torch.testing.assert_close(same[0], same[1], rtol=0, atol=0)

    def test_gradients_reach_both_frames_metadata_and_every_module(self):
        pairs, phenomena, dates = self.sample()
        pairs.requires_grad_()
        metadata = encode_metadata(phenomena, dates).requires_grad_()
        head = make_head(7)
        logits = head(pairs, metadata)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1.]))
        loss.backward()
        self.assertTrue(torch.isfinite(pairs.grad).all())
        self.assertGreater(float(pairs.grad[:, 0].abs().sum()), 0)
        self.assertGreater(float(pairs.grad[:, 1].abs().sum()), 0)
        self.assertGreater(float(metadata.grad[:, 2:].abs().sum()), 0)
        for name, parameter in head.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        bad = metadata.detach().clone(); bad[0, 2] = float('nan')
        with self.assertRaises(ValueError):
            head(pairs.detach(), bad)
        bad = metadata.detach().clone(); bad[0, :2] = 0
        with self.assertRaises(ValueError):
            head(pairs.detach(), bad)
        with self.assertRaises(ValueError):
            head(pairs.detach(), metadata.detach().double())

    def test_one_fixed_toy_fit_has_label_direction_and_frame_swap_control(self):
        # Fixed before running: seed7, eight samples, 80 full-batch AdamW updates,
        # lr1e-3/wd.01, no sweep/early stop/tuning. Same dates for both classes.
        signs = torch.tensor([-1., 1.]*4)
        pattern = torch.linspace(-1, 1, 768).reshape(1, 1, 768).repeat(8, 64, 1)
        later = .5*signs[:, None, None]*pattern
        pairs = torch.stack((-later, later), dim=1)
        labels = (signs > 0).float()
        metadata = encode_metadata(['flood']*8, [('2020-01-01', '2020-02-02')]*8)
        head = make_head(7)
        initial_hash = state_hash(head)
        optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=.01,
                                     betas=(.9, .999), eps=1e-8)
        initial = float(torch.nn.functional.binary_cross_entropy_with_logits(head(pairs, metadata), labels).detach())
        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(head(pairs, metadata), labels)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()))
            optimizer.step()
        with torch.no_grad():
            logits = head(pairs, metadata)
            final = float(torch.nn.functional.binary_cross_entropy_with_logits(logits, labels))
            swapped = head(pairs.flip(1), metadata)
        self.assertLess(final, initial*.25)
        self.assertTrue(torch.equal(logits >= 0, labels.bool()))
        self.assertTrue(torch.equal(swapped >= 0, ~labels.bool()))
        self.assertNotEqual(state_hash(head), initial_hash)
        restored = make_head(99)
        restored.load_state_dict(head.state_dict(), strict=True)
        self.assertEqual(state_hash(restored), state_hash(head))
        torch.testing.assert_close(restored(pairs, metadata), logits, rtol=0, atol=0)
        print(f'Fixed synthetic sanity only: BCE {initial:.6f} -> {final:.6f}; 8/8 correct; swapped 8/8 reversed')


if __name__ == '__main__':
    unittest.main(verbosity=2)
