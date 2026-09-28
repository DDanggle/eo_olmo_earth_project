"""Synthetic CPU contract/gradient tests; not real EO, Qwen, or semantic tests."""
from __future__ import annotations
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from matching_head import RoleObjectMatchingHead
from episode_adapter import make_episode_model_class, verify_dependencies, PINNED_DEPENDENCIES

BASE_ROOT = Path(os.environ.get('OE11_BASE_ROOT',
    '/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code/oe10_text_mask_v2'))
Episode = make_episode_model_class(BASE_ROOT)
from contracts import PROMPT


class ToyEncoder(nn.Module):
    embedding_size = 8
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(12, 8)
    def forward(self, sample, patch_size, fast_pass):
        x = sample.sentinel2_l2a
        t = x.shape[3]
        z = self.project(x.reshape(1, 32, 4, 32, 4, t, 12).mean((2, 4))).unsqueeze(4)
        return {'tokens_and_masks': SimpleNamespace(sentinel2_l2a=z)}


class ToyTokenizer:
    def __call__(self, text, **kwargs):
        ids = torch.tensor([[ord(c) % 255 + 1 for c in text]])
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}


class ToyProcessor:
    tokenizer = ToyTokenizer()
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert not tokenize and not add_generation_prompt and len(messages) == 1
        return messages[0]['content'][0]['text'] + '\nUSER_END'


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(256, 16)
    def forward(self, input_ids, attention_mask, use_cache, output_hidden_states, return_dict):
        assert not use_cache and not output_hidden_states and return_dict
        x = self.embed(input_ids)
        weight = torch.arange(1, x.shape[1] + 1, dtype=x.dtype)[None, :, None]
        return SimpleNamespace(last_hidden_state=(x * weight).cumsum(1) / weight.cumsum(1))


class ToyQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyBackbone()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=16))
    def get_input_embeddings(self):
        return self.model.embed


def model(arm='B2', seed=53, **kwargs):
    torch.manual_seed(seed)
    return Episode(ToyEncoder(), arm, 'cpu', None, True, qwen_model=ToyQwen(),
                   processor=ToyProcessor(), sample_factory=SimpleNamespace,
                   online_mask_value=1,
                   reader_identity={'kind': 'synthetic_cpu_reader_not_Qwen', 'seed': seed},
                   eo_identity={'kind': 'synthetic_cpu_encoder_not_EO', 'seed': seed,
                                'normalization': 'synthetic_standard_normal'},
                   **kwargs)


def packet(seed, dates, support=False):
    rng = np.random.default_rng(seed)
    result = dict(s2=rng.normal(size=(128, 128, dates, 12)).astype(np.float32),
                  raw_s2=np.ones((dates, 10, 128, 128), dtype=np.int16),
                  timestamps=np.asarray([[1 + i, 0, 2019] for i in range(dates)], dtype=np.int64),
                  dates_yyyymmdd=np.arange(20190101, 20190101 + dates),
                  observation_valid=np.ones((dates, 128, 128), dtype=bool),
                  band_observed=np.array([True] * 10 + [False] * 2))
    if support:
        result['mask'] = np.zeros((128, 128), dtype=bool)
        result['mask'][12:92, 16:80] = True
    return result


def inputs(k=1):
    x = dict(prompt=PROMPT, query=packet(1, 2), support_pairs=[
        dict(positive=packet(2 + 2 * i, 8, True), counterexample=packet(3 + 2 * i, 8, True))
        for i in range(k)])
    y = np.zeros((128, 128), dtype=bool)
    y[:72, :48] = True
    return x, dict(target_mask=y, label_valid=np.ones_like(y))


def context(positive='grapevine', counter='leguminous fodder'):
    return dict(instruction=PROMPT, positive=positive, counterexample=counter)


class HeadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, k=4, b=2):
        torch.manual_seed(47)
        head = RoleObjectMatchingHead(8, 16, attention_dimension=6,
                                      hidden_dimension=12, output_size=(8, 12))
        values = [torch.randn(b, 2, 3, 8), torch.randn(b, k, 8),
                  torch.randn(b, k, 8), torch.randn(b, 2, 16)]
        return head, values

    def test_all_k_and_mask_loss_gradient_paths(self):
        for k in (1, 2, 4, 8):
            with self.subTest(k=k):
                head, values = self.fixture(k)
                values = [x.requires_grad_() for x in values]
                out = head(*values, return_details=True)
                self.assertEqual(tuple(out['logits'].shape), (2, 8, 12))
                self.assertEqual(tuple(out['attention'].shape), (2, 2, 3, 2, k))
                self.assertTrue(torch.allclose(out['attention'].sum(-1), torch.ones(2, 2, 3, 2)))
                target = torch.zeros_like(out['logits']); target[:, :4] = 1
                F.binary_cross_entropy_with_logits(out['logits'], target).backward()
                for value in values:
                    self.assertTrue(bool(torch.isfinite(value.grad).all()))
                    self.assertGreater(float(value.grad.norm()), 0)
                self.assertGreater(float(head.text_project.weight.grad.norm()), 0)
                self.assertGreater(float(head.background_scorer[-1].weight.grad.norm()), 0)
                if k > 1:
                    self.assertGreater(float(head.text_key.weight.grad.norm()), 0)

    def test_independent_object_permutations_and_role_swap(self):
        head, (q, p, n, t) = self.fixture()
        original = head(q, p, n, t, return_details=True)
        pp, pn = [2, 0, 3, 1], [1, 3, 0, 2]
        changed = head(q, p[:, pp], n[:, pn], t, return_details=True)
        self.assertTrue(torch.allclose(original['logits'], changed['logits'], atol=1e-7))
        self.assertTrue(torch.allclose(original['attention'][..., 0, pp], changed['attention'][..., 0, :]))
        swapped = head(q, n, p, t[:, [1, 0]], return_details=True)
        self.assertTrue(torch.allclose(original['role_scores'], swapped['role_scores'].flip(-1)))
        self.assertTrue(torch.equal(original['background_scores'], swapped['background_scores']))
        self.assertTrue(bool((original['logits'].sigmoid() + swapped['logits'].sigmoid() <= 1 + 1e-7).all()))

    def test_text_and_query_change_object_selection(self):
        head, (q, p, n, t) = self.fixture()
        a = head(q, p, n, t, return_details=True)
        b = head(q, p, n, torch.roll(t, 3, -1), return_details=True)
        c = head(torch.roll(q, 2, -1), p, n, t, return_details=True)
        for changed in (b, c):
            self.assertGreater(float((a['attention'] - changed['attention']).abs().max().detach()), 1e-6)
            self.assertGreater(float((a['logits'] - changed['logits']).abs().max().detach()), 1e-6)

    def test_k1_text_route_and_identical_roles(self):
        head, (q, p, n, t) = self.fixture(1)
        a = head(q, p, n, t, return_details=True)
        b = head(q, p, n, torch.roll(t, 3, -1))
        self.assertTrue(torch.equal(a['attention'], torch.ones_like(a['attention'])))
        self.assertGreater(float((a['logits'] - b).abs().max().detach()), 1e-6)
        identical = head(q, p, p, t[:, :1].expand(-1, 2, -1))
        self.assertTrue(bool((identical < 0).all()))

    def test_neither_role_present_can_be_negative_in_both_directions(self):
        head, (q, p, n, t) = self.fixture()
        # Controlled output scores show the representational ability to reject
        # both concepts; this is not an observed EO absence detector.
        with torch.no_grad():
            for parameter in head.scorer.parameters(): parameter.zero_()
            for parameter in head.background_scorer.parameters(): parameter.zero_()
            head.background_scorer[-1].bias.fill_(2)
        a = head(q, p, n, t)
        b = head(q, n, p, t.flip(1))
        self.assertTrue(bool(((a < 0) & (b < 0)).all()))
        self.assertTrue(bool((a.sigmoid() + b.sigmoid() < 1).all()))
        F.binary_cross_entropy_with_logits(a, torch.zeros_like(a)).backward()
        self.assertGreater(float(head.background_scorer[-1].bias.grad.abs()), 0)

    def test_masked_objects_do_not_affect_output(self):
        head, (q, p, n, t) = self.fixture()
        valid = torch.tensor([[[1, 1, 0, 0], [1, 0, 1, 0]]] * 2, dtype=torch.bool)
        a = head(q, p, n, t, support_valid=valid, return_details=True)
        p2, n2 = p.clone(), n.clone()
        p2[~valid[:, 0]] = 100 * torch.randn_like(p2[~valid[:, 0]])
        n2[~valid[:, 1]] = 100 * torch.randn_like(n2[~valid[:, 1]])
        b = head(q, p2, n2, t, support_valid=valid)
        self.assertTrue(torch.equal(a['logits'], b))
        self.assertTrue(torch.equal(a['attention'].masked_select(~valid[:, None, None]),
                                    torch.zeros_like(a['attention'].masked_select(~valid[:, None, None]))))

    def test_batch_independence_and_duplicate_support(self):
        head, values = self.fixture(1)
        together = head(*values)
        split = torch.cat([head(*(v[i:i + 1] for v in values)) for i in range(2)])
        self.assertTrue(torch.allclose(together, split, atol=1e-7))
        q, p, n, t = values
        repeated = head(q, p.repeat(1, 8, 1), n.repeat(1, 8, 1), t)
        self.assertTrue(torch.allclose(together, repeated, atol=1e-7))

    def test_invalid_contracts_fail_closed(self):
        head, (q, p, n, t) = self.fixture()
        with self.assertRaises(ValueError): head(q, p[:, :3], n[:, :3], t)
        with self.assertRaises(ValueError): head(q, p, n, t[:, :1])
        with self.assertRaises(ValueError): head(q, p, n, t, support_valid=torch.zeros(2, 2, 4, dtype=torch.bool))
        bad = t.clone(); bad[0, 0, 0] = float('nan')
        with self.assertRaises(ValueError): head(q, p, n, bad)


class WrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_b2_mask_loss_reaches_encoder_and_matching_head_only(self):
        m = model(); x, y = inputs(2)
        a = m(x, context(), y, return_details=True)
        a['loss'].backward()
        self.assertGreater(float(m.encoder.project.weight.grad.norm()), 0)
        self.assertGreater(float(m.head.text_project.weight.grad.norm()), 0)
        self.assertGreater(float(m.head.text_key.weight.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in m.qwen.parameters()))
        self.assertTrue(all(p.grad is None for p in m.connector.parameters()))
        self.assertEqual(a['metrics']['query_dates'], 2)
        self.assertEqual(a['metrics']['support_date_instances'], 32)
        self.assertEqual(a['metrics']['costs_cumulative']['patch_date_encodes'], 34)
        # Inherited B2 checkpointing repeats the same five encodes on backward.
        self.assertEqual(m.costs['patch_date_encodes'], 68)
        self.assertEqual(m.costs['checkpoint_recompute_calls'], 5)
        facts = context('grapevine: perennial woody vines with seasonal dormancy',
                        'leguminous fodder: some forage legumes regrow after cutting')
        b = m(x, facts, return_details=True)
        self.assertGreater(float((a['logits'] - b['logits']).abs().max().detach()), 1e-8)
        self.assertGreater(float((a['attention'] - b['attention']).abs().max().detach()), 1e-8)
        m.close()

    def test_b0_freeze_target_leakage_and_role_routing(self):
        m = model('B0'); x, y = inputs()
        a = m(x, context(), y)
        b = m(x, context(), dict(target_mask=~y['target_mask'], label_valid=y['label_valid']))
        self.assertTrue(torch.equal(a['logits'], b['logits']))
        self.assertFalse(torch.equal(a['loss'], b['loss']))
        a['loss'].backward()
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in m.encoder.parameters()))
        changed = copy.deepcopy(x)
        for pair in changed['support_pairs']:
            pair['positive'], pair['counterexample'] = pair['counterexample'], pair['positive']
        c = m(changed, context('leguminous fodder', 'grapevine'))
        self.assertTrue(bool((a['logits'].sigmoid() + c['logits'].sigmoid() <= 1 + 1e-7).all()))
        query_changed = copy.deepcopy(x)
        query_changed['query']['s2'] = np.roll(query_changed['query']['s2'], 3, axis=-1)
        d = m(query_changed, context())
        self.assertGreater(float((a['logits'] - d['logits']).abs().max().detach()), 1e-8)
        m.close()

    def test_restore_includes_new_architecture_and_frozen_identities(self):
        m = model(); x, y = inputs()
        optimizer = torch.optim.SGD(m.trainable_parameters(), lr=.01)
        m(x, context(), y)['loss'].backward(); optimizer.step()
        state = copy.deepcopy(m.trainable_state_dict())
        expected = m(x, context())['logits'].detach()
        other = model(); other.load_trainable_state_dict(state)
        self.assertTrue(torch.equal(expected, other(x, context())['logits'].detach()))
        self.assertEqual(state['identity']['architecture'], 'oe11_role_object_matching_v0')
        for field in ('reader', 'eo', 'architecture', 'matching_head', 'role_serialization'):
            changed = copy.deepcopy(state); changed['identity'][field] = 'WRONG'
            with self.assertRaisesRegex(ValueError, 'identity'):
                other.load_trainable_state_dict(changed)
        m.close(); other.close()

    def test_inherited_cache_boundaries_and_token_budget(self):
        m = model('B0', max_text_cache_entries=2)
        x, _ = inputs()
        m(x, context()); calls = m.costs['text_prefill_calls']
        m(x, context()); self.assertEqual(m.costs['text_prefill_calls'], calls)
        with torch.no_grad(): m.qwen.model.embed.weight.add_(.1)
        m(x, context()); self.assertEqual(m.costs['text_prefill_calls'], calls + 2)
        m.close()
        b2 = model()
        with self.assertRaises(ValueError): b2.set_cache_mode('frozen')
        b2.close()
        tiny = model(max_text_tokens=4)
        with self.assertRaisesRegex(ValueError, 'truncation'):
            tiny(x, context())
        tiny.close()

    def test_input_boundary_and_dependency_hash(self):
        m = model(); x, _ = inputs()
        with self.assertRaises(ValueError): m(x, dict(context(), target_mask='hidden'))
        with self.assertRaises(ValueError): m(dict(x, class_id=8), context())
        m.close()
        with tempfile.TemporaryDirectory(prefix='oe11_pin_test_') as temp:
            for name in PINNED_DEPENDENCIES:
                out = Path(temp) / name; out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes((BASE_ROOT / name).read_bytes())
            verify_dependencies(temp)
            with (Path(temp) / 'text_mask_model.py').open('a') as f: f.write('\n# changed\n')
            with self.assertRaisesRegex(ValueError, 'SHA'): verify_dependencies(temp)


if __name__ == '__main__': unittest.main(verbosity=2)
