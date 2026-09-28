"""Opt-in wrapper over an unchanged, hash-verified OE10 text-mask v2 snapshot.

No training launcher, model download, GPU launch, or dataset access lives here.
Pass the local frozen v2 directory to make_episode_model_class().
"""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from matching_head import RoleObjectMatchingHead

PINNED_DEPENDENCIES = {
    'text_mask_model.py': '30e05f955b88de7aa114d0a6ab7d9f5c7af926b62f704e8e5e0bd18ef75e571e',
    'contracts.py': '4908c438bc7c06813bd26817a4f8ef35735ac00e4a9530a0c3b7c74be452ae06',
    'base_snapshot/episode_model.py': '4997e85692b5600a034af351cd16360ff303a3a133bacb5d5756c53c89098297',
    'base_snapshot/episode_loader.py': '536de03b77bcd5bfedf439b5456d9e0fe60a3789c5f9c1a53c5511ded05d07f6',
    'base_snapshot/native_replay.py': '26c0fab96c328e9a5a022a574caf5f78f7ffda4e6fb01164128283f7918ff670',
}
ROLE_SERIALIZATION = 'oe11_symmetric_isolated_role_context_v0'
NEUTRAL_COUNTER = 'No competing concept is specified in this isolated text encoding.'


def verify_dependencies(base_root):
    base_root = Path(base_root).resolve()
    actual = {name: hashlib.sha256((base_root / name).read_bytes()).hexdigest()
              for name in PINNED_DEPENDENCIES}
    if actual != PINNED_DEPENDENCIES:
        raise ValueError('Pinned OE10 dependency SHA mismatch')
    return actual


def _load(name, path):
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path.resolve():
            raise RuntimeError('Conflicting module namespace: ' + name)
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def make_episode_model_class(base_root):
    """Build a subclass; actual EO/reader/norm identities remain required by v2."""
    base_root = Path(base_root).resolve()
    verify_dependencies(base_root)
    contracts = _load('contracts', base_root / 'contracts.py')
    base = _load('_oe11_verified_text_mask_v2', base_root / 'text_mask_model.py')

    class ObjectMatchingEpisodeModel(base.TextConditionedEpisodeModel):
        def __init__(self, *args, matching_attention_dimension=64,
                     matching_hidden_dimension=128, **kwargs):
            verify_dependencies(base_root)
            super().__init__(*args, **kwargs)
            self.head = RoleObjectMatchingHead(
                self.dimension, int(self.qwen.config.text_config.hidden_size),
                attention_dimension=matching_attention_dimension,
                hidden_dimension=matching_hidden_dimension).to(self.device)

        def role_text_representations(self, context):
            context = contracts.validate_context(context)
            vectors, audits = [], []
            # Use the SAME serialized slot for both concepts. Positive/confuser
            # semantics are conveyed by the head's role axis, not unequal text
            # templates. This preserves whole-role permutation behavior.
            for role in ('positive', 'counterexample'):
                isolated = dict(instruction=context['instruction'], positive=context[role],
                                counterexample=NEUTRAL_COUNTER)
                vector, audit = super().text_representation(isolated)
                vectors.append(vector)
                audits.append(dict(role=role, **audit))
            return torch.stack(vectors, dim=1), audits

        def forward(self, model_input, context, target=None, *, return_details=False):
            text, text_audit = self.role_text_representations(context)
            query, positive, counter = self.features(model_input)
            details = self.head(query, positive, counter, text, return_details=True)
            logits = details['logits'][0]
            loss = None
            if target is not None:
                y, valid = self._target_arrays(target, self.device)
                z, yt = logits[valid], y[valid]
                p = z.sigmoid()
                loss = F.binary_cross_entropy_with_logits(z, yt) + 1 - (
                    2 * (p * yt).sum() + 1) / (p.sum() + yt.sum() + 1)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('Nonfinite mask loss')
            result = dict(logits=logits, loss=loss, mask_loss=loss, metrics={
                'role_text': text_audit, 'role_serialization': ROLE_SERIALIZATION,
                'k_pairs': len(model_input['support_pairs']), 'query_dates': 2,
                'support_date_instances': 16 * len(model_input['support_pairs']),
                'temporal_pooling': 'unchanged_date_mean',
                'language_ce_used': False, 'language_generation_evaluated': False,
                'gold_in_text_prefill': False, 'costs_cumulative': dict(self.costs)})
            if return_details:
                result['attention'] = details['attention'][0]
                result['role_scores'] = details['role_scores'][0]
                result['background_scores'] = details['background_scores'][0]
            return result

        def trainable_state_dict(self):
            state = super().trainable_state_dict()
            state['identity'].update(architecture='oe11_role_object_matching_v0',
                                     matching_head=self.head.configuration(),
                                     role_serialization=ROLE_SERIALIZATION,
                                     matching_dependencies=verify_dependencies(base_root))
            return state

    return ObjectMatchingEpisodeModel
