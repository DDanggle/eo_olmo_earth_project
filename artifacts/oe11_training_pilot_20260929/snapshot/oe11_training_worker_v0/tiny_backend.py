"""Synthetic CPU backend for worker mechanics, never an actual EO/Qwen claim."""
from __future__ import annotations
import sys
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from training_core import require


class TinyBackend:
    def __init__(self, protocol, cases, arm):
        sys.path.insert(0, protocol['paths']['matching_code'])
        from matching_head import RoleObjectMatchingHead
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Linear(12, 8)
                self.encoder.requires_grad_(arm == 'B2')
                self.head = RoleObjectMatchingHead(8, 16, attention_dimension=6,
                                                   hidden_dimension=12, output_size=(4, 4))
                self.qwen = nn.Linear(4, 16); self.qwen.requires_grad_(False)
                self.connector = nn.Linear(2, 2); self.connector.requires_grad_(False)
                self.dropout = nn.Dropout(.2)  # Makes RNG restoration observable.
                self.costs = {'tiny_calls': 0}
            def forward(self, data, context, target=None):
                self.costs['tiny_calls'] += 1
                q = self.dropout(self.encoder(data['query'])).reshape(1, 2, 2, 8)
                p = self.encoder(data['positive'])[None]; n = self.encoder(data['counterexample'])[None]
                vectors = []
                with torch.no_grad():
                    for role in ('positive', 'counterexample'):
                        raw = context[role]
                        value = torch.tensor([len(raw), sum(map(ord, raw)) % 97,
                                              raw.count(' '), raw.count('\n')], dtype=torch.float32) / 97
                        vectors.append(self.qwen(value))
                z = self.head(q, p, n, torch.stack(vectors)[None])[0]
                loss = None
                if target is not None:
                    y = torch.as_tensor(target['target_mask'], dtype=torch.float32)
                    pz = z.sigmoid()
                    loss = F.binary_cross_entropy_with_logits(z, y) + 1 - (2 * (pz * y).sum() + 1) / (pz.sum() + y.sum() + 1)
                return dict(logits=z, loss=loss, metrics={'synthetic_cpu': True})
            def trainable_parameters(self):
                for obj in (self.encoder, self.head):
                    yield from (p for p in obj.parameters() if p.requires_grad)
            def trainable_state_dict(self):
                return dict(identity={'kind': 'synthetic_worker_mechanics', 'arm': arm},
                            head=self.head.state_dict(),
                            **({'encoder': self.encoder.state_dict()} if arm == 'B2' else {}))
            def load_trainable_state_dict(self, state):
                require(state['identity'] == self.trainable_state_dict()['identity'], 'Tiny model identity')
                self.head.load_state_dict(state['head'])
                if arm == 'B2': self.encoder.load_state_dict(state['encoder'])
            def clear_cache(self): pass
            def clear_text_cache(self): pass
            def close(self): pass
        self.model = TinyModel()
        self.allowed = list(cases['episode_ids'])
        self.identity = {'kind': 'synthetic_cpu_no_actual_EO_or_Qwen'}

    def fetch(self, eid, condition):
        require(eid in self.allowed, 'Unregistered tiny case')
        index = sorted({x.split(':')[0] for x in self.allowed}).index(eid.split(':')[0])
        k = 8 if eid.endswith(':k8') else 1
        generator = torch.Generator().manual_seed(index + 170)
        data = dict(query=torch.randn(4, 12, generator=generator),
                    positive=torch.randn(k, 12, generator=generator),
                    counterexample=torch.randn(k, 12, generator=generator))
        y = np.zeros((4, 4), dtype=bool)
        if index % 2 == 0: y[:2, :3] = True
        context = dict(instruction='synthetic CPU test', positive='crop alpha', counterexample='crop beta')
        if condition == 'matched_knowledge':
            context['positive'] += '\nperennial woody growth with dormancy'
            context['counterexample'] += '\nforage with regrowth after cutting'
        return data, dict(target_mask=y, label_valid=np.ones_like(y)), context, k
