"""Mask-only engineering loop with safe state dictionaries and resume comparison.

No process creation, model download, development evaluation, or GPU selection.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_torch(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary); temporary.replace(path)
    return sha(path)


def load_owned(path, expected_sha):
    """Only externally hashed own artifacts, never general pickle objects."""
    path = Path(path)
    require(path.is_file() and sha(path) == expected_sha, 'Resume/reference artifact hash mismatch')
    return torch.load(path, map_location='cpu', weights_only=True, mmap=True)


def rng_state():
    numpy_state = np.random.get_state()
    return dict(python=random.getstate(), numpy=dict(kind=numpy_state[0],
        keys=numpy_state[1].tolist(), position=numpy_state[2], has_gauss=numpy_state[3],
        cached_gaussian=numpy_state[4]), torch_cpu=torch.get_rng_state(),
        torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    random.setstate(state['python']); n = state['numpy']
    np.random.set_state((n['kind'], np.asarray(n['keys'], dtype=np.uint32), n['position'],
                        n['has_gauss'], n['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        require(torch.cuda.is_initialized(), 'CUDA RNG restored without CUDA model')
        torch.cuda.set_rng_state_all(state['torch_cuda'])


def compare_nested(actual, expected, tolerance=0.0):
    """Compare every model/AdamW tensor and nested metadata; bounded diagnostics."""
    result = dict(passed=True, max_abs_difference=0.0, tensor_count=0, failures=[])
    def fail(path, reason):
        result['passed'] = False
        if len(result['failures']) < 20: result['failures'].append(dict(path=path, reason=reason))
    def visit(a, b, path):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            result['tensor_count'] += 1
            if a.shape != b.shape or a.dtype != b.dtype:
                fail(path, 'shape/dtype mismatch'); return
            ac, bc = a.detach().cpu(), b.detach().cpu()
            if ac.is_floating_point():
                if not bool(torch.isfinite(ac).all() and torch.isfinite(bc).all()):
                    fail(path, 'nonfinite tensor'); return
                delta = float((ac.double() - bc.double()).abs().max()) if ac.numel() else 0.0
                result['max_abs_difference'] = max(result['max_abs_difference'], delta)
                if delta > tolerance: fail(path, 'tensor difference exceeds tolerance')
            elif not torch.equal(ac, bc): fail(path, 'integer/bool tensor mismatch')
        elif isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b): fail(path, 'dictionary keys mismatch'); return
            for key in a: visit(a[key], b[key], path + '/' + str(key))
        elif isinstance(a, (tuple, list)) and isinstance(b, type(a)):
            if len(a) != len(b): fail(path, 'sequence length mismatch'); return
            for index, (av, bv) in enumerate(zip(a, b)): visit(av, bv, path + '/' + str(index))
        elif type(a) != type(b) or a != b: fail(path, 'metadata mismatch')
    visit(actual, expected, '')
    return result


def parameter_versions(module):
    return tuple((name, id(p), p._version, p.requires_grad) for name, p in module.named_parameters())


class EngineeringTrainer:
    def __init__(self, backend, protocol, cases, identity, out, arm, condition):
        self.backend = backend; self.model = backend.model
        self.protocol = protocol; self.cases = cases; self.identity = identity
        self.out = Path(out); self.arm = arm; self.condition = condition
        self.order = list(cases['order'])
        self.order_sha = hashlib.sha256(('\n'.join(self.order) + '\n').encode()).hexdigest()
        self.fit_ids = cases.get('fit_episode_ids', cases['episode_ids'])
        self.step = 0; self.initial_metrics = None
        self.costs = dict(optimizer_steps=0, training_observation_instances=0,
                          training_seconds=0.0, diagnostic_seconds=0.0)
        opt = protocol['optimizer']
        head_parameters = [p for p in self.model.head.parameters() if p.requires_grad]
        encoder_parameters = [p for p in self.model.encoder.parameters() if p.requires_grad]
        require(bool(encoder_parameters) == (arm == 'B2'), 'Encoder freeze arm mismatch')
        groups = []
        if encoder_parameters:
            groups.append(dict(params=encoder_parameters, lr=opt['encoder_lr'], role='encoder'))
        groups.append(dict(params=head_parameters, lr=opt['head_lr'], role='head'))
        self.parameters = encoder_parameters + head_parameters
        require(len({id(p) for p in self.parameters}) == len(self.parameters), 'Duplicate optimizer parameter')
        require({id(p) for p in self.parameters} == {id(p) for p in self.model.trainable_parameters()},
                'Optimizer/model trainable parameters differ')
        self.optimizer = torch.optim.AdamW(groups, weight_decay=opt['weight_decay'],
                                            eps=opt['eps'], foreach=False, fused=False)
        self.reader_versions = parameter_versions(self.model.qwen)
        self.unused_versions = parameter_versions(self.model.connector)
        self.frozen_encoder_versions = parameter_versions(self.model.encoder) if arm == 'B0' else None
        self.attention_gradient_seen = {name: False for name in ('query_key', 'object_key', 'text_key')}

    def sync(self):
        if next(self.model.head.parameters()).is_cuda: torch.cuda.synchronize()

    def clear_caches(self):
        self.model.clear_cache(); self.model.clear_text_cache()

    def frozen_gate(self):
        frozen = [('reader', self.model.qwen, self.reader_versions),
                  ('unused_connector', self.model.connector, self.unused_versions)]
        if self.arm == 'B0': frozen.append(('encoder', self.model.encoder, self.frozen_encoder_versions))
        for name, module, versions in frozen:
            require(all(not p.requires_grad and p.grad is None for p in module.parameters()),
                    name + ' must stay frozen with grad=None')
            require(parameter_versions(module) == versions, name + ' parameter version changed')

    def gradients(self, k):
        counts, anchors = {}, {}
        for name, module in (('encoder', self.model.encoder), ('head', self.model.head)):
            count = 0; best = (0.0, None, None)
            for pn, p in module.named_parameters():
                if p.grad is None: continue
                require(bool(torch.isfinite(p.grad).all()), 'Nonfinite gradient: ' + name + '/' + pn)
                magnitude = float(p.grad.detach().abs().max())
                count += magnitude > 0
                if magnitude > best[0]: best = (magnitude, pn, p)
            counts[name] = int(count)
            if best[2] is not None:
                anchors[name] = (best[1], best[2], best[2].detach().cpu().clone())
        require(counts['head'] > 0, 'No head gradient')
        require((counts['encoder'] > 0) == (self.arm == 'B2'), 'EO mask-gradient arm gate')
        detail = {}
        for name in ('text_project', 'background_scorer', 'query_key', 'object_key', 'text_key'):
            module = getattr(self.model.head, name)
            detail[name] = sum(p.grad is not None and bool(p.grad.abs().max() > 0) for p in module.parameters())
        require(detail['text_project'] > 0 and detail['background_scorer'] > 0,
                'Text/background loss-gradient route absent')
        if k == 8:
            for name in self.attention_gradient_seen:
                self.attention_gradient_seen[name] |= detail[name] > 0
        self.frozen_gate()
        return counts, detail, anchors

    def train_one(self):
        full_started = time.monotonic()
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        eid = self.order[self.step]
        model_input, target, context, k = self.backend.fetch(eid, self.condition)
        fetch_seconds = time.monotonic() - full_started
        self.sync(); started = time.monotonic()
        output = self.model(model_input, context, target)
        loss = output['loss']; require(bool(torch.isfinite(loss)) and loss.numel() == 1, 'Finite scalar mask loss required')
        prediction = output['logits'].detach().cpu().clone()
        loss.backward(); counts, detail, anchors = self.gradients(k)
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.protocol['optimizer']['clip_grad_norm'],
                                             error_if_nonfinite=True, foreach=False)
        self.optimizer.step(); self.sync()
        deltas = {name: dict(parameter=pn, max_abs_change=float((p.detach().cpu() - before).abs().max()))
                  for name, (pn, p, before) in anchors.items()}
        require(deltas['head']['max_abs_change'] > 0, 'AdamW head did not change')
        if self.arm == 'B2': require(deltas['encoder']['max_abs_change'] > 0, 'AdamW encoder did not change')
        self.frozen_gate(); self.step += 1
        elapsed = time.monotonic() - started
        self.costs['optimizer_steps'] += 1
        self.costs['training_observation_instances'] += 2 + 16 * k
        full_elapsed = time.monotonic() - full_started
        self.costs['training_seconds'] += full_elapsed
        row = dict(step=self.step, episode_id=eid, k_pairs=k, mask_loss=float(loss.detach()),
                   grad_norm=float(norm), finite_nonzero_gradient_tensors=counts,
                   head_gradient_routes=detail, actual_parameter_changes=deltas,
                   learning_rates={g['role']: g['lr'] for g in self.optimizer.param_groups},
                   forward_backward_optimizer_seconds=elapsed, fetch_seconds=fetch_seconds,
                   full_step_seconds=full_elapsed, model_metrics=output.get('metrics', {}))
        with (self.out / 'train_log.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')
        return row, prediction

    def evaluate_training_cases(self):
        """Fixed train cases only; never reports a development/generalization score."""
        self.model.eval(); started = time.monotonic(); rows = []
        with torch.no_grad():
            for eid in self.fit_ids:
                model_input, target, context, k = self.backend.fetch(eid, self.condition)
                output = self.model(model_input, context, target)
                require(bool(torch.isfinite(output['loss'])), 'Nonfinite training diagnostic loss')
                z = output['logits'].detach().cpu()
                valid = torch.as_tensor(target['label_valid'], dtype=torch.bool)
                y = torch.as_tensor(target['target_mask'], dtype=torch.bool)[valid]
                pred = z[valid] >= 0
                positive = int(y.sum()); predicted = int(pred.sum())
                intersection = int((pred & y).sum()); union = int((pred | y).sum())
                rows.append(dict(episode_id=eid, k_pairs=k, mask_loss=float(output['loss']),
                    target_pixels=positive, predicted_pixels=predicted,
                    iou=intersection / union if positive else None,
                    recall=intersection / positive if positive else None,
                    absent_positive_fraction=predicted / y.numel() if not positive else None))
        self.costs['diagnostic_seconds'] += time.monotonic() - started
        self.model.train()
        present = [r for r in rows if r['target_pixels']]
        absent = [r for r in rows if not r['target_pixels']]
        return dict(scope='fixed_training_cases_not_generalization', rows=rows,
            mean_loss=sum(r['mask_loss'] for r in rows) / len(rows),
            mean_positive_iou=sum(r['iou'] for r in present) / len(present) if present else None,
            mean_positive_recall=sum(r['recall'] for r in present) / len(present) if present else None,
            mean_absent_positive_fraction=sum(r['absent_positive_fraction'] for r in absent) / len(absent) if absent else None,
            positive_cases=len(present), absent_cases=len(absent))

    def text_diagnostics(self):
        eid = self.fit_ids[0]
        model_input, _, names, _ = self.backend.fetch(eid, 'names_only')
        _, _, facts, _ = self.backend.fetch(eid, 'matched_knowledge')
        # Keep concept names fixed; swap only the explanatory suffixes.
        def swap_suffix(name, other):
            parts = other.split('\n', 1)
            return name + ('\n' + parts[1] if len(parts) == 2 else '')
        swapped = dict(instruction=names['instruction'],
            positive=swap_suffix(names['positive'], facts['counterexample']),
            counterexample=swap_suffix(names['counterexample'], facts['positive']))
        conditions = dict(names_only=names, matched_knowledge=facts,
                          removed_knowledge_diagnostic=dict(names), swapped_knowledge_diagnostic=swapped)
        self.model.eval(); predictions = {}
        with torch.no_grad():
            for name, ctx in conditions.items():
                predictions[name] = self.model(model_input, ctx)['logits'].detach().cpu()
        self.model.train()
        differences = {name: float((value - predictions['names_only']).abs().max())
                       for name, value in predictions.items()}
        require(differences['removed_knowledge_diagnostic'] == 0, 'Removed facts differ from names-only')
        self.frozen_gate()
        return dict(episode_id=eid, logit_max_difference_vs_names=differences,
                    interpretation='Sensitivity wiring only; no semantic or efficacy claim')

    def finite_optimizer_gate(self):
        for parameter in self.parameters:
            require(bool(torch.isfinite(parameter).all()), 'Nonfinite trainable parameter after optimizer')
        for state in self.optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    require(bool(torch.isfinite(value).all()), 'Nonfinite AdamW state')

    def snapshot(self):
        self.finite_optimizer_gate()
        return dict(schema='oe11_training_state_v0', identity=self.identity,
                    model=self.model.trainable_state_dict(), optimizer=self.optimizer.state_dict(),
                    rng=rng_state(), step=self.step, order=self.order, order_sha256=self.order_sha,
                    model_costs=dict(self.model.costs), cumulative_costs=dict(self.costs),
                    attention_gradient_seen=dict(self.attention_gradient_seen),
                    initial_metrics=self.initial_metrics, writer_pid=os.getpid())

    def restore(self, state):
        require(state['schema'] == 'oe11_training_state_v0' and state['identity'] == self.identity,
                'Checkpoint identity mismatch')
        require(state['order'] == self.order and state['order_sha256'] == self.order_sha, 'Checkpoint order mismatch')
        require(state['writer_pid'] != os.getpid(), 'Resume must use a fresh process')
        self.model.load_trainable_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.step = state['step']; self.costs = dict(state['cumulative_costs'])
        self.model.costs = dict(state['model_costs'])
        self.initial_metrics = state['initial_metrics']
        self.attention_gradient_seen = dict(state['attention_gradient_seen'])
        self.clear_caches(); self.model.train()
        # Everything that can consume RNG during construction/preflight is done.
        restore_rng(state['rng'])
        checks = dict(model=compare_nested(self.model.trainable_state_dict(), state['model']),
                      optimizer=compare_nested(self.optimizer.state_dict(), state['optimizer']),
                      rng=compare_nested(rng_state(), state['rng']))
        require(all(v['passed'] for v in checks.values()), 'Exact immediate restore failed')
        self.frozen_gate()
        return checks

    def comparison(self, row, prediction):
        self.finite_optimizer_gate()
        self.model.eval()
        model_input, _, context, _ = self.backend.fetch(row['episode_id'], self.condition)
        with torch.no_grad():
            post_prediction = self.model(model_input, context)['logits'].detach().cpu().clone()
        self.model.train()
        return dict(identity=self.identity, step=self.step, writer_pid=os.getpid(),
                    model=self.model.trainable_state_dict(), optimizer=self.optimizer.state_dict(),
                    rng=rng_state(), order_sha256=self.order_sha,
                    pre_update_prediction=prediction, post_update_prediction=post_prediction,
                    mask_loss=row['mask_loss'])

    def compare(self, expected, row, prediction):
        require(expected['identity'] == self.identity and expected['step'] == self.step and
                expected['order_sha256'] == self.order_sha, 'Comparison identity/step/order mismatch')
        require(expected['writer_pid'] != os.getpid(), 'Comparison must use a distinct reference process')
        tolerance = self.protocol['resume_tolerance']
        actual = self.comparison(row, prediction)
        result = dict(model=compare_nested(self.model.trainable_state_dict(), expected['model'], tolerance),
                      optimizer=compare_nested(self.optimizer.state_dict(), expected['optimizer'], tolerance),
                      rng=compare_nested(rng_state(), expected['rng']),
                      logits=compare_nested(prediction, expected['pre_update_prediction'], tolerance),
                      post_update_logits=compare_nested(actual['post_update_prediction'], expected['post_update_prediction'], tolerance),
                      mask_loss_abs_difference=abs(row['mask_loss'] - expected['mask_loss']))
        result['passed'] = all(result[k]['passed'] for k in ('model', 'optimizer', 'rng', 'logits', 'post_update_logits')) and result['mask_loss_abs_difference'] <= tolerance
        return result
