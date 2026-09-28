"""Read-only CPU inspection of trusted, locally generated OE10 v3 checkpoints.

Does not reproduce backward, diagnose nondeterminism, or independently rerun
cold resume. Requires completed engineering receipts. Never imports a GPU model.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

PINNED_CONFIG = '0d531a67ad3e477e7011efabcceb01ed80f430aa0a0a3d344fe18cec0f229b8a'
PINNED_WEIGHTS = '57f7b66faf206db1307670673839e639d3a19c305f6ad968c62392ad3e88deec'
PINNED_LOADER = '2ff982cda47fa53e8bb3a9f4ff3879816e4297770ed6e6c6cf49f5e323562ba7'
EXPECTED_EPS = 1e-6
OFFENDER = 'encoder.blocks.1.attn.rope_mixed_freqs'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def stats(values):
    import numpy as np
    a = np.asarray(values, dtype=np.float64)
    require(a.size > 0 and np.isfinite(a).all(), 'empty/nonfinite diagnostic array')
    return {'shape': list(a.shape), 'numel': int(a.size), 'min': float(a.min()),
            'max': float(a.max()), 'mean': float(a.mean()), 'std': float(a.std()),
            'abs_max': float(np.abs(a).max()), 'abs_median': float(np.median(np.abs(a))),
            'l2': float(np.linalg.norm(a.ravel())), 'nonzero_count': int(np.count_nonzero(a))}


def moment_stats(state, group):
    """First-step exp_avg/(1-beta1) is the stored clipped gradient, not raw grad."""
    import numpy as np
    step = int(state['step'])
    require(step >= 1, 'optimizer step must be positive')
    b1, b2 = map(float, group['betas'])
    m = np.asarray(state['exp_avg'], dtype=np.float64)
    v = np.asarray(state['exp_avg_sq'], dtype=np.float64)
    require(m.shape == v.shape and np.isfinite(v).all() and (v >= 0).all(), 'invalid Adam moments')
    root = np.sqrt(v / (1 - b2 ** step))
    corrected_m = m / (1 - b1 ** step)
    return {'step': step, 'exp_avg': stats(m), 'sqrt_exp_avg_sq': stats(np.sqrt(v)),
            'bias_corrected_sqrt_exp_avg_sq': stats(root),
            'bias_corrected_exp_avg': stats(corrected_m),
            'bias_corrected_rms_below_eps_fraction': float(np.mean(root < group['eps'])),
            'eps_fraction_of_denominator_mean': float(np.mean(group['eps'] / (root + group['eps']))),
            'lr_times_adam_ratio_without_weight_decay': stats(group['lr'] * corrected_m / (root + group['eps'])),
            'gradient_note': 'At step1 only, bias_corrected_exp_avg reconstructs the clipped gradient stored by Adam; raw pre-clipping gradient is unavailable.'}


def validate_groups(groups, arm, encoder_names):
    roles = [g.get('role') for g in groups]
    require(len(set(roles)) == len(roles), 'duplicate optimizer roles')
    require(set(roles) == ({'readouts'} if arm == 'B0' else {'encoder', 'native_decoder', 'readouts'}), 'optimizer role/arm mismatch')
    ids = [pid for g in groups for pid in g['params']]
    require(len(ids) == len(set(ids)), 'duplicate optimizer parameter IDs')
    for group in groups:
        require(group['eps'] == EXPECTED_EPS, 'saved optimizer epsilon differs from v3')
        require(group.get('foreach') is False and group.get('fused') is False, 'optimizer foreach/fused differs from v3')
    if arm == 'B0':
        return {}
    eg = next(g for g in groups if g['role'] == 'encoder')
    require(len(eg['params']) == len(encoder_names), 'encoder optimizer length/order cannot be mapped')
    return dict(zip(encoder_names, eg['params']))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['checkpoint-dir', 'source-root', 'deps-root', 'engineering-root', 'out']:
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args()
    require(not a.out.exists(), 'output already exists; preserve audit history')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    report = {'status': 'initializing', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'engineering_root': str(a.engineering_root.resolve()), 'cpu_only': True,
              'scope': 'Independent saved epsilon, optimizer parameter order, and actual encoder/RoPE weight changes; train-only engineering, not P2 efficacy.',
              'limitations': ['No new backward or cold-resume rerun.', 'Resume parity is receipt-reported; resume worker does not save its final weights.',
                             'Moment statistics do not establish the cause of nondeterminism.', 'No raw pre-clipping gradients saved.'],
              'arms': {}, 'files_read': []}
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        sys.path[:0] = [str(a.source_root.resolve()), str(a.deps_root.resolve())]
        import numpy as np
        import torch
        import olmoearth_pretrain
        from olmoearth_pretrain.model_loader import load_model_from_path
        torch.set_num_threads(4)
        require(Path(olmoearth_pretrain.__file__).resolve().is_relative_to(a.source_root.resolve()), 'wrong official package')
        require(sha(a.source_root / 'olmoearth_pretrain/model_loader.py') == PINNED_LOADER, 'official loader source mismatch')
        report['versions'] = {'torch': torch.__version__, 'numpy': np.__version__}
        config = a.checkpoint_dir / 'config.json'
        weights = a.checkpoint_dir / 'model.pt'
        # Official native identity names the weights file; do not guess another checkpoint.
        refs = {}
        for arm in ['B0', 'B2']:
            ref = a.engineering_root / f'{arm}_270927_reference'
            rr = json.loads((ref / 'receipt.json').read_text())
            resume = json.loads((a.engineering_root / f'{arm}_270927_resume/receipt.json').read_text())
            require(rr['status'] == 'reference_completed' and resume['status'] == 'cold_resume_passed' and resume['pass_result'], 'engineering not completed/passed')
            require(rr['source_hashes'] == resume['source_hashes'], 'reference/resume source mismatch')
            require(not rr['development_metrics_accessed'] and not resume['development_metrics_accessed'], 'development metrics accessed')
            native_hashes = rr['native_identity']['checkpoint_hashes']
            weight_names = [n for n, h in native_hashes.items() if h == PINNED_WEIGHTS]
            require(len(weight_names) == 1, 'reference missing pinned original weights')
            weights = a.checkpoint_dir / weight_names[0]
            require(weights.parent.resolve() == a.checkpoint_dir.resolve(), 'weights filename escapes checkpoint')
            refs[arm] = (ref, rr, resume)
        require(refs['B0'][1]['source_hashes'] == refs['B2'][1]['source_hashes'], 'arms use different worker sources')
        require(sha(config) == PINNED_CONFIG and sha(weights) == PINNED_WEIGHTS, 'original checkpoint identity mismatch')
        report['original_checkpoint'] = {'config_sha256': PINNED_CONFIG, 'weights_path': str(weights), 'weights_sha256': PINNED_WEIGHTS}
        controller = json.loads((a.engineering_root / 'status.json').read_text())
        require(controller['status'] == 'completed' and controller['protected_and_snapshot_unchanged'], 'controller integrity gate')
        model = load_model_from_path(str(a.checkpoint_dir)).to(device='cpu', dtype=torch.float32)
        # Exactly the worker's model.parameters() order, deduplicated by named_parameters.
        encoder_ids = {id(p) for p in model.encoder.parameters()}
        encoder_trainable = [n for n, p in model.named_parameters() if id(p) in encoder_ids and p.requires_grad]
        original = {n: p.detach().cpu().numpy().copy() for n, p in model.named_parameters() if id(p) in encoder_ids}
        rope_names = [n for n in original if 'rope_mixed_freqs' in n]
        require(OFFENDER in original and rope_names, 'actual encoder has no expected RoPE frequencies')
        report['encoder_parameter_count'] = len(original)
        report['optimizer_order_method'] = 'Filter official full model.named_parameters() by identity in model.encoder.parameters() and requires_grad, matching the worker deduplication/group order; validate every saved parameter and moment shape.'
        del model
        gc.collect()

        def load(path):
            size = path.stat().st_size
            require(size <= 5 * 2**30, 'unexpected checkpoint above 5 GiB read cap')
            report['files_read'].append({'path': str(path), 'bytes': size})
            # Trusted worker-generated checkpoint includes RNG Python/NumPy objects.
            return torch.load(path, map_location='cpu', weights_only=False, mmap=True)

        for arm in ['B0', 'B2']:
            ref, rr, resume = refs[arm]
            path = ref / 'step1.pt'
            h = sha(path)
            require(h == rr['step1_sha256'], 'saved step1 SHA differs from reference receipt')
            step1 = load(path)
            require(step1['step'] == 1 and step1['identity']['arm'] == arm and step1['identity']['seed'] == 270927, 'wrong saved step1 identity')
            require(step1['identity']['sources'] == rr['source_hashes'], 'step1 source identity differs')
            groups = step1['optimizer']['param_groups']
            mapping = validate_groups(groups, arm, encoder_trainable)
            ar = {'step1_sha256': h, 'saved_optimizer_groups': [{k: v for k, v in g.items() if k != 'params'} | {'parameter_count': len(g['params'])} for g in groups],
                  'rope': {}, 'resume_receipt': {k: resume[k] for k in ['pass_result', 'state_max_abs_differences', 'prediction_max_abs_difference', 'total_loss_abs_difference']}}
            native1 = step1['modules']['native']
            for name in original:
                require(name in native1 and tuple(native1[name].shape) == original[name].shape, 'saved encoder parameter mismatch: ' + name)
            if arm == 'B2':
                eg = next(g for g in groups if g['role'] == 'encoder')
                for name, pid in mapping.items():
                    state = step1['optimizer']['state'].get(pid)
                    if state:
                        require(tuple(state['exp_avg'].shape) == original[name].shape and tuple(state['exp_avg_sq'].shape) == original[name].shape, 'optimizer parameter/moment order mismatch: ' + name)
                pid = mapping[OFFENDER]
                require(pid in step1['optimizer']['state'], 'offending RoPE has no optimizer state')
                s = step1['optimizer']['state'][pid]
                ar['block1_step1_moments'] = moment_stats({k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v) for k, v in s.items()}, eg)
                require(int(s['step']) == 1 and bool(s['exp_avg'].abs().max() > 0), 'RoPE optimizer did not receive nonzero first-step gradient')
            for name in rope_names:
                current = native1[name].float().numpy()
                ar['rope'][name] = {'original': stats(original[name]), 'step1': stats(current),
                                    'step1_minus_original': stats(current.astype(np.float64) - original[name]),
                                    '_step1_array': current.copy()}
            ar['step1_all_encoder_max_abs_delta'] = max(float(np.max(np.abs(native1[n].float().numpy().astype(np.float64) - v))) for n, v in original.items())
            del native1, step1
            if arm == 'B2':
                del s, state
            gc.collect()
            step2 = load(ref / 'expected_step2.pt')
            native2 = step2['modules']['native']
            require(step2['row']['step'] == 2 and step2['row']['k'] == 8, 'unexpected reference second step')
            for name in original:
                require(name in native2 and tuple(native2[name].shape) == original[name].shape, 'step2 encoder shape/name mismatch')
            ar['step2_all_encoder_max_abs_delta'] = max(float(np.max(np.abs(native2[n].float().numpy().astype(np.float64) - v))) for n, v in original.items())
            for name in rope_names:
                current = native2[name].float().numpy()
                old = ar['rope'][name].pop('_step1_array')
                ar['rope'][name].update(step2=stats(current), step2_minus_original=stats(current.astype(np.float64) - original[name]), step2_minus_step1=stats(current.astype(np.float64) - old))
            if arm == 'B0':
                require(ar['step1_all_encoder_max_abs_delta'] == 0 and ar['step2_all_encoder_max_abs_delta'] == 0, 'B0 encoder unexpectedly changed')
            else:
                require(ar['rope'][OFFENDER]['step2_minus_original']['abs_max'] > 0, 'B2 offending RoPE did not actually update')
                require(ar['rope'][OFFENDER]['step2_minus_step1']['abs_max'] > 0, 'B2 offending RoPE second update absent')
            report['arms'][arm] = ar
            del native2, step2
            gc.collect()
        require(not torch.cuda.is_initialized(), 'CUDA unexpectedly initialized')
        report['status'] = 'passed_saved_optimizer_and_weight_audit'
        report['diagnosis'] = 'Actual stored epsilon and weight updates checked. Nondeterminism cause remains unproven; resume parity values are copied from completed worker receipts.'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = repr(exc)
        raise
    finally:
        a.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'out': str(a.out)}))


if __name__ == '__main__':
    main()
