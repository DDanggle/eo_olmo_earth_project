"""Bounded train-only engineering worker. Launch stages as separate processes.

No child processes, scheduler, development/final evaluation or generation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()


def verify_protocol(path, expected_sha, arm, condition):
    require(sha(path) == expected_sha, 'Protocol SHA mismatch')
    protocol = json.loads(Path(path).read_text())
    require(protocol['schema'] == 'oe11_training_pilot_v0', 'Unexpected protocol schema')
    require(protocol['backend'] in ('actual', 'tiny_cpu'), 'Explicit execution backend required')
    require(arm in protocol['allowed_arms'] and condition in protocol['allowed_conditions'],
            'Arm/condition not authorized by protocol')
    for name in ('seed', 'total_steps', 'split_step', 'max_stage_seconds'):
        require(type(protocol[name]) is int and protocol[name] > 0, 'Positive integer required: ' + name)
    require(0 < protocol['split_step'] < protocol['total_steps'], 'Invalid split boundary')
    require(protocol['max_stage_seconds'] <= 1800, 'Per-stage engineering cap exceeds 30 minutes')
    require(protocol['resume_tolerance'] == 1e-6, 'Frozen resume tolerance must be 1e-6')
    for name in ('encoder_lr', 'head_lr', 'weight_decay', 'eps', 'clip_grad_norm'):
        value = protocol['optimizer'][name]
        require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                'Finite positive optimizer value required: ' + name)
    files = protocol['file_sha256']
    require(type(files) is dict and bool(files), 'Explicit source/input file hashes required')
    for name, digest in files.items():
        require(Path(name).is_absolute(), 'Protocol file keys must be absolute paths')
        require(sha(name) == digest, 'Pinned file SHA mismatch: ' + name)
    root = Path(__file__).resolve().parent
    required = {root / 'worker.py', root / 'training_core.py',
                root / ('actual_backend.py' if protocol['backend'] == 'actual' else 'tiny_backend.py')}
    paths = {name: Path(value).resolve() for name, value in protocol['paths'].items()}
    required.add(paths['cases'])
    require(paths['worker'] == Path(__file__).resolve(), 'Wrong worker source path')
    if protocol['backend'] == 'actual':
        require(os.environ.get('OE11_MANUAL_BOUNDED_JOB') == '1',
                'Actual worker requires the manual bounded-controller marker')
        expected_uuid = os.environ.get('OE11_EXPECTED_GPU_UUID', '')
        require(os.environ.get('OE11_PHYSICAL_GPU_INDEX') in ('0', '1') and
                expected_uuid.startswith('GPU-') and os.environ.get('CUDA_VISIBLE_DEVICES') == expected_uuid,
                'Controller must bind CUDA_VISIBLE_DEVICES to the prechecked physical GPU UUID')
        for key in ('contexts', 'model_identity'): required.add(paths[key])
        required.update(paths['matching_code'] / name for name in ('matching_head.py', 'episode_adapter.py'))
        required.update(paths['loader_code'] / name for name in ('loader_adapter.py', 'frozen_episode_loader.py'))
        required.update(paths['text_mask_code'] / name for name in ('text_mask_model.py', 'contracts.py',
            'base_snapshot/episode_model.py', 'base_snapshot/episode_loader.py', 'base_snapshot/native_replay.py'))
        required.update((paths['prepared'] / 'manifest.jsonl', paths['episodes'] / 'episode_contract.json',
                         paths['episodes'] / 'episodes_train.jsonl', paths['episodes'] / 'scoring/scoring_train.jsonl'))
        require(protocol['max_text_tokens'] == 1024 and 0 <= protocol['max_text_cache_entries'] <= 64,
                'Text token/cache limits outside engineering contract')
    else:
        required.add(paths['matching_code'] / 'matching_head.py')
    require(required <= {Path(p).resolve() for p in files}, 'Protocol source/input hash coverage incomplete')
    cases = json.loads(paths['cases'].read_text())
    require(cases['split'] == 'train', 'Only train cases allowed')
    for field in ('seed', 'total_steps', 'split_step'):
        require(cases[field] == protocol[field], 'Cases/protocol mismatch: ' + field)
    ids, order = cases['episode_ids'], cases['order']
    require(type(ids) is list and ids and len(set(ids)) == len(ids), 'Unique nonempty registered cases required')
    require(len(order) == protocol['total_steps'] and set(order) <= set(ids), 'Fixed complete training order required')
    fit = cases.get('fit_episode_ids', ids)
    require(bool(fit) and len(set(fit)) == len(fit) and set(fit) <= set(ids), 'Fit probe outside registered train cases')
    require(cases['query_positions'] == [2, 5] and cases['support_positions'] == list(range(8)),
            'Engineering observation budget changed')
    for p, digest in cases.get('input_hashes', {}).items():
        require(files.get(p) == digest, 'Case-bound dataset hash absent/different in protocol')
    return protocol, cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--stage', choices=('uninterrupted', 'split', 'resume'), required=True)
    parser.add_argument('--arm', choices=('B0', 'B2'), required=True)
    parser.add_argument('--condition', choices=('names_only', 'matched_knowledge'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path); parser.add_argument('--resume-sha256')
    parser.add_argument('--reference-result', type=Path); parser.add_argument('--reference-sha256')
    parser.add_argument('--verify-only', action='store_true', help='Validate protocol/source/input pins without importing torch or loading models')
    args = parser.parse_args()
    protocol, cases = verify_protocol(args.protocol, args.protocol_sha256, args.arm, args.condition)
    if args.verify_only:
        print(json.dumps(dict(status='protocol_source_input_verified_only', backend=protocol['backend'],
                              total_steps=protocol['total_steps'], train_cases=len(cases['episode_ids']),
                              model_loaded=False, GPU_initialized=False)))
        return
    need_resume = (args.resume_from, args.resume_sha256, args.reference_result, args.reference_sha256)
    require(all(x is not None for x in need_resume) if args.stage == 'resume' else all(x is None for x in need_resume),
            'Resume/reference arguments must appear only for resume stage')
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = dict(status='initializing', backend=protocol['backend'], stage=args.stage, arm=args.arm,
        condition=args.condition, pid=os.getpid(), protocol_sha256=args.protocol_sha256,
        actual_GPU_model_execution=False, development_or_final_access=False,
        language_generation=False, language_CE=False, official_native_replay=False,
        claim='Mask-only trainability/resume engineering; not EO preservation or performance superiority')
    def update(**values):
        receipt.update(values); receipt['stage_elapsed_seconds'] = time.monotonic() - started
        tmp = args.out / 'receipt.json.tmp'
        tmp.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n'); tmp.replace(args.out / 'receipt.json')
        print(json.dumps({k: receipt[k] for k in ('status', 'stage', 'pid', 'stage_elapsed_seconds')}), flush=True)
    def timeout(signum, frame): raise TimeoutError('Engineering stage wall-time cap; no implicit retry')
    signal.signal(signal.SIGALRM, timeout); signal.alarm(protocol['max_stage_seconds'])
    backend = None
    update()
    try:
        # Set deterministic CUDA environment before importing torch/initializing it.
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        import random
        import numpy as np
        import torch
        from training_core import EngineeringTrainer, load_owned, save_torch
        torch.set_num_threads(2); torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        random.seed(protocol['seed']); np.random.seed(protocol['seed']); torch.manual_seed(protocol['seed'])
        if protocol['backend'] == 'actual':
            require(torch.cuda.is_available(), 'Actual backend requires the prechecked GPU')
            torch.cuda.set_device(0); torch.cuda.manual_seed_all(protocol['seed'])
            actual_uuid = str(getattr(torch.cuda.get_device_properties(0), 'uuid', ''))
            normalized = actual_uuid.lower().removeprefix('gpu-').replace('-', '')
            expected_uuid = os.environ['OE11_EXPECTED_GPU_UUID'].lower().removeprefix('gpu-').replace('-', '')
            require(bool(normalized) and normalized == expected_uuid, 'Torch physical GPU UUID mismatch/unavailable')
            receipt['physical_gpu_index'] = os.environ['OE11_PHYSICAL_GPU_INDEX']
            receipt['verified_torch_gpu_uuid'] = actual_uuid
            from actual_backend import ActualBackend
            backend = ActualBackend(protocol, cases, args.arm)
            receipt['actual_GPU_model_execution'] = True
        else:
            from tiny_backend import TinyBackend
            backend = TinyBackend(protocol, cases, args.arm)
        identity = dict(protocol_sha256=args.protocol_sha256, arm=args.arm, condition=args.condition,
                        backend=protocol['backend'], source_and_input_files=protocol['file_sha256'],
                        model=backend.identity, torch_version=str(torch.__version__),
                        fixed_seed=protocol['seed'], optimizer=protocol['optimizer'])
        trainer = EngineeringTrainer(backend, protocol, cases, identity, args.out, args.arm, args.condition)
        update(status='ready', total_steps=protocol['total_steps'], split_step=protocol['split_step'],
               registered_train_cases=len(cases['episode_ids']), fit_train_cases=len(trainer.fit_ids),
               order_sha256=trainer.order_sha, trainable_parameters=sum(p.numel() for p in trainer.parameters))
        reference = None
        if args.stage == 'resume':
            saved = load_owned(args.resume_from, args.resume_sha256)
            require(saved['step'] == protocol['split_step'], 'Checkpoint must be exactly at split boundary')
            exact = trainer.restore(saved); del saved
            # Loading mapped read-only comparison tensors consumes no RNG.
            reference = load_owned(args.reference_result, args.reference_sha256)
            require(reference['comparison']['identity'] == identity, 'Reference identity mismatch')
            update(status='restored', completed_updates=trainer.step, immediate_restore_exact=exact)
        else:
            trainer.initial_metrics = trainer.evaluate_training_cases()
            # Ensure first train step starts from empty caches in both seed-matched processes.
            trainer.clear_caches()
            update(status='training', initial_train_fit=trainer.initial_metrics)
        stop = protocol['split_step'] if args.stage == 'split' else protocol['total_steps']
        next_path = args.out / 'next_step_comparison.pt'; next_sha = None
        resume_checks = {}
        while trainer.step < stop:
            row, prediction = trainer.train_one()
            if trainer.step == protocol['split_step']:
                trainer.clear_caches()
                if args.stage == 'split':
                    checkpoint_sha = save_torch(args.out / 'checkpoint.pt', trainer.snapshot())
                    update(status='split_completed', completed_updates=trainer.step,
                           checkpoint_sha256=checkpoint_sha, costs=trainer.costs, model_costs=backend.model.costs,
                           initial_train_fit=trainer.initial_metrics,
                           attention_gradient_seen=trainer.attention_gradient_seen,
                           frozen_reader_parameter_versions_unchanged=True)
            if trainer.step == protocol['split_step'] + 1:
                if args.stage == 'uninterrupted':
                    next_sha = save_torch(next_path, trainer.comparison(row, prediction))
                elif args.stage == 'resume':
                    record = reference['next_step_artifact']
                    require(record['name'] == 'next_step_comparison.pt', 'Unexpected comparison reference path')
                    expected_next = load_owned(args.reference_result.parent / record['name'], record['sha256'])
                    resume_checks['next_step'] = trainer.compare(expected_next, row, prediction); del expected_next
                    require(resume_checks['next_step']['passed'], 'First step after fresh-process resume differs')
            if trainer.step == stop and args.stage != 'split':
                if args.stage == 'uninterrupted':
                    comparison_sha = save_torch(args.out / 'final_comparison.pt', dict(
                        comparison=trainer.comparison(row, prediction),
                        next_step_artifact={'name': next_path.name, 'sha256': next_sha}))
                else:
                    resume_checks['final'] = trainer.compare(reference['comparison'], row, prediction)
                    require(resume_checks['final']['passed'], 'Final resumed trajectory differs')
            update(status='split_completed' if args.stage == 'split' and trainer.step == stop else 'training',
                   completed_updates=trainer.step, last_step=row)
        if args.stage != 'split':
            require(all(trainer.attention_gradient_seen.values()), 'K8 attention training gradients were never established')
            final_fit = trainer.evaluate_training_cases()
            sensitivity = trainer.text_diagnostics()
            trainer.clear_caches()
            final_checkpoint_sha = save_torch(args.out / 'checkpoint.pt', trainer.snapshot())
            update(status='resume_passed' if args.stage == 'resume' else 'uninterrupted_completed',
                completed_updates=trainer.step, checkpoint_sha256=final_checkpoint_sha,
                **({'final_comparison_sha256': comparison_sha} if args.stage == 'uninterrupted' else {'resume_checks': resume_checks}),
                initial_train_fit=trainer.initial_metrics, final_train_fit=final_fit,
                loss_improvement_diagnostic=trainer.initial_metrics['mean_loss'] - final_fit['mean_loss'],
                text_diagnostics=sensitivity, costs=trainer.costs, model_costs=backend.model.costs,
                attention_gradient_seen=trainer.attention_gradient_seen,
                frozen_reader_parameter_versions_unchanged=True,
                frozen_reader_bitwise_bytes_reread=False,
                peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0)
        # Re-check frozen source/input pins. Never rewrite running source or data.
        for path, digest in protocol['file_sha256'].items(): require(sha(path) == digest, 'Pinned file changed during stage: ' + path)
        update(frozen_source_and_input_hashes_unchanged=True)
    except BaseException as exc:
        update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        if backend is not None: backend.model.close()
        signal.alarm(0)


if __name__ == '__main__': main()
