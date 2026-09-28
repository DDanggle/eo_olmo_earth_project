"""CPU-only fresh-process test driver; production worker never spawns children.

Uses the actual matching head with tiny synthetic encoder/reader and dropout.
No actual EO/Qwen/data or GPU. Existing local cached packages need no install.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
CACHE_CANDIDATES = [
    '/Users/dongdong/.cache/uv/archive-v0/-9vyqq1ua9hgPPi7O2w9x',
    '/Users/dongdong/.cache/uv/archive-v0/aRjw5OGy0GtnccMLFJjlG',
    '/Users/dongdong/.cache/uv/archive-v0/XB7_0fCA2kRkHeBss8f87',
    '/Users/dongdong/.cache/uv/archive-v0/VzX04Ed3UZypQ22z8pu3i',
    '/Users/dongdong/.cache/uv/archive-v0/JhU7Pey1FuSgjsbHNHXOa',
    '/Users/dongdong/.cache/uv/archive-v0/muz7oRtV7l16pQXSuJp0l',
    '/Users/dongdong/.cache/uv/archive-v0/_Lhpq92BBd8ilFqjvbojY',
    '/Users/dongdong/.cache/uv/archive-v0/aYsdlVFWzFpmhjeMpNu3C',
    '/Users/dongdong/.cache/uv/archive-v0/VQehQ7W0llP6ht0ypN6pD',
    '/Users/dongdong/.cache/uv/archive-v0/J4XIwSDJ6ejkm4HlkElvm',
]
CACHE = [p for p in CACHE_CANDIDATES if sys.platform == 'darwin' and Path(p).is_dir()]
sys.path[:0] = CACHE
import torch


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    matching = Path(os.environ.get('OE11_MATCHING_CODE',
        '/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code/oe11_matching_v0')).resolve()
    requested_output = os.environ.get('OE11_TEST_OUTPUT_DIRECTORY')
    output_directory = (Path(requested_output).resolve() if requested_output else
                        Path(tempfile.mkdtemp(prefix='oe11_cpu_test_output_')))
    output_directory.mkdir(parents=True, exist_ok=True)
    base = Path(tempfile.mkdtemp(prefix='run_', dir=output_directory))
    cases_path = base / 'cases.json'
    ids = ['alpha:k1', 'alpha:k8', 'beta:k1', 'beta:k8']
    cases = dict(split='train', seed=290929, total_steps=4, split_step=2,
                 episode_ids=ids, order=ids, fit_episode_ids=ids[::2],
                 query_positions=[2, 5], support_positions=list(range(8)))
    cases_path.write_text(json.dumps(cases))
    files = [ROOT / name for name in ('worker.py', 'training_core.py', 'tiny_backend.py')]
    files += [matching / 'matching_head.py', cases_path]
    protocol = dict(schema='oe11_training_pilot_v0', backend='tiny_cpu', seed=290929,
                    total_steps=4, split_step=2, max_stage_seconds=120, resume_tolerance=1e-6,
                    allowed_arms=['B0', 'B2'], allowed_conditions=['names_only', 'matched_knowledge'],
                    optimizer=dict(encoder_lr=1e-5, head_lr=1e-3, weight_decay=.01, eps=1e-6, clip_grad_norm=1.),
                    paths=dict(worker=str(ROOT / 'worker.py'), cases=str(cases_path), matching_code=str(matching)),
                    file_sha256={str(p): sha(p) for p in files})
    protocol_path = base / 'protocol.json'; protocol_path.write_text(json.dumps(protocol))
    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = ''
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT)] + CACHE)
    receipts = {}
    for arm, condition in (('B2', 'matched_knowledge'), ('B0', 'names_only')):
        job = base / (arm + '_' + condition); job.mkdir()
        def launch(stage, out, checkpoint=None, checkpoint_sha=None, expect_success=True):
            argv = [sys.executable, str(ROOT / 'worker.py'), '--protocol', str(protocol_path),
                    '--protocol-sha256', sha(protocol_path), '--stage', stage,
                    '--arm', arm, '--condition', condition, '--out', str(out)]
            if stage == 'resume':
                cp = checkpoint or job / 'split/checkpoint.pt'
                reference = job / 'uninterrupted/final_comparison.pt'
                argv += ['--resume-from', str(cp), '--resume-sha256', checkpoint_sha or sha(cp),
                         '--reference-result', str(reference), '--reference-sha256', sha(reference)]
            result = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=150)
            (job / (out.name + '.stdout.txt')).write_text(result.stdout)
            (job / (out.name + '.stderr.txt')).write_text(result.stderr)
            if (result.returncode == 0) != expect_success:
                raise AssertionError(f'{arm}/{stage} return={result.returncode}\n{result.stdout}\n{result.stderr}')
            return json.loads((out / 'receipt.json').read_text())
        uninterrupted = launch('uninterrupted', job / 'uninterrupted')
        split = launch('split', job / 'split')
        resumed = launch('resume', job / 'resume')
        assert len({uninterrupted['pid'], split['pid'], resumed['pid']}) == 3
        assert uninterrupted['status'] == 'uninterrupted_completed'
        assert split['status'] == 'split_completed' and resumed['status'] == 'resume_passed'
        assert all(resumed['immediate_restore_exact'][k]['passed'] for k in ('model', 'optimizer', 'rng'))
        assert all(resumed['resume_checks'][k]['passed'] for k in ('next_step', 'final'))
        assert all(resumed['attention_gradient_seen'].values())
        assert not resumed['actual_GPU_model_execution']
        # A valid but wrong RNG checkpoint must restore exactly as supplied and
        # then fail the independent trajectory comparison, proving RNG matters.
        saved = torch.load(job / 'split/checkpoint.pt', weights_only=True, map_location='cpu')
        saved['rng']['torch_cpu'] = torch.Generator().manual_seed(123).get_state()
        bad_path = job / 'wrong_rng.pt'; torch.save(saved, bad_path)
        bad = launch('resume', job / 'wrong_rng_resume', checkpoint=bad_path, expect_success=False)
        assert 'First step after fresh-process resume differs' in bad['error']
        receipts[arm] = dict(condition=condition, pid_triplet=[uninterrupted['pid'], split['pid'], resumed['pid']],
                             status=resumed['status'], immediate_restore_exact=resumed['immediate_restore_exact'],
                             resume_checks=resumed['resume_checks'],
                             attention_gradient_seen=resumed['attention_gradient_seen'],
                             wrong_rng_resume_rejected=True)
    report = dict(status='passed', scope='tiny synthetic CPU + actual matching head; no actual EO/Qwen/GPU',
                  output_root=str(base), GPU_used=False, actual_model_wiring_verified=False,
                  stages_successful=6, deliberate_wrong_rng_failures=2, receipts=receipts,
                  python=sys.executable, torch_version=str(torch.__version__))
    (output_directory / 'cpu_fresh_process_receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
