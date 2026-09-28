"""Read local official sources and write a static-only evaluation inventory."""
import argparse
import ast
import hashlib
import json
import shlex
from pathlib import Path

ROOT = Path('/Users/dongdong/DongDong/ai_projects/earth_paper/olmoearth/learning_sources/olmoearth_pretrain')
OUT = Path('/private/tmp/oe3_capability_program_20260927')

def source(relative):
    path = ROOT / relative
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

argv = [
    '--cluster=local',
    '--checkpoint_path=/REPLACE_WITH_VERIFIED_DISTRIBUTED_CHECKPOINT',
    '--module_path=scripts/official/v1/tiny.py',
    '--model_name=oe3_v1tiny_original_dense',
    '--project_name=oe3_capability_program',
    '--task-names=sen1floods11,m_cashew_plant',
    '--defaults_only', '--select_best_val', '--dry_run',
    '--trainer.callbacks.wandb.enabled=False',
    '--trainer.callbacks.downstream_evaluator.run_on_test=False',
]
# Verify the exact public parser, without importing the training dependency tree.
tree = ast.parse((ROOT / 'olmoearth_pretrain/internal/full_eval_sweep.py').read_text())
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
parser_statements = []
for node in main.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Tuple) for t in node.targets):
        break
    parser_statements.append(node)
ns = {'argparse': argparse, '_parse_model_arg': str}
exec(compile(ast.fix_missing_locations(ast.Module(body=parser_statements, type_ignores=[])), '<official-parser-AST>', 'exec'), ns)
parsed, extra = ns['parser'].parse_known_args(argv)

common = {
    'task_type': 'segmentation', 'eval_mode': 'linear_probe', 'primary_metric': 'miou',
    'patch_size': 4, 'pooling': 'mean', 'preserve_spatial_grid': True,
    'epochs': 50, 'probe_lr': 0.1, 'probe_type': 'linear',
    'head': 'nn.Linear(D, num_classes * patch_size**2); rearrange to dense logits',
    'loss': 'cross_entropy', 'ignore_label': -1,
    'label_fraction': 1.0, 'encoder_gradient': False,
    'select_best_val': True, 'linear_probe_eval_interval_with_select_best_val': 5,
    'run_on_test_for_development': False,
}
tasks = [
    dict(common, task_id='sen1floods11', dataset_id='sen1floods11',
         modality='sentinel1', num_classes=2, input_hw=64,
         embedding_batch_size=128, probe_batch_size=128, num_workers=4,
         dataset_env='FLOODS_DIR',
         required_layout=['flood_train_data.pt', 'flood_valid_data.pt', 'flood_test_data.pt'],
         tensor_contract={'s1': '[N,2,64,64]', 'labels': '[N,1,64,64]'},
         registry_norm_stats_from_pretrained=True,
         selected_sweep_norm_stats_from_pretrained=True,
         normalization='Official Normalizer(COMPUTED): (x - mean)/(4*std) + 0.5; no clipping',
         caveats=['The implemented loader supports S1 only, even though preprocessing can store S2.',
                  'The prepared 64x64 benchmark filters tiles with no positive labels; preserve official prepared data for comparability.',
                  'Loader removes nonfinite S1 tiles. Report actual post-filter split counts.',
                  'Loader uses a fixed placeholder acquisition date; do not claim temporal transfer.']),
    dict(common, task_id='m_cashew_plant', dataset_id='m-cashew-plant',
         modality='sentinel2_l2a', num_classes=7, input_hw=256,
         embedding_batch_size=32, probe_batch_size=8, num_workers=2,
         dataset_env='GEOBENCH_DIR',
         required_layout=['segmentation_v1.0/m-cashew-plant/ with GeoBench task specification, samples and official partitions'],
         splits=['train', 'valid', 'test'],
         registry_norm_stats_from_pretrained=False,
         selected_sweep_norm_stats_from_pretrained=True,
         normalization='The full_eval_sweep --defaults_only path forces official pretrained COMPUTED statistics for every task; this differs from the cashew registry setting.',
         registry_normalization='Dataset band_stats, NORM_NO_CLIP_2_STD',
         caveats=['GeoBench wrapper includes a B11-to-B10 missing-band imputation contract. Preserve its official input mapping.',
                  'Do not say registry defaults and full_eval_sweep defaults are identical.',
                  'Loader uses a fixed placeholder acquisition date; do not claim temporal transfer.']),
]

inventory = {
 'schema': 'oe3-official-eval-inventory-v0', 'date': '2026-09-27',
 'status': 'source_and_parser_verified_not_runtime_ready',
 'official_repo': str(ROOT), 'official_commit': '0497dfbb6711ded4e6bf10cf089fc1e4d58c186b',
 'development_model': {
    'id': 'allenai/OlmoEarth-v1-Tiny',
    'historical_server_hf_path': '/home/work/data/olmoearth/oe1_bentxt_v0/models/OlmoEarth-v1-Tiny',
    'historical_server_path_rechecked_this_task': False,
    'format': 'config.json + weights.pth',
    'official_config_module': 'scripts/official/v1/tiny.py',
    'public_weights_loader': 'olmoearth_pretrain.model_loader.load_model_from_path',
    'sweep_checkpoint_loader': '--checkpoint_path -> --trainer.load_path (distributed checkpoint)',
    'format_compatibility_status': 'No verified direct HF-bundle-to-trainer.load_path path; do not substitute the HF folder into the distributed-checkpoint template.',
    'next_bridge_requirement': 'Implement a small model-config adapter calling the official local-path loader, or export a verified distributed checkpoint, then confirm encoder tensor hashes and fixed-input output parity.',
    'version_scope': 'Development baseline only. The local source also contains v1.1/v1.2; latest-model and Base-scale runs remain follow-up work.'
 },
 'tasks': tasks,
 'command_templates': {
    'cwd': str(ROOT),
    'data_environment': {'FLOODS_DIR': '/REPLACE_WITH_PREPARED_FLOODS_ROOT', 'GEOBENCH_DIR': '/REPLACE_WITH_GEOBENCH_ROOT'},
    'distributed_checkpoint_sweep_dry_run_argv': ['python','-m','olmoearth_pretrain.internal.full_eval_sweep'] + argv,
    'distributed_checkpoint_sweep_dry_run_shell': 'env -u PYTHONPATH python -m olmoearth_pretrain.internal.full_eval_sweep ' + shlex.join(argv),
    'notes': ['Contains explicit unresolved path placeholders; not a launch-ready command.',
              '--dry_run in this source launches a python subprocess with dry_run_evaluate to build/print configuration; it is not a dependency-free print-only path.',
              '--defaults_only leaves registry probe LR=0.1 and epochs=50 intact, but overrides all tasks to pretrained normalization.',
              'all_evals.py defaults run_on_test=True; the supplied development command explicitly overrides it to False.',
              'Do not remove --dry_run until checkpoint loading, data layout, dependencies, resource/output paths and resolved config are verified.']
 },
 'verification': {
    'public_argparse_ast_extracted_and_parsed': True,
    'parsed_public_arguments': vars(parsed), 'known_passthrough_overrides': extra,
    'full_module_import_or_official_dry_run_executed': False,
    'local_python_checked': '/Users/dongdong/DongDong/ai_projects/earth_paper/olmoearth/.venv-nano-lab/bin/python',
    'local_import_spec_presence': {'torch': True, 'olmo_core': False, 'geobench': False, 'upath': True, 'omegaconf': False, 'rslearn': False, 'h5py': False},
    'server_checked': False, 'datasets_downloaded': False, 'model_evaluated': False,
 },
 'retained_development_reference': 'Existing BEN S2 questions/20-scene diagnostic remain historical debugging evidence; they are not the independent EO benchmark or a substitute for original-objective map targets.',
 'fair_continued_pretraining_requirement': 'Use the same fixed checkpoint, examples, sensor inputs, map targets, exposure budget and validation protocol. Original v1 includes derived-map targets and patch discrimination + InfoNCE; S2 BEN alone does not reproduce it.',
 'sources': [source(p) for p in [
    'scripts/official/v1/tiny.py', 'scripts/official/v1/script.py',
    'olmoearth_pretrain/model_loader.py', 'olmoearth_pretrain/internal/full_eval_sweep.py',
    'olmoearth_pretrain/internal/all_evals.py', 'olmoearth_pretrain/internal/experiment.py',
    'olmoearth_pretrain/evals/eval_wrapper.py', 'olmoearth_pretrain/evals/embeddings.py',
    'olmoearth_pretrain/evals/linear_probe.py', 'olmoearth_pretrain/evals/datasets/paths.py',
    'olmoearth_pretrain/evals/datasets/configs.py', 'olmoearth_pretrain/evals/datasets/floods_dataset.py',
    'olmoearth_pretrain/evals/datasets/geobench_dataset.py', 'olmoearth_pretrain/train/callbacks/evaluator_callback.py',
    'olmoearth_pretrain/data/normalize.py', 'pyproject.toml',
 ]]
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT/'official_eval_inventory.json').write_text(json.dumps(inventory, indent=2, ensure_ascii=False)+'\n')
print(json.dumps({'output':str(OUT/'official_eval_inventory.json'),'tasks':[t['task_id'] for t in tasks], 'parser_verified':True,'runtime_verified':False}))
