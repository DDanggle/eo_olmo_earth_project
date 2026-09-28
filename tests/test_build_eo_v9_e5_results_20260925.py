"""Synthetic immutable E5-card builder tests. Never open actual E5 outputs."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

# Support repo/code/tests installation and standalone /private/tmp execution.
_HERE = Path(__file__).resolve().parent
for _candidate in (_HERE, _HERE.parent):
    if (_candidate / 'build_eo_v9_e5_results_20260925.py').is_file():
        sys.path.insert(0, str(_candidate))
        break
import build_eo_v9_e5_results_20260925 as builder


class Fixture:
    def __init__(self, root, verdict='mixed_or_inconclusive'):
        root = root.resolve()
        self.root = root
        self.base = root / 'v8'; self.base.mkdir()
        self.e5 = root / 'replica'; self.e5.mkdir()
        self.out = root / 'v9'
        self.audit_path = root / 'audit.json'
        self.original = Path('/synthetic/original/e5_equal_budget_v0')
        self.other = {'id': 'E4-D', 'status': 'complete', 'findings': ['Historical synthetic finding'],
                      'limitations': ['Preserve me'], 'next_step': 'Frozen'}
        self.pending = {'id': 'E5-EB', 'status': '학습 진행 스냅샷 · 결과 없음',
                        'findings': ['As of old snapshot'], 'limitations': ['No results'], 'next_step': 'Wait'}
        runs = {'schema_version': 'eo_research_snapshot_v1', 'checked_at': '2026-09-25T06:14:07Z',
                'runs': [self.other, self.pending]}
        files = {'research_runs.json': runs, 'research_sources.json': {'checked_at': '2026-09-25T06:14:07Z',
                    'source_dates_v8': {'historical': True, 'source_sha256': 'a' * 64}},
                 'reader_cases.json': {'frozen_synthetic_answers': 'never modify'},
                 'catalog.json': {'source_dates': 'keep these dates'}, 'meta.json': {'count': 2}}
        for name, value in files.items():
            builder.write(self.base / name, value)
        (self.base / 'index.html').write_text('<p>Unchanged existing UI</p>\n')
        (self.base / 'previews').mkdir()
        (self.base / 'previews' / 'case.png').write_bytes(b'\x89PNG\r\n\x1a\nSYNTHETIC')
        v8 = {'schema': 'eo-v8-source-dates-build-v0',
              'output_files_sha256': {str(p.relative_to(self.base)): builder.sha(p)
                                     for p in self.base.rglob('*') if p.is_file()}}
        builder.write(self.base / 'v8_build_manifest.json', v8)
        for name in builder.V8_EXTRA - {'v8_build_manifest.json'}:
            builder.write(self.base / name, {'scope': 'old v8 validation only'})
        self.base_hashes = {str(p.relative_to(self.base)): builder.sha(p) for p in self.base.rglob('*') if p.is_file()}
        plan = {'synthetic': 'not the real E5 plan', 'root': str(self.original.parent), 'output_directory': self.original.name}
        builder.write(self.e5 / 'prereg.json', plan)
        self.plan_sha = builder.sha(self.e5 / 'prereg.json')
        manifest = {'schema': 'e5-equal-budget-prepared-v0', 'files_sha256': {'prereg.json': self.plan_sha},
                    'code_snapshot_sha256': {'synthetic_train.py': '1' * 64}}
        builder.write(self.e5 / 'manifest.json', manifest)
        self.parent_sha = builder.sha(self.e5 / 'manifest.json')
        models = [{'seed': seed, 'model_arm': arm, 'updates': 1590, 'exposures': 12702}
                  for seed in (1, 2, 3) for arm in ('full', 'pair', 'later', 'delta')]
        self.generated = {'predictions.jsonl': '2' * 64, 'prompts.jsonl': '3' * 64, 'runtime_environment.json': '4' * 64}
        linkage = {'predictions_sha256': self.generated['predictions.jsonl'], 'prompts_sha256': self.generated['prompts.jsonl'],
                   'runtime_sha256': self.generated['runtime_environment.json']}
        summary = {'schema': 'e5-training-summary-v0', 'n_models': 12, 'updates_per_model': 1590,
                   'exposures_per_model': 12702, 'models': models, 'manifest_sha256': self.parent_sha,
                   'code_snapshot_sha256': manifest['code_snapshot_sha256'], **linkage}
        inference = {'schema': 'e5-training-inference-completed-v0', 'n_models': 12, 'n_rows': 26325,
                     'n_prompts': 5989, 'models': models, **linkage}
        status = {'status': 'completed', 'n_models': 12, 'n_rows': 26325, 'verdict': verdict,
                  'at': '2026-09-25T07:30:00Z'}
        primary = {}
        for seed in ('1', '2', '3'):
            evaluations = {}
            for j, arm in enumerate(builder.EVALUATIONS):
                value = .61 + .025 * j + .002 * int(seed)
                evaluations[arm] = {'n_events': 8, 'n_items': 902, 'macro_ba': value,
                                    'events': {f'event{i}': {'ba': value} for i in range(8)}}
            primary[seed] = {'evaluations': evaluations, 'contrasts': {'pair_minus_full': {
                'left': 'pair/native', 'right': 'full/native', 'delta': .025,
                'ci95_delta': [-.01234, .06987], 'event_deltas': {f'event{i}': .025 for i in range(8)}}}}
        decisions = {s: {'synthetic_registered_rule': False} for s in ('1', '2', '3')}
        for name, value in [('training_summary.json', summary), ('inference_completed.json', inference), ('status.json', status)]:
            builder.write(self.e5 / name, value)
        scores = {'schema': 'e5-equal-budget-scores-v0', 'valid': True, 'verdict': verdict,
                  'coverage': {'expected': 26325, 'received': 26325, 'n_items': 5989, 'n_train': 4234,
                               'n_test': 1755, 'n_primary': 902},
                  'manifest_sha256': self.parent_sha, 'training_summary_sha256': builder.sha(self.e5 / 'training_summary.json'),
                  'metrics': {s: {'primary_same_prompt': p} for s, p in primary.items()}, 'seed_decisions': decisions, **linkage}
        builder.write(self.e5 / 'scores.json', scores)
        self.audit = {'schema': 'e5-independent-result-audit-v0', 'consistent': True,
                      'checkpoint_tensors_loaded_and_checked_on_cpu': True, 'audit_code_sha256': builder.AUDITOR_SHA,
                      'n_models': 12, 'n_steps': 19080, 'n_training_exposures': 152424, 'n_answers': 26325,
                      'primary_n': 902, 'primary_events': 8, 'checked_at': '2026-09-25T07:40:00Z',
                      'artifact': str(self.original), 'verdict': verdict, 'primary_metrics': copy.deepcopy(primary),
                      'seed_decisions': copy.deepcopy(decisions),
                      'hashes_verified': {str(self.original / n): builder.sha(self.e5 / n) for n in builder.REQUIRED_NAMES}}
        self.audit['hashes_verified'].update({str(self.original / name): digest for name, digest in self.generated.items()})
        self.save_audit()

    def save_audit(self):
        builder.write(self.audit_path, self.audit)

    def mutate_replica(self, name, mutate, repin=True):
        value = builder.read(self.e5 / name); mutate(value); builder.write(self.e5 / name, value)
        if repin:
            self.audit['hashes_verified'][str(self.original / name)] = builder.sha(self.e5 / name)
            self.save_audit()

    def build(self):
        with mock.patch.multiple(builder, V8_SHA=builder.sha(self.base / 'v8_build_manifest.json'),
                                 PLAN_SHA=self.plan_sha, PARENT_SHA=self.parent_sha):
            return builder.build(self.base, self.e5, self.audit_path, self.out)


class V9BuilderTests(unittest.TestCase):
    def fixture(self):
        context = tempfile.TemporaryDirectory(); self.addCleanup(context.cleanup)
        return Fixture(Path(context.name))

    def test_happy_path_preserves_base_bytes_and_other_runs(self):
        f = self.fixture(); report = f.build()
        self.assertEqual(report['changed_existing_files'], ['research_runs.json', 'research_sources.json'])
        for name, digest in f.base_hashes.items():
            self.assertEqual(builder.sha(f.base / name), digest)
            if name not in builder.CHANGED:
                self.assertEqual(builder.sha(f.out / name), digest)
        runs = builder.read(f.out / 'research_runs.json')['runs']
        self.assertEqual(runs[0], f.other)
        self.assertEqual(builder.read(f.out / 'e5_results_v9' / 'prior_E5_card.json'), f.pending)
        sources = builder.read(f.out / 'research_sources.json')
        self.assertEqual(sources['source_dates_v8'], builder.read(f.base / 'research_sources.json')['source_dates_v8'])
        self.assertFalse(sources['e5_results_v9']['individual_reader_cases_modified'])
        self.assertEqual(len(runs[1]['metrics']['rows']), 7)
        self.assertEqual(runs[1]['metrics']['rows'][5][1:], [.025] * 3)
        self.assertEqual(runs[1]['metrics']['rows'][6][1:], ['[-0.0123, 0.0699]'] * 3)
        # Large files deliberately do not exist; builder consumes only audited small JSON replicas.
        self.assertFalse((f.e5 / 'predictions.jsonl').exists())
        for name in builder.REQUIRED_NAMES:
            self.assertEqual(builder.sha(f.out / 'e5_results_v9' / name), builder.sha(f.e5 / name))
        self.assertNotIn('E6', json.dumps(runs, ensure_ascii=False))

    def test_existing_read_research_api_accepts_synthetic_card(self):
        f = self.fixture(); f.build()
        candidates = [_HERE.parent, _HERE, _HERE.parent / 'code',
                      Path('/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project/code')]
        source_dir = next(p for p in candidates if all((p / n).is_file()
                          for n in ('eo_evidence_search_v0.py', 'eo_query_core_v0.py')))
        sys.path.insert(0, str(source_dir)); self.addCleanup(lambda: sys.path.remove(str(source_dir)))
        spec = importlib.util.spec_from_file_location('eo_v9_test_api', source_dir / 'eo_evidence_search_v0.py')
        api = importlib.util.module_from_spec(spec); spec.loader.exec_module(api)
        result, status = api.read_research_snapshot(f.out)
        self.assertEqual(status, 200); self.assertIs(result['available'], True)
        self.assertEqual(result['runs'][0], f.other)
        metrics = result['runs'][1]['metrics']
        self.assertEqual(len(metrics['columns']), 4); self.assertEqual(len(metrics['rows']), 7)
        self.assertTrue(all(len(row) == 4 for row in metrics['rows']))

    def test_all_registered_verdicts_are_rendered_without_selection(self):
        for verdict in sorted(builder.VERDICTS):
            with self.subTest(verdict=verdict), tempfile.TemporaryDirectory() as temp:
                f = Fixture(Path(temp), verdict); report = f.build()
                self.assertEqual(report['registered_verdict'], verdict)
                card = builder.read(f.out / 'research_runs.json')['runs'][1]
                self.assertIn('등록 판정: ' + verdict, card['findings'])
                self.assertEqual(len(card['metrics']['rows']), 7)

    def test_cpu_audit_false_or_incomplete_is_rejected_before_copy(self):
        for key, value in [('checkpoint_tensors_loaded_and_checked_on_cpu', False), ('consistent', False), ('n_steps', 19079)]:
            with self.subTest(key=key):
                f = self.fixture(); f.audit[key] = value; f.save_audit()
                with self.assertRaises(ValueError):
                    f.build()
                self.assertFalse(f.out.exists())

    def test_raw_replica_tamper_rejected(self):
        f = self.fixture()
        f.mutate_replica('scores.json', lambda x: x.update(valid=False), repin=False)
        with self.assertRaisesRegex(ValueError, 'Source hash mismatch'):
            f.build()
        self.assertFalse(f.out.exists())

    def test_self_consistent_replica_hash_cannot_override_independent_metrics(self):
        f = self.fixture()
        def mutate(scores):
            scores['metrics']['1']['primary_same_prompt']['evaluations']['pair/native']['macro_ba'] = .99
        f.mutate_replica('scores.json', mutate)
        with self.assertRaisesRegex(ValueError, 'independently recalculated primary metrics differ'):
            f.build()
        self.assertFalse(f.out.exists())

    def test_generated_file_lineage_mismatch_rejected(self):
        f = self.fixture()
        f.mutate_replica('inference_completed.json', lambda x: x.update(predictions_sha256='b' * 64))
        with self.assertRaisesRegex(ValueError, 'generated-file identity differs'):
            f.build()

    def test_failure_and_unregistered_verdict_rejected(self):
        f = self.fixture(); builder.write(f.e5 / 'failure.json', {'error': 'synthetic'})
        with self.assertRaisesRegex(ValueError, 'failure artifact'):
            f.build()
        (f.e5 / 'failure.json').unlink()
        f.mutate_replica('scores.json', lambda x: x.update(verdict='selected_best_seed'))
        with self.assertRaisesRegex(ValueError, 'verdict disagreement'):
            f.build()

    def test_source_artifact_path_and_seed_identity_rejected(self):
        f = self.fixture(); f.audit['artifact'] = 'relative/e5_equal_budget_v0'; f.save_audit()
        with self.assertRaisesRegex(ValueError, 'original E5 artifact path'):
            f.build()
        f = self.fixture()
        old = f.original
        other = Path('/synthetic/other/e5_equal_budget_v0')
        f.audit['artifact'] = str(other)
        f.audit['hashes_verified'] = {str(other / Path(k).name): v for k, v in f.audit['hashes_verified'].items()}
        f.save_audit()
        with self.assertRaisesRegex(ValueError, 'differs from pinned plan'):
            f.build()
        f = self.fixture()
        f.mutate_replica('inference_completed.json', lambda x: x['models'][0].update(seed=3))
        with self.assertRaisesRegex(ValueError, 'Final model records differ'):
            f.build()

    def test_base_tamper_unknown_file_and_overwrite_rejected(self):
        f = self.fixture(); (f.base / 'reader_cases.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Source hash mismatch'):
            f.build()
        f = self.fixture(); (f.base / 'unexpected.bin').write_bytes(b'x')
        with self.assertRaisesRegex(ValueError, 'Unexpected base snapshot files'):
            f.build()
        f = self.fixture(); f.build(); before = builder.sha(f.out / 'v9_build_manifest.json')
        with self.assertRaisesRegex(ValueError, 'no overwrite'):
            f.build()
        self.assertEqual(builder.sha(f.out / 'v9_build_manifest.json'), before)

    def test_mid_copy_mutation_preserves_invalid_partial_artifact(self):
        f = self.fixture(); original_copyfile = builder.shutil.copyfile
        def tamper_after_copy(source, destination, **kwargs):
            result = original_copyfile(source, destination, **kwargs)
            if Path(source) == f.e5 / 'scores.json':
                Path(destination).write_text('{"tampered":true}')
            return result
        with mock.patch.object(builder.shutil, 'copyfile', side_effect=tamper_after_copy):
            with self.assertRaisesRegex(ValueError, 'Result copy changed'):
                f.build()
        self.assertTrue(f.out.exists())
        self.assertTrue(builder.read(f.out / 'v9_failure.json')['partial_copy_not_valid'])
        self.assertFalse((f.out / 'v9_build_manifest.json').exists())
        for name, digest in f.base_hashes.items():
            self.assertEqual(builder.sha(f.base / name), digest)


if __name__ == '__main__':
    unittest.main(verbosity=2)
