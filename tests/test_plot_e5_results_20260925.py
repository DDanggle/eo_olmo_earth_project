"""Synthetic schema tests only. No actual E5 outputs or figures are loaded/made."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import plot_e5_results_20260925 as plot

PLAN = Path(__file__).resolve().parents[1] / 'config/e5_equal_budget_prereg_v0.json'


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def fixture(root):
    artifact = root/'synthetic_artifact'
    artifact.mkdir()
    audit_path = root/'synthetic_audit.json'
    primary = {}
    for seed in plot.SEEDS:
        evaluations = {}
        for j, key in enumerate(plot.EVALUATIONS):
            events = {}
            for k in range(8):
                # Hand-specified artificial values; never sent to the renderer.
                events[f'synthetic_event_{k}'] = {'ba': .58 + .02*int(seed) + .012*j + .003*k,
                    'n_pos':55 if k<7 else 60,'n_hard_neg':57 if k<7 else 58}
            evaluations[key] = {'events':events, 'n_events':8,'n_items':902,
                                'macro_ba':float(np.mean([e['ba'] for e in events.values()]))}
        differences = {e:evaluations['pair/native']['events'][e]['ba'] - evaluations['full/native']['events'][e]['ba']
                       for e in sorted(events)}
        v = np.array(list(differences.values()))
        draws = np.random.default_rng(20260925).integers(0,8,(5000,8))
        contrast = {'left':'pair/native','right':'full/native','delta':float(v.mean()),
                    'event_deltas':differences,'ci95_delta':np.quantile(v[draws].mean(1),[.025,.975],method='linear').tolist()}
        primary[seed] = {'evaluations':evaluations,'contrasts':{'pair_minus_full':contrast}}
    scores = {'schema':'e5-equal-budget-scores-v0','valid':True,'verdict':'mixed_or_inconclusive',
              'coverage':{'expected':26325,'received':26325,'n_items':5989,'n_train':4234,'n_test':1755,'n_primary':902},
              'metrics':{s:{'primary_same_prompt':p} for s,p in primary.items()}}
    status = {'status':'completed','n_models':12,'n_rows':26325,'verdict':scores['verdict']}
    write(artifact/'scores.json',scores);write(artifact/'status.json',status)
    (artifact/'prereg.json').write_bytes(PLAN.read_bytes())
    audit = {'schema':'e5-independent-result-audit-v0','consistent':True,'artifact':str(artifact),
             'n_models':12,'n_steps':19080,'n_training_exposures':152424,'n_answers':26325,
             'primary_n':902,'primary_events':8,'verdict':scores['verdict'],
             'primary_metrics':primary,'checkpoint_tensors_loaded_and_checked_on_cpu':False,
             'hashes_verified':{str(artifact/n):plot.sha(artifact/n) for n in ('scores.json','status.json','prereg.json')}}
    write(audit_path,audit)
    return artifact,audit_path


def refresh(artifact, audit_path, sync_primary=True):
    audit=plot.read(audit_path)
    if sync_primary:
        scores=plot.read(artifact/'scores.json')
        audit['primary_metrics']={s:m['primary_same_prompt'] for s,m in scores['metrics'].items()}
    audit['hashes_verified']={str(artifact/n):plot.sha(artifact/n) for n in ('scores.json','status.json','prereg.json')}
    write(audit_path,audit)


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.artifact,self.audit=fixture(Path(self.tmp.name).resolve())
    def tearDown(self):
        self.tmp.cleanup()
    def load(self):
        return plot.load_verified(self.artifact,self.audit)
    def scores_mutate(self,fn,sync=True):
        value=plot.read(self.artifact/'scores.json');fn(value)
        write(self.artifact/'scores.json',value);refresh(self.artifact,self.audit,sync)

    def test_valid_keeps_three_seeds_five_conditions_and_signed_contrast(self):
        value=self.load()
        self.assertEqual(set(value['series']),{'1','2','3'})
        self.assertEqual(set(value['series']['1']),set(plot.EVALUATIONS))
        self.assertLess(value['series']['1']['full/native'],value['series']['3']['full/native'])
        for seed in plot.SEEDS:
            self.assertAlmostEqual(value['primary_contrast'][seed]['delta'],.012)
            self.assertAlmostEqual(value['primary_contrast'][seed]['ci95_delta'][0],.012)
        self.assertEqual((value['n_items'],value['n_events']),(902,8))
        self.assertEqual(list(Path(self.tmp.name).rglob('*.png')),[])
        self.assertEqual(list(Path(self.tmp.name).rglob('*.pdf')),[])

    def test_running_status_rejected_even_when_hash_reaudited(self):
        status=plot.read(self.artifact/'status.json');status['status']='training'
        write(self.artifact/'status.json',status);refresh(self.artifact,self.audit)
        with self.assertRaisesRegex(ValueError,'not completed'):self.load()

    def test_failed_audit_invalid_score_or_failure_marker_rejected(self):
        audit=plot.read(self.audit);audit['consistent']=False;write(self.audit,audit)
        with self.assertRaisesRegex(ValueError,'audit did not pass'):self.load()
        audit['consistent']=True;write(self.audit,audit)
        self.scores_mutate(lambda s:s.update(valid=False))
        with self.assertRaisesRegex(ValueError,'Scores are invalid'):self.load()
        self.scores_mutate(lambda s:s.update(valid=True))
        (self.artifact/'failure.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'failed run'):self.load()

    def test_stale_hash_and_substituted_audited_root_rejected(self):
        p=self.artifact/'scores.json';p.write_text(p.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.load()
        refresh(self.artifact,self.audit)
        a=plot.read(self.audit);a['artifact']=str(self.artifact.parent/'other');write(self.audit,a)
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.load()

    def relocate_audit(self):
        audit=plot.read(self.audit)
        remote='/home/work/data/olmoearth/e5_equal_budget_v0'
        audit['artifact']=remote
        audit['hashes_verified']={remote+'/'+name:plot.sha(self.artifact/name)
            for name in ('scores.json','status.json','prereg.json')}
        write(self.audit,audit)
        return remote

    def test_exact_remote_root_anchored_relocation_and_provenance(self):
        remote=self.relocate_audit()
        result=self.load()
        self.assertEqual(result['audit_original_artifact_root'],remote)
        self.assertEqual(result['local_render_artifact_root'],str(self.artifact))
        self.assertEqual(set(result['audited_input_lookup_hashes']),
            {remote+'/'+n for n in ('scores.json','status.json','prereg.json')})
        self.assertIn('does not rerun the full audit',result['verification_scope'])
        self.assertIn(str(self.artifact/'scores.json'),result['source_hashes'])

    def test_relocation_never_falls_back_to_basename_or_other_root(self):
        remote=self.relocate_audit()
        for use_basename in (True,False):
            audit=plot.read(self.audit)
            audit['hashes_verified']={(name if use_basename else '/wrong/root/'+name):plot.sha(self.artifact/name)
                for name in ('scores.json','status.json','prereg.json')}
            write(self.audit,audit)
            with self.subTest(basename=use_basename),self.assertRaisesRegex(ValueError,'hash mismatch'):self.load()

    def test_relocated_local_scores_or_status_tampering_rejected(self):
        self.relocate_audit()
        for name in ('scores.json','status.json'):
            path=self.artifact/name
            original=path.read_bytes()
            path.write_bytes(original+b' ')
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'hash mismatch'):self.load()
            path.write_bytes(original)
        self.load()

    def test_noncanonical_audited_root_rejected(self):
        self.relocate_audit()
        audit=plot.read(self.audit)
        for root in ('relative/root','/home/work/../data/run','/home/work/data/run/'):
            audit['artifact']=root;write(self.audit,audit)
            with self.subTest(root=root),self.assertRaisesRegex(ValueError,'canonical absolute'):self.load()

    def test_coverage_and_dropped_seed_rejected(self):
        self.scores_mutate(lambda s:s['coverage'].update(n_primary=901))
        with self.assertRaisesRegex(ValueError,'coverage'):self.load()
        self.scores_mutate(lambda s:s['coverage'].update(n_primary=902))
        self.scores_mutate(lambda s:s['metrics'].pop('3'))
        with self.assertRaisesRegex(ValueError,'Seed support'):self.load()

    def test_dropped_condition_or_event_rejected(self):
        self.scores_mutate(lambda s:s['metrics']['1']['primary_same_prompt']['evaluations'].pop('delta/native'))
        with self.assertRaisesRegex(ValueError,'condition support'):self.load()

    def test_no_seven_event_or_pooled_mean_substitution(self):
        self.scores_mutate(lambda s:s['metrics']['1']['primary_same_prompt']['evaluations']['full/native']['events'].pop('synthetic_event_7'))
        with self.assertRaisesRegex(ValueError,'Primary support'):self.load()

    def test_nan_and_bool_score_rejected(self):
        self.scores_mutate(lambda s:s['metrics']['1']['primary_same_prompt']['evaluations']['full/native'].update(macro_ba=True))
        with self.assertRaises(ValueError):self.load()
        self.scores_mutate(lambda s:s['metrics']['1']['primary_same_prompt']['evaluations']['full/native'].update(macro_ba=float('nan')),sync=False)
        with self.assertRaisesRegex(ValueError,'Nonfinite JSON'):self.load()

    def test_audit_metric_disagreement_rejected(self):
        self.scores_mutate(lambda s:s['metrics']['1']['primary_same_prompt']['evaluations']['full/native'].update(macro_ba=.99),sync=False)
        with self.assertRaisesRegex(ValueError,'Audit numeric'):self.load()

    def test_mean_and_event_bootstrap_must_match_when_audit_copy_also_changed(self):
        self.scores_mutate(lambda s:s['metrics']['1']['primary_same_prompt']['evaluations']['full/native'].update(macro_ba=.99))
        with self.assertRaisesRegex(ValueError,'Equal-event mean'):self.load()

    def test_wrong_contrast_or_ci_rejected(self):
        self.scores_mutate(lambda s:s['metrics']['1']['primary_same_prompt']['contrasts']['pair_minus_full'].update(ci95_delta=[-.1,.1]))
        with self.assertRaisesRegex(ValueError,'Audit numeric'):self.load()

    def test_scientific_plan_change_rejected(self):
        plan=plot.read(self.artifact/'prereg.json');plan['primary_analysis']['bootstrap']['draws']=500
        write(self.artifact/'prereg.json',plan);refresh(self.artifact,self.audit)
        with self.assertRaisesRegex(ValueError,'scientific plan'):self.load()


if __name__=='__main__':
    unittest.main()
