import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from audit_p1_outputs import audit, metrics, sha
from episode_loader import EpisodeLoader
from test_episode_loader import fixture, refresh_hashes, write_jsonl


def completed_fixture(root):
    prep,eps=fixture(root)
    ep=json.loads((eps/'episodes_train.jsonl').read_text())
    sc=json.loads((eps/'scoring/scoring_train.jsonl').read_text())
    manifest={r['patch_id']:r for r in map(json.loads,(prep/'manifest.jsonl').read_text().splitlines())}
    second=copy.deepcopy(ep);second.update(episode_id='second:k1',base_id='second',query_patch_id='t1')
    second['query_input_npz']='inputs/t1.npz';second['query_input_sha256']=manifest['t1']['npz_sha256']
    second['initial_observation_ids']=['t1:obs:2','t1:obs:5']
    for i,o in enumerate(second['query_observations']):o['observation_id']=f't1:obs:{i}'
    support=second['support_pairs'][0]['positive'];support.update(patch_id='t0',object_key='t0:1',input_npz='inputs/t0.npz',input_sha256=manifest['t0']['npz_sha256'],mask_npz='support_masks/t0_1.npz',observation_ids=[f't0:obs:{i}' for i in range(8)])
    (eps/'support_masks/t0_1.npz').write_bytes((eps/'support_masks/t1_1.npz').read_bytes())
    s2=copy.deepcopy(sc);s2.update(episode_id='second:k1',base_id='second',query_patch_id='t1',query_label_npz='labels/t1.npz',query_label_sha256=manifest['t1']['label_sha256'],expected_query_label_sha256=manifest['t1']['label_sha256'])
    write_jsonl(eps/'episodes_train.jsonl',[ep,second]);write_jsonl(eps/'scoring/scoring_train.jsonl',[sc,s2]);refresh_hashes(prep,eps)
    selected=sorted([sc,s2],key=lambda r:hashlib.sha256(('oe9-p1-two-case:'+r['episode_id']).encode()).hexdigest())
    run=root/'run';run.mkdir();loader=EpisodeLoader(prep,eps,'train');audits=[];final=[]
    for i,s in enumerate(selected):
        audits.append(loader.load(s['episode_id'])['audit'])
        target=loader.training_target(s['episode_id'],training=True)
        prob=np.where(target['target_mask'],.9,.1).astype(np.float32)
        v,_=metrics(prob,target['target_mask'],target['label_valid'])
        final.append({'iou':v['iou'],'predicted_fraction':v['predicted_fraction'],'target_fraction':v['target_fraction'],'loss':v['approximate_mask_loss_from_saved_probabilities']})
        np.savez_compressed(run/f'train_prediction_{i}.npz',probability=prob,target=target['target_mask'],valid=target['label_valid'])
    (run/'receipt.json').write_text(json.dumps({'status':'synthetic_fixture','query_dates':[2,5],'support_dates':8,'spatial_resolution':[128,128],'head_fit':{'final':final,'all_iou_at_least_0_80':True}}))
    (run/'selected_examples.json').write_text(json.dumps({'rows':selected,'audits':audits}))
    (run/'head_curve.json').write_text(json.dumps([{'step':16,'cases':final}]))
    return run,prep,eps


class AuditTests(unittest.TestCase):
    def test_complete_synthetic_receipt_preview_and_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run,prep,eps=completed_fixture(root)
            report=audit(run,prep,eps,root/'export')
            self.assertEqual(report['status'],'passed_prediction_and_input_audit')
            self.assertEqual([c['metrics']['iou'] for c in report['cases']],[1.,1.])
            self.assertTrue((root/'export/p1_training_qa.png').is_file())
            self.assertTrue((root/'export/export_manifest.json').is_file())
            self.assertFalse(report['checkpoint_torch_loaded'])

    def test_saved_gold_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run,prep,eps=completed_fixture(root)
            path=run/'train_prediction_0.npz'
            with np.load(path) as z: data={k:z[k] for k in z.files}
            data['target'][5,5]=~data['target'][5,5]
            np.savez_compressed(path,**data)
            with self.assertRaisesRegex(ValueError,'gold differs'):
                audit(run,prep,eps,root/'rejected')
            self.assertEqual(json.loads((root/'rejected/independent_audit.json').read_text())['status'],'failed')

    def test_threshold_void_and_nonfinite(self):
        valid=np.ones((128,128),bool);valid[0,0]=False
        target=np.zeros((128,128),bool);target[1,1]=True
        prob=np.full((128,128),.5,np.float32);prob[1,1]=.6;prob[0,0]=1
        m,p=metrics(prob,target,valid)
        self.assertEqual(m['iou'],1);self.assertEqual(p.sum(),1)
        prob[1,2]=np.nan
        with self.assertRaisesRegex(ValueError,'probability range'):metrics(prob,target,valid)


if __name__=='__main__':unittest.main(verbosity=2)
