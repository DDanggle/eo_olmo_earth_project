#!/usr/bin/env python3
"""Independent CPU P1 prediction audit/preview. No torch import or checkpoint load."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw


def check(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def jsonl(path, key):
    out = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        check(row[key] not in out, 'duplicate row: ' + key)
        out[row[key]] = row
    return out


def path_in(root, relative, category):
    p = Path(relative)
    check(not p.is_absolute() and p.parts[0] == category and '..' not in p.parts, 'invalid category path')
    result = (root / p).resolve()
    check(result.is_relative_to(root.resolve() / category), 'path escapes root')
    return result


def metrics(prob, target, valid):
    check(prob.shape == target.shape == valid.shape == (128, 128), 'prediction shape')
    check(np.issubdtype(prob.dtype, np.floating) and target.dtype == valid.dtype == np.bool_, 'prediction dtype')
    check(np.isfinite(prob).all() and (prob >= 0).all() and (prob <= 1).all(), 'probability range')
    check(valid.any(), 'no valid pixels')
    pred = (prob > .5) & valid
    truth = target & valid
    union = int((pred | truth).sum())
    intersection = int((pred & truth).sum())
    p, y = prob[valid].astype(np.float64), truth[valid].astype(np.float64)
    clip = np.clip(p, 1e-7, 1-1e-7)
    bce = float(-(y*np.log(clip) + (1-y)*np.log1p(-clip)).mean())
    dice_loss = float(1 - (2*(p*y).sum()+1)/(p.sum()+y.sum()+1))
    return {'iou': intersection / max(union, 1), 'intersection': intersection, 'union': union,
            'predicted_fraction': float(pred.sum()/valid.sum()),
            'target_fraction': float(truth.sum()/valid.sum()),
            'bce_from_saved_probabilities_clipped_1e_7': bce,
            'dice_loss_from_saved_probabilities': dice_loss,
            'approximate_mask_loss_from_saved_probabilities': bce+dice_loss,
            'saturated_probability_pixels': int(((p == 0) | (p == 1)).sum()),
            'valid_pixels': int(valid.sum()), 'threshold': 'probability > 0.5'}, pred


def figure(cases, out):
    # Fixed radiometric display for all cases/dates; no label-dependent contrast.
    width, cell, gap = 1140, 256, 18
    canvas = Image.new('RGB', (width, 720), '#ffffff')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), 'P1: two TRAINING cases, frozen-feature head fit (before joint encoder/VLM steps)', fill='black')
    draw.text((20, 29), 'Engineering QA only. Sentinel-2 B04/B03/B02, fixed reflectance display raw / 3000; no generalization evidence.', fill='black')
    for row, case in enumerate(cases):
        y = 66 + row*300
        titles = [f"Query position 2: {case['dates'][2]}", f"Query position 5: {case['dates'][5]}",
                  'Gold target mask', f"Prediction errors: IoU {case['metrics']['iou']:.4f}"]
        imgs = []
        for pos in (2,5):
            rgb = np.clip(case['raw'][pos, [2,1,0]].transpose(1,2,0).astype(np.float32)/3000, 0, 1)
            rgb = np.round(rgb*255).astype(np.uint8)
            rgb[~case['observation_valid'][pos]] = (90,90,90)
            imgs.append(rgb)
        valid, target, pred = case['valid'], case['target'], case['pred']
        gold = np.full((128,128,3), (245,245,245), dtype=np.uint8)
        gold[target & valid] = (20,160,65); gold[~valid] = (100,100,100)
        error = np.full((128,128,3), (245,245,245), dtype=np.uint8)
        error[target & pred & valid] = (20,160,65)
        error[~target & pred & valid] = (230,65,40)
        error[target & ~pred & valid] = (40,95,230)
        error[~valid] = (100,100,100)
        imgs.extend([gold,error])
        draw.text((20,y-14), f"Case {row+1}, patch {case['patch_id']} (train_pool); target class {case['target_class']}", fill='black')
        for col, (title, arr) in enumerate(zip(titles, imgs)):
            x = 20 + col*(cell+gap)
            draw.text((x,y+2), title, fill='black')
            canvas.paste(Image.fromarray(arr).resize((cell,cell), Image.Resampling.NEAREST), (x,y+20))
    x = 20
    for label, color in [('TP / gold target',(20,160,65)),('False positive',(230,65,40)),
                         ('False negative',(40,95,230)),('Void / nodata',(100,100,100)),('True negative',(245,245,245))]:
        draw.rectangle((x,674,x+14,688), fill=color, outline='black')
        draw.text((x+20,675),label,fill='black'); x += 215
    draw.text((20,703),'Mask gold is annual crop reference. Clouds and parcel independence are not certified.',fill='black')
    canvas.save(out/'p1_training_qa.png')


def audit(run, prepared, episodes, out):
    run, prepared, episodes, out = map(lambda p:Path(p).resolve(), (run,prepared,episodes,out))
    out.mkdir(parents=True,exist_ok=False)
    report = {'status':'running','scope':'P1 engineering audit, not P2 or independent evaluation',
              'audit_script_sha256':sha(__file__), 'model_or_gpu_executed':False,
              'checkpoint_torch_loaded':False,'semantic_language_learning_demonstrated':False,
              'development_scored':False,'generalization_demonstrated':False,'cases':[]}
    try:
        receipt = load_json(run/'receipt.json')
        selected = load_json(run/'selected_examples.json')
        contract = load_json(episodes/'episode_contract.json')
        check(sha(prepared/'manifest.jsonl') == contract['prepared_manifest_sha256'],'prepared manifest hash')
        check(sha(episodes/'episodes_train.jsonl') == contract['public_catalog_sha256']['train'],'public train catalog hash')
        manifest = jsonl(prepared/'manifest.jsonl','patch_id')
        public = jsonl(episodes/'episodes_train.jsonl','episode_id')
        scoring = jsonl(episodes/'scoring/scoring_train.jsonl','episode_id')
        eligible = [r for r in scoring.values() if r['k_pairs'] == 1 and .10 <= r['target_pixels']/r['label_valid_pixels'] <= .50]
        eligible.sort(key=lambda r:hashlib.sha256(('oe9-p1-two-case:'+r['episode_id']).encode()).hexdigest())
        expected, seen = [],set()
        for r in eligible:
            if r['query_patch_id'] not in seen: expected.append(r);seen.add(r['query_patch_id'])
            if len(expected) == 2: break
        check(len(expected) == 2 and selected['rows'] == expected, 'deterministic two-case selection mismatch')
        check(len(selected['audits']) == 2, 'audit count')
        check(receipt['query_dates'] == [2,5] and receipt['support_dates'] == 8
              and receipt['spatial_resolution'] == [128,128], 'worker observation/resolution contract')
        final = receipt['head_fit']['final']
        check(len(final) == 2, 'worker final count')
        cases = []
        for i,(s,a) in enumerate(zip(selected['rows'],selected['audits'])):
            e = public[s['episode_id']];m=manifest[e['query_patch_id']]
            check(e['split'] == 'train' and e['k_pairs'] == 1 and m['role'] == 'train'
                  and m['training_partition'] == 'train_pool' and m['supervised_training_allowed'] is True,'query training role')
            check(a['episode_id'] == e['episode_id'] and a['query_patch_id'] == e['query_patch_id']
                  and a['split'] == 'train' and a['acquired_query_positions'] == [2,5]
                  and a['returned_query_observations'] == 2
                  and a['returned_support_observation_instances'] == 16
                  and a['spatial_shape'] == [128,128] and a['downsampling_applied'] is False
                  and a['query_gold_read'] is False,'runtime query/support acquisition audit')
            support_ids, objects = set(),set()
            for sup in e['support_pairs'][0].values():
                sm = manifest[sup['patch_id']]
                check(sm['training_partition'] == 'train_pool' and sm['role'] == 'train'
                      and sm['supervised_training_allowed'] is True,'support training role')
                check(sup['patch_id'] != e['query_patch_id'] and sup['object_key'] not in objects,'query/support overlap')
                check(sup['observation_ids'] == [f"{sup['patch_id']}:obs:{j}" for j in range(8)]
                      and sup['dates_yyyymmdd'] == sm['selected_dates'],'support all8 dates')
                check(sup['input_npz'] == sm['npz_path'] and sup['input_sha256'] == sm['npz_sha256'],'support input identity')
                check(sha(path_in(prepared,sm['npz_path'],'inputs')) == sm['npz_sha256'],'support file hash')
                check(sha(path_in(episodes,sup['mask_npz'],'support_masks')) == sup['mask_sha256'],'support mask hash')
                support_ids.add(sup['patch_id']);objects.add(sup['object_key'])
            check(a['support_patch_ids'] == sorted(support_ids)
                  and a['cpu_decoded_observations'] == 8*(len(support_ids)+1),'CPU all8 accounting')
            check(e['query_input_npz'] == m['npz_path'] and e['query_input_sha256'] == m['npz_sha256'],'query input identity')
            ipath=path_in(prepared,m['npz_path'],'inputs');lpath=path_in(prepared,m['label_path'],'labels')
            check(sha(ipath) == m['npz_sha256'],'original input hash')
            check(sha(lpath) == m['label_sha256'] == s['query_label_sha256']
                  == s['expected_query_label_sha256'],'original label hash')
            with np.load(lpath,allow_pickle=False) as z: semantic=z['semantic'];valid=z['label_valid']
            check(semantic.shape == valid.shape == (128,128) and semantic.dtype == np.int64
                  and valid.dtype == np.bool_ and np.array_equal(valid,semantic != 19),'source semantic/validity')
            target=(semantic == s['target_class']) & valid
            with np.load(run/f'train_prediction_{i}.npz',allow_pickle=False) as z:
                check(set(z.files) == {'probability','target','valid'},'prediction keys')
                prob=z['probability'];saved_target=z['target'];saved_valid=z['valid']
            check(np.array_equal(saved_target,target) and np.array_equal(saved_valid,valid),'saved prediction gold differs from original')
            values,pred=metrics(prob,saved_target,saved_valid)
            check(abs(values['iou']-final[i]['iou']) <= 2e-6,'worker IoU mismatch')
            check(abs(values['predicted_fraction']-final[i]['predicted_fraction']) <= 2e-6,'prediction fraction mismatch')
            check(abs(values['target_fraction']-final[i]['target_fraction']) <= 2e-6,'target fraction mismatch')
            with np.load(ipath,allow_pickle=False) as z: raw=z['raw_selected_s2'];ov=z['observation_valid']
            check(raw.shape == (8,10,128,128) and ov.shape == (8,128,128),'original input spatial shape')
            result={'episode_id':s['episode_id'],'query_patch_id':e['query_patch_id'],
                    'original_input_sha256':m['npz_sha256'],'original_label_sha256':m['label_sha256'],
                    'metrics':values,'worker_iou':final[i]['iou'],
                    'worker_mask_loss':final[i]['loss'],
                    'probability_reconstructed_loss_abs_difference':abs(values['approximate_mask_loss_from_saved_probabilities']-final[i]['loss'])}
            report['cases'].append(result)
            cases.append({'raw':raw,'observation_valid':ov,'valid':valid,'target':target,'pred':pred,
                          'patch_id':e['query_patch_id'],'target_class':s['target_class'],
                          'dates':m['selected_dates'],'metrics':values})
        curve=load_json(run/'head_curve.json')
        check(curve and all(curve[i]['step'] < curve[i+1]['step'] for i in range(len(curve)-1)),'head curve ordering')
        for i in range(2): check(abs(curve[-1]['cases'][i]['iou']-final[i]['iou']) <= 2e-6,'last curve vs final IoU')
        measured_gate=all(r['metrics']['iou'] >= .80 for r in report['cases'])
        check(measured_gate is receipt['head_fit']['all_iou_at_least_0_80'],'overfit gate flag mismatch')
        report.update(worker_status=receipt['status'],head_fit_gate_pass=measured_gate,
                      worker_full_p1_gate_pass=receipt.get('p1_full_gate_pass'),
                      worker_continuation=receipt.get('continuation'),
                      continuation_independently_reexecuted=False,
                      prediction_stage='frozen-feature head fit before live joint encoder/VLM updates',
                      loss_note='BCE+dice recomputed from saved sigmoid probabilities with clipping 1e-7; '
                      'not exact original-logit BCE, so loss difference is informational and not a pass/fail gate.')
        cp=run/'trainer_step1.pt'
        report['checkpoint_file_present']=cp.is_file()
        report['checkpoint_sha256_independently_verified']=False
        if cp.is_file():
            actual_sha=sha(cp)
            check(actual_sha == receipt.get('checkpoint_sha256'),'checkpoint SHA mismatch')
            report.update(checkpoint_sha256=actual_sha,checkpoint_bytes=cp.stat().st_size,
                          checkpoint_sha256_independently_verified=True)
        figure(cases,out)
        for name in ['receipt.json','selected_examples.json','head_curve.json','train_prediction_0.npz','train_prediction_1.npz']:
            shutil.copy2(run/name,out/name)
        optional={'loader_server_audit.json':[run/'loader_server_audit.json',run.parent/'loader_server_audit.json',run.parent/'loader_audit_v0.json'],
                  'controller_status.json':[run.parent/'p1_status_v0.json'],
                  'p1_worker_v0.log':[run.parent/'p1_worker_v0.log']}
        copied={}
        for name,options in optional.items():
            source=next((x for x in options if x.is_file()),None)
            if source is not None:shutil.copy2(source,out/name);copied[name]=str(source)
        report['optional_artifacts_copied']=copied
        report['status']='passed_prediction_and_input_audit'
    except Exception as exc:
        report.update(status='failed',error=repr(exc))
        (out/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
        raise
    (out/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    files={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir()) if p.is_file()}
    (out/'export_manifest.json').write_text(json.dumps({'files':files,'checkpoint_exported':False},indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-root',required=True);p.add_argument('--prepared-root',required=True)
    p.add_argument('--episodes-root',required=True);p.add_argument('--out',required=True)
    a=p.parse_args()
    r=audit(a.run_root,a.prepared_root,a.episodes_root,a.out)
    print(json.dumps({'status':r['status'],'head_fit_gate_pass':r['head_fit_gate_pass'],
                      'ious':[c['metrics']['iou'] for c in r['cases']], 'out':a.out}))


if __name__ == '__main__':main()
