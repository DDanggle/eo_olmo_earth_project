"""Post-hoc read-only OE10 label/readout audit; does not change frozen training."""
import argparse, hashlib, json, statistics
from pathlib import Path

def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]

def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--score',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    manifest=rows(a.bundle/'manifest.jsonl')
    training=rows(a.bundle/'episodes/scoring/scoring_train.jsonl')
    score=json.loads(a.score.read_text())
    dataset={}
    for role in ('train_pool','source_bank','dev_query'):
        selected=[r for r in manifest if r['training_partition']==role]
        dataset[role]={'unique_patches':len(selected),'classes':{}}
        for c in (1,2,3,4):
            present=[r for r in selected if r['class_pixel_counts'][c]>0]
            dataset[role]['classes'][str(c)]={
                'present_unique_patches':len(present),
                'total_class_pixels':sum(r['class_pixel_counts'][c] for r in selected),
                'mean_fraction_among_present_patches':statistics.mean(
                    r['class_pixel_counts'][c]/sum(r['class_pixel_counts'][:19]) for r in present) if present else None}
        dataset[role]['query_date_pairs_if_positions_2_5']={
            key:sum(1 for r in selected if str([r['selected_dates'][i] for i in (2,5)])==key)
            for key in sorted({str([r['selected_dates'][i] for i in (2,5)]) for r in selected})}
    byclass={}
    for c in (1,2,3,4):
        selected=[r for r in training if r['target_class']==c]
        byclass[str(c)]={'episodes':len(selected),'target_present_episodes':sum(r['target_present'] for r in selected),
            'present_query_patches':len({r['query_patch_id'] for r in selected if r['target_present']}),
            'target_absent_episodes':sum(not r['target_present'] for r in selected)}
    predictions={}
    for k in (1,2,4,8):
        predictions[str(k)]={}
        for c in (1,2,3):
            s=[r for r in score['per_episode_scoring_only'] if r['k']==k and r['target_class']==c and r['target_present']]
            gt=sum(r['target_pixels'] for r in s);pred=sum(r['predicted_valid_pixels'] for r in s);inter=sum(r['intersection'] for r in s)
            predictions[str(k)][str(c)]={
                'target_present_episode_rows':len(s),'unique_query_patches':len({r['query_patch_id'] for r in s}),
                'mean_episode_iou':statistics.mean(r['target_iou'] for r in s) if s else None,
                'zero_predicted_area_episode_rows':sum(r['predicted_valid_pixels']==0 for r in s),
                'sum_target_pixels_repeated_over_counterexamples':gt,
                'predicted_over_true_area_ratio':pred/gt if gt else None,
                'pooled_foreground_recall':inter/gt if gt else None}
    report={'schema_version':'oe10_posthoc_label_readout_audit_v1',
        'source_hashes':{'manifest':file_hash(a.bundle/'manifest.jsonl'),
            'scoring_train':file_hash(a.bundle/'episodes/scoring/scoring_train.jsonl'),'score':file_hash(a.score)},
        'dataset_partition_counts':dataset,'training_target_distribution':byclass,
        'current_readout_by_k_class':predictions,
        'scope':'Existing metadata and scored pixel counts only. No raw input re-extraction, threshold sweep, inference or training. Pooled recall here is descriptive, not a replacement primary metric.',
        'causal_explanations_verified':False}
    with a.out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({'dataset':dataset,'train':byclass,'readout':predictions},ensure_ascii=False))
if __name__=='__main__':main()
