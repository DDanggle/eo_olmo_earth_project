#!/usr/bin/env python3
"""Reproduce D1 contract issues with synthetic records; never load models or real gold.

Usage: python audit_d1_contract_20260924.py --repo /path/to/eo_olmo_earth_project
Requires numpy for the existing scorer. Writes only to an automatically removed temp dir.
The assertions describe the audited 2026-09-24 implementation, not desired behavior.
"""
import argparse
import contextlib
import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    sys.path.insert(0, str(args.repo / 'code'))
    import sn7_visible_pack_v05 as pack
    import sn7_visible_contract_v05 as contract
    import sn7_d1_reader_run_v0 as reader

    def episode(i):
        return {'id': f'ep-{i}', 'aoi': f'AOI-{i // 2}', 'region': ('NW', 'SE')[i % 2],
                'cutoff': '2020-04', 'reference_id': 'F000',
                'frames': [{'id': f'F{j:03d}', 'date': f'2020-{j + 1:02d}',
                            'path': f'frames/{i}-{j}.png'} for j in range(4)]}

    def annotation(ep, who, outcome, first=1):
        states = {f['id']: 'no_visible_change' for f in ep['frames']}
        if outcome == 'change_supported':
            for f in ep['frames'][first:]:
                states[f['id']] = 'visible_change'
        elif outcome == 'insufficient_evidence':
            states['F001'] = 'ambiguous'
        return {'episode_id': ep['id'], 'annotator_id': who, 'status': 'complete', 'states': states}

    episodes = [episode(i) for i in range(12)]
    payload = {'pack_id': 'synthetic-d1-audit-only', 'episodes': episodes}
    outcomes = ['change_supported', 'no_visible_change', 'insufficient_evidence']
    exports = []
    for who in ('a', 'b'):
        labels = [annotation(ep, who, outcomes[i] if i < 3 else
                             ('change_supported' if who == 'a' else 'no_visible_change'))
                  for i, ep in enumerate(episodes)]
        exports.append({'schema': 'sn7-visible-annotations-v0.5',
                        'pack_id': payload['pack_id'], 'annotator_id': who, 'annotations': labels})
    review = pack.review_exports(payload, exports)
    rate = len(review['targets']) / len(episodes)
    assert rate == .25 and review['diagnostic_manifest_ready']

    matrix = []
    for eid, target in review['targets'].items():
        ep = next(e for e in episodes if e['id'] == eid)
        for mode in ('real', 'metadata_only', 'blank'):
            matrix.append({'id': f'{eid}|{mode}', 'episode_id': eid, 'aoi': ep['aoi'],
                           'condition': mode, 'target': target if mode == 'real' else pack.withheld_target(ep),
                           'real_image_reference_target': target})
    for source, donor in (('ep-0', 'ep-1'), ('ep-1', 'ep-0')):
        matrix.append({'id': f'{source}|swap|{donor}', 'episode_id': source, 'aoi': 'AOI-0',
                       'condition': 'different_outcome_swap', 'target': review['targets'][donor]})
    fields = ('answer', 'first_change_id', 'last_clear_no_change_id', 'evidence_ids', 'current_state')
    answers = [{'id': r['id'], 'pred': {k: r['target'].get(k) for k in fields}} for r in matrix]

    with tempfile.TemporaryDirectory(prefix='d1-contract-audit-') as tmp:
        root = Path(tmp)
        (root / 'review_report.json').write_text(json.dumps(review))
        (root / 'diagnostic_matrix.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in matrix))
        (root / 'answers.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in answers))
        reached_loader = False
        # The sentinel replaces the loader, so this cannot load weights or invoke any GPU.
        with patch.object(reader, 'load_reader', side_effect=RuntimeError('MODEL_LOAD_SENTINEL')):
            try:
                reader.cmd_run(SimpleNamespace(root=str(root), matrix=str(root), pack=str(root),
                                               out=str(root / 'run'), reader='qwen'))
            except RuntimeError as exc:
                if str(exc) != 'MODEL_LOAD_SENTINEL':
                    raise
                reached_loader = True
        assert reached_loader
        with contextlib.redirect_stdout(io.StringIO()):
            reader.cmd_score(SimpleNamespace(matrix=str(root), answers=str(root / 'answers.jsonl'),
                                             out=str(root / 'score')))
        scored = json.loads((root / 'score/scores_answers.json').read_text())
        assert scored['content_check']['diff'] == 1.0
        assert scored['per_condition']['real']['answer_acc'] == 1.0
        assert scored['per_condition']['metadata_only']['answer_acc'] == 1.0
        assert scored['registered_reading'] == 'does_not_read'

        # Truncated answer files are accepted by the scorer rather than rejected as incomplete.
        (root / 'partial.jsonl').write_text(json.dumps(answers[0]) + '\n')
        with contextlib.redirect_stdout(io.StringIO()):
            reader.cmd_score(SimpleNamespace(matrix=str(root), answers=str(root / 'partial.jsonl'),
                                             out=str(root / 'partial-score')))
        partial = json.loads((root / 'partial-score/scores_partial.json').read_text())

    left, right = episodes[:2]
    def target(ep, first):
        return contract.consensus_target(ep, [annotation(ep, w, 'change_supported', first) for w in ('a', 'b')])
    source_target, donor_target = target(left, 1), target(right, 2)
    contract.make_control_pair(left, source_target, right, donor_target)
    pred = {k: donor_target.get(k) for k in fields}
    source_score = reader.score_one(pred, source_target)
    donor_score = reader.score_one(pred, donor_target)
    assert donor_score['first_ok'] and not source_score['first_ok']
    class_delta = int(donor_score['answer_ok']) - int(source_score['answer_ok'])
    assert class_delta == 0

    files = ['config/decision_experiment_d1_prereg_v0.json', 'code/sn7_visible_pack_v05.py',
             'code/sn7_visible_contract_v05.py', 'code/sn7_evidence_loss_v0.py',
             'code/sn7_d1_reader_run_v0.py']
    report = {
        'schema': 'synthetic-d1-contract-audit-20260924',
        'scope': 'Synthetic software audit only; no human annotation, satellite-image or model result.',
        'source_sha256': {f: hashlib.sha256((args.repo / f).read_bytes()).hexdigest() for f in files},
        'H_gate': {'agreed': 3, 'total': 12, 'agreement_rate': rate,
                   'diagnostic_manifest_ready': review['diagnostic_manifest_ready'],
                   'would_reach_model_loader': reached_loader},
        'perfect_reader': {'real_accuracy': 1.0, 'metadata_withheld_target_accuracy': 1.0,
                           'content_difference': scored['content_check']['diff'],
                           'content_ci95': scored['content_check']['ci95'],
                           'registered_reading': scored['registered_reading']},
        'temporal_swap': {'pair_accepted': True, 'source_first_id': source_target['first_change_id'],
                          'donor_first_id': donor_target['first_change_id'],
                          'prediction_matches_donor_first': donor_score['first_ok'],
                          'content_difference_using_answer_only': class_delta},
        'incomplete_answers': {'matrix_rows': len(matrix), 'answer_rows': 1,
                               'scorer_wrote_report': True, 'registered_reading': partial['registered_reading']},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
