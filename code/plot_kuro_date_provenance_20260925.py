"""Plot two verified metadata joins; does not replace image acquisition dates."""
import argparse
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    root = a.repo / 'artifacts/kuro_original_metadata_pin_20260925'
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    for name in ('two_case_metadata_join_audit.json', 'published_geobench_slot_contract.json'):
        if sha(root / name) != manifest['files'][name]['sha256']:
            raise ValueError('Pinned source changed')
    audit = json.loads((root / 'two_case_metadata_join_audit.json').read_text())
    contract = json.loads((root / 'published_geobench_slot_contract.json').read_text())
    keys = ('SL2', 'SL1', 'MS1')
    declared = [contract['published_mapping'][s]['source_date'] for s in ('pre_event_1', 'pre_event_2', 'post_event')]
    for tile in ('ks_06770', 'ks_05265'):
        case = audit['cases'][tile]
        if case['metadata_join_status'] != 'unique_exact_rectangle_and_source_fields':
            raise ValueError('Exact metadata join missing')
        if [case['upstream_date_and_id_sources'][k]['source_date'] for k in keys] != declared:
            raise ValueError('Source date mapping differs')
    # Recover the actual imputed E5 dates from the preserved source questions.
    case = json.loads((a.repo / 'artifacts/fixed_case_grounding_v0_20260925/ks_06770/source_manifest.json').read_text())
    by_kind = {q['kind']: q['dates'] for q in case['qa_items']}
    imputed = [by_kind['neg'][0], by_kind['pos'][0], by_kind['pos'][1]]
    event = date.fromisoformat(case['event_date'])
    if by_kind['neg'][1] != imputed[1]:
        raise ValueError('Imputed overlap differs')
    actual = [date.fromisoformat(d) for d in declared]
    assumed = [date.fromisoformat(d) for d in imputed]
    spans = [(d[2]-d[1]).days for d in (actual, assumed)]
    if spans != [168, 12]:
        raise ValueError('Unexpected fixed-case date intervals')
    a.out.mkdir(exist_ok=False)
    fig, ax = plt.subplots(figsize=(12, 5.2), constrained_layout=False)
    colors = ('#2166ac', '#209b86', '#b24b43')
    names = ('pre_1', 'pre_2', 'post')
    for dates, y, span in ((actual, 1.25, spans[0]), (assumed, .25, spans[1])):
        ax.plot(dates, [y]*3, color='#c8cdd3', lw=2, zorder=1)
        for i, (day, name, color) in enumerate(zip(dates, names, colors)):
            ax.scatter(day, y, s=65, c=color, zorder=3)
            ax.annotate(name+'\n'+day.isoformat(), (day, y), xytext=(0, 13 if i != 1 else -33),
                        textcoords='offset points', ha='center', fontsize=9, color=color)
        y_arrow = y + .38
        ax.annotate('', (dates[2], y_arrow), (dates[1], y_arrow), arrowprops={'arrowstyle': '<->', 'color': '#374151'})
        ax.text(dates[1]+(dates[2]-dates[1])/2, y_arrow+.06, str(span)+' days', ha='center', fontsize=10)
    ax.axvline(event, color='#8c8c8c', ls=':', lw=1)
    ax.text(event-timedelta(days=4), 2.04, 'Event: '+event.isoformat(), ha='right', color='#555555', fontsize=9)
    ax.set_yticks([.25, 1.25], ['Current E5\nimputed dates', 'Original metadata\n+ published mapping'])
    ax.set_ylim(-.45, 2.2)
    ax.set_xlim(actual[0]-timedelta(days=23), actual[-1]+timedelta(days=25))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.grid(axis='x', color='#eceff2', lw=.7)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0, pad=12)
    fig.suptitle('The observation interval is not the event-date approximation', x=.52, y=.98, fontsize=14)
    fig.subplots_adjust(left=.19, right=.98, top=.89, bottom=.27)
    caption = ('Two preselected development tiles: ks_06770 and ks_05265, event 562. Exact geometry/metadata joins.\n'
               'Dates are documented by original metadata and the published GEO-Bench pipeline; original pixel correspondence\n'
               'and the historical packaging revision remain unverified. No temporal-performance claim; E5 inputs unchanged.')
    fig.text(.19, .055, caption, fontsize=8.5, linespacing=1.5, color='#454545')
    fig.savefig(a.out / 'date_provenance.png', dpi=180)
    fig.savefig(a.out / 'date_provenance.pdf')
    plt.close(fig)
    output = {'created_at': datetime.now(timezone.utc).isoformat(), 'source_dates_published_mapping': declared,
              'e5_imputed_dates': imputed, 'pre2_post_days': {'documented': spans[0], 'imputed': spans[1]},
              'source_sha256': {name: sha(root / name) for name in ('MANIFEST.json', 'two_case_metadata_join_audit.json', 'published_geobench_slot_contract.json')},
              'script_sha256': sha(Path(__file__)), 'matplotlib': matplotlib.__version__, 'caption': caption,
              'outputs': {name: sha(a.out / name) for name in ('date_provenance.png', 'date_provenance.pdf')}}
    (a.out / 'plot_provenance.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()
