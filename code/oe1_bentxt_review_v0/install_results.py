#!/usr/bin/env python3
"""Install only the completed OE1 local receipts and explicitly reviewed documents."""
import hashlib
import json
from pathlib import Path
import shutil

STAGE = Path('/private/tmp/oe1_resume')
ROOT = Path('/Users/dongdong/DongDong/ai_projects/eo_olmo_earth_project')
ART = ROOT / 'artifacts/oe1_bentxt_v0_20260926'
EXPECTED = {
    'GOAL.md': '12730b17b203b44aacb9a80d530cd44f27a66255d0e0e92f3f411313b07e4c26',
    'README.md': 'f8107f2725fbd780d93ed593a49fc79b8c70d2f7564984178b26fde13d26270e',
    'RESTART_HERE.md': '1ae9e57ca1ac8507396b2c946e8b8e8dd7762cff4899c57ac03ab49da67f2105',
    'STUDY.md': '41a8b75a804ccdcaf1c160e8a4fde33b79350f0fa3170a2c05dbd419bc1cd525',
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for name, expected in EXPECTED.items():
        if sha(ROOT / name) != expected:
            raise RuntimeError('Document changed since staging: ' + name)
    copies = [(STAGE / name, ROOT / name) for name in EXPECTED]
    copies.append((STAGE / 'OE1_BENTXT_FEASIBILITY_20260926.md', ROOT / 'docs/OE1_BENTXT_FEASIBILITY_20260926.md'))
    for name in ('pilot_v1_review_20260926.tar.gz', 'result_receipt_audit.json', 'launch_and_descriptive_audit.json'):
        copies.append((STAGE / name, ART / name))
    for directory in ('pilot_v1', 'data'):
        for path in sorted((STAGE / directory).rglob('*')):
            if path.is_file():
                assert path.suffix != '.pt'
                copies.append((path, ART / path.relative_to(STAGE)))
    for name in ('summarize_pilot.py', 'install_results.py'):
        copies.append((STAGE / name, ROOT / 'code/oe1_bentxt_review_v0' / name))
    # New artifact files may only replace identical previous copies.
    for source, target in copies:
        if target.is_file() and target.name not in EXPECTED:
            if sha(source) != sha(target):
                raise RuntimeError('Refusing to replace different artifact: ' + str(target))
    records = []
    for source, target in copies:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        assert sha(target) == sha(source)
        records.append({'path': str(target.relative_to(ROOT)), 'bytes': target.stat().st_size, 'sha256': sha(target)})
    delivery = {'scope': 'OE1 completed run receipts, local audit scripts and research documentation; model tensors remain on server',
                'files': records, 'count': len(records)}
    (ART / 'local_delivery_manifest.json').write_text(json.dumps(delivery, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'installed_and_verified': len(records), 'bytes': sum(x['bytes'] for x in records), 'report': str(ROOT / 'docs/OE1_BENTXT_FEASIBILITY_20260926.md')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
