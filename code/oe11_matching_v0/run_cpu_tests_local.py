"""Use existing local cached packages; install/download nothing. CPU only.

This environment helper is optional. Ordinary environments can instead execute
python -m unittest discover -s . -p test_matching.py -v.
"""
from pathlib import Path
import sys
import json
import unittest

ROOT = Path(__file__).resolve().parent
CACHED_PACKAGES = [
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
sys.path[:0] = [str(ROOT)] + CACHED_PACKAGES
import torch
import numpy as np
torch.set_num_threads(1)
suite = unittest.defaultTestLoader.discover(str(ROOT), pattern='test_matching.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
receipt = dict(scope='synthetic CPU head and toy EO/reader wrapper; no actual weights or data',
               python=sys.executable, torch_version=torch.__version__, numpy_version=np.__version__,
               torch_file=torch.__file__, numpy_file=np.__file__,
               gpu_used=False, actual_model_performance_measured=False,
               tests_run=result.testsRun, errors=len(result.errors), failures=len(result.failures),
               skipped=len(result.skipped), successful=result.wasSuccessful())
(ROOT / 'cpu_test_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
raise SystemExit(0 if result.wasSuccessful() else 1)
