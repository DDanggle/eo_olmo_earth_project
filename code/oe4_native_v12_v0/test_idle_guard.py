"""Regression: unknown nvidia-smi output must never authorize a GPU launch."""
import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('native_controller',Path(__file__).with_name('run_bounded_v1.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class IdleGuardTests(unittest.TestCase):
 def check(self,gpus,apps,expected):
  with patch.object(module.subprocess,'check_output',side_effect=[gpus,apps]):
   self.assertEqual(module.idle(),expected)
 def test_idle_priority(self):self.check('0, GPU-a, 0, 0\n1, GPU-b, 0, 0\n','',[1,0])
 def test_busy_pid(self):self.check('0, GPU-a, 0, 0\n','GPU-a, 123\n',[])
 def test_missing_util(self):self.check('0, GPU-a, 0, [Not Found]\n','',[])
 def test_missing_memory(self):self.check('0, GPU-a, N/A, 0\n','',[])
 def test_malformed_apps(self):self.check('0, GPU-a, 0, 0\n','[Unknown]\n',[])
 def test_threshold(self):self.check('0, GPU-a, 512, 0\n1, GPU-b, 0, 5\n','',[])
 def test_query_timeout(self):
  with patch.object(module.subprocess,'check_output',side_effect=subprocess.TimeoutExpired('nvidia-smi',15)):
   self.assertEqual(module.idle(),[])

if __name__=='__main__':unittest.main()
