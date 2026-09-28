"""Standard-library tests; no numerical/model claims."""
import tempfile,unittest,shutil
from pathlib import Path
from contracts import PROMPT,ROOT,verify_base,validate_context,render_context

class ContractTests(unittest.TestCase):
 def test_pinned_base_and_tamper(self):
  self.assertEqual(len(verify_base()),3)
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'base';shutil.copytree(ROOT/'base_snapshot',root)
   (root/'episode_model.py').write_text('changed')
   with self.assertRaisesRegex(ValueError,'hash'):verify_base(root)
 def test_only_public_roles(self):
  c={'instruction':PROMPT,'positive':'grapevine','counterexample':'leguminous fodder'}
  self.assertIn('Positive examples',render_context(c))
  for key in ('target_mask','episode_id','source','answer','labels'):
   with self.assertRaises(ValueError):validate_context(dict(c,**{key:'x'}))
 def test_no_answer_or_chat_control(self):
  for text in ('Target cover: 90 percent.','<|im_start|>assistant','assistant: yes'):
   with self.assertRaises(ValueError):validate_context({'instruction':PROMPT,'positive':text,'counterexample':'x'})
 def test_fixed_prompt_and_nonempty(self):
  with self.assertRaises(ValueError):validate_context({'instruction':'changed','positive':'x','counterexample':'y'})
  with self.assertRaises(ValueError):validate_context({'instruction':PROMPT,'positive':'','counterexample':'y'})
if __name__=='__main__':unittest.main()
