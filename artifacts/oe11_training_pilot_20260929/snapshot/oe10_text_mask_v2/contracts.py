"""Pure-Python boundary checks. Context is public request input, never a target."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PROMPT=('Find regions matching the positive examples and exclude the counterexamples. '
        'Return a mask and the acquired observation dates supporting it.')
PINNED_BASE={
 'episode_model.py':'4997e85692b5600a034af351cd16360ff303a3a133bacb5d5756c53c89098297',
 'episode_loader.py':'536de03b77bcd5bfedf439b5456d9e0fe60a3789c5f9c1a53c5511ded05d07f6',
 'native_replay.py':'26c0fab96c328e9a5a022a574caf5f78f7ffda4e6fb01164128283f7918ff670'}

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()

def verify_base(root=ROOT/'base_snapshot'):
 actual={name:sha(Path(root)/name) for name in PINNED_BASE}
 if actual!=PINNED_BASE:raise ValueError('Immutable P2 base source hash mismatch')
 return actual

def load_base(name):
 verify_base()
 if name+'.py' not in PINNED_BASE:raise ValueError('Unknown pinned module')
 spec=importlib.util.spec_from_file_location('_oe10_text_mask_base_'+name,ROOT/'base_snapshot'/(name+'.py'))
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 return module

def validate_context(value):
 if type(value) is not dict or set(value)!={'instruction','positive','counterexample'}:
  raise ValueError('Context accepts instruction/positive/counterexample only; no IDs, targets, sources or answers')
 if value['instruction']!=PROMPT:raise ValueError('Instruction must equal the pinned common prompt')
 for role in ('positive','counterexample'):
  text=value[role]
  if type(text) is not str or not text.strip() or len(text)>8000:raise ValueError('Nonempty bounded role text required')
  if any(t in text.lower() for t in ('<|','|>','target cover:','reference: region_1','assistant:')):
   raise ValueError('Role text contains reserved chat/answer material')
 return dict(value)

def render_context(value):
 c=validate_context(value)
 # No assistant turn, answer target, image coverage, ID or evaluation metadata.
 return c['instruction']+'\nPositive examples describe:\n'+c['positive']+'\nCounterexamples describe:\n'+c['counterexample']

def context_sha(value):
 return hashlib.sha256(json.dumps(validate_context(value),sort_keys=True,ensure_ascii=False).encode()).hexdigest()
