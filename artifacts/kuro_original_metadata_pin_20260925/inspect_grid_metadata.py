"""Interpret only data-building pickle opcodes; no pickle.load/import/callables."""
import gzip,hashlib,json,pickletools,re,sys
from pathlib import Path
P=Path(__file__).resolve().parent
ALLOWED={'PROTO','FRAME','EMPTY_DICT','EMPTY_LIST','MARK','SETITEMS','APPENDS','APPEND','SHORT_BINUNICODE','BININT1','BININT2','BININT','BINFLOAT','NEWFALSE','NEWTRUE','MEMOIZE','BINGET','LONG_BINGET','STOP'}
def data_only(path):
 with gzip.open(path,'rb') as f:b=f.read(200_000_001)
 if len(b)>200_000_000:raise ValueError('metadata decompression cap')
 # Only strings/numbers/bools/list/dict operations are supported. No GLOBAL/REDUCE/BUILD.
 stack=[]; memo={}; mark=object()
 for op,arg,pos in pickletools.genops(b):
  n=op.name
  if n not in ALLOWED:raise ValueError(f'Unsupported opcode {n} at {pos}')
  if n in ('PROTO','FRAME'):continue
  if n=='EMPTY_DICT':stack.append({})
  elif n=='EMPTY_LIST':stack.append([])
  elif n=='MARK':stack.append(mark)
  elif n=='MEMOIZE':memo[len(memo)]=stack[-1]
  elif n in ('BINGET','LONG_BINGET'):stack.append(memo[arg])
  elif n in ('SHORT_BINUNICODE','BININT1','BININT2','BININT','BINFLOAT'):stack.append(arg)
  elif n=='NEWFALSE':stack.append(False)
  elif n=='NEWTRUE':stack.append(True)
  elif n=='APPEND':
   v=stack.pop();stack[-1].append(v)
  elif n in ('SETITEMS','APPENDS'):
   k=len(stack)-1
   while stack[k] is not mark:k-=1
   vals=stack[k+1:]; del stack[k:]
   if n=='APPENDS':stack[-1].extend(vals)
   else:
    if len(vals)%2:raise ValueError('odd key/value list')
    for j in range(0,len(vals),2):stack[-1][vals[j]]=vals[j+1]
  elif n=='STOP':
   if len(stack)!=1:raise ValueError('malformed stack')
   return stack[0]
 raise ValueError('missing STOP')
def main():
 file=P/(sys.argv[1] if len(sys.argv)>1 else 'KuroV2_grid_dict.gz'); x=data_only(file)
 targets={'ks_06770':(15540435,-2008105,15542675,-2005865),'ks_05265':(15526995,-2005865,15529235,-2003625)}
 def bounds(wkt):
  nums=[float(v) for v in re.findall(r'-?\d+(?:\.\d+)?',wkt)]
  return(min(nums[::2]),min(nums[1::2]),max(nums[::2]),max(nums[1::2]))
 candidates=[]; matches={k:[] for k in targets}
 for key,v in x.items():
  info=v['info']
  if info.get('actid')!=562 or info.get('aoiid')!=13:continue
  candidates.append({'key':key,**v}); bb=bounds(info['geom'])
  for tile,target in targets.items():
   if bb==target:matches[tile].append({'key':key,**v})
 out={'source_file':str(file),'source_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'total_records':len(x),'act562_aoi13_records':len(candidates),'matches':matches,'same_aoi_source_signatures':sorted(set(json.dumps(c['info']['sources'],sort_keys=True) for c in candidates))}
 f=P/(file.name+'.exact_matches.json'); f.write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
