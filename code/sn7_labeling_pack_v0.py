#!/usr/bin/env python3
"""Build the prefix-visible gold labelling package (docs/PREFIX_VISIBLE_GOLD_PROTOCOL_v0.md) from diag_lite_v0_3 items.
Output: spacenet7/labeling_v0/{index.html, items_public.json, frames/<aoi>/<month>_<Q>.jpg}. items_public.json carries NO gold and NO
privileged frames; only frames <= cutoff for the item's quadrant are listed (post-cutoff frames are not even referenced)."""
import json, sys, random, shutil
from pathlib import Path
from PIL import Image
ROOT=Path("/home/work/data/olmoearth/spacenet7"); SRC=ROOT/"diag_lite_v0_3"; OUT=ROOT/"labeling_v0"; (OUT/"frames").mkdir(parents=True,exist_ok=True)
items=[json.loads(l) for l in (SRC/"items.jsonl").read_text().splitlines() if l]
pub=[]; rng=random.Random(20260922)
for it in items:
    pre=it["conds"]["full_prefix"]; q=it["region"]
    for m in pre:
        src=Path(it["png"][m][q]); dst=OUT/"frames"/it["aoi"]/f"{m}_{q}.jpg"; dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists(): Image.open(src).convert("RGB").save(dst,quality=85,optimize=True)
    # silver candidate frames offered as suggestions (label-derived; annotator may reject/add). Q1 'no' items have none.
    cand=it["conds"].get("privileged_silver") or []
    pub.append({"id":it["id"],"aoi":it["aoi"],"cutoff":it["cutoff"],"q":it["q"],"region":q,"window":it.get("window"),"frames":[f"frames/{it['aoi']}/{m}_{q}.jpg" for m in pre],"months":pre,"candidates":cand})
rng.shuffle(pub)
(OUT/"items_public.json").write_text(json.dumps(pub,ensure_ascii=False))
html=r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>prefix-visible gold v0</title>
<style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee}#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:6px}
.fr{position:relative;border:2px solid #333;cursor:pointer}.fr.sel{border-color:#0f0}.fr.cand{outline:2px dashed #fa0;outline-offset:-4px}.fr img{width:100%;display:block}
.fr span{position:absolute;left:4px;top:4px;background:#000a;padding:2px 4px;font-size:12px}#big{position:fixed;inset:0;background:#000d;display:none;align-items:center;justify-content:center}#big img{max-width:96vw;max-height:96vh}
button{margin:4px;padding:6px 10px}#q{font-size:18px;margin:10px 0}.ans button.on{background:#0a0;color:#fff}</style></head><body>
<div><b>검수자</b> <input id="who" placeholder="이니셜"> <span id="prog"></span></div>
<div id="q"></div>
<div class="ans" id="ans"></div>
<div>근거 프레임(2~4장, 클릭으로 선택; 주황 점선 = 라벨에서 뽑은 후보, 그대로 받아쓰지 말고 확인/기각). 클릭한 프레임은 더블클릭으로 확대.</div>
<div id="grid"></div>
<div><button onclick="save()">저장·다음</button> <button onclick="skip()">판독 불가</button> <button onclick="dl()">JSON 내려받기</button></div>
<div id="big" onclick="this.style.display='none'"><img id="bigimg"></div>
<script>
let items=[],i=0,sel=new Set(),ans=null,t0=0,out=[];
fetch('items_public.json').then(r=>r.json()).then(d=>{items=d;const s=localStorage.getItem('pvg_out');if(s){out=JSON.parse(s);i=out.length}show();});
function qtext(it){if(it.q=='Q1')return `Q1. 이 영역(${it.region})에서 <b>${it.window[0]} ~ ${it.window[1]}</b> 사이에 새 건물이 생겼나?`;return `Q2. 이 영역(${it.region})에서 새 건설이 <b>처음 분명히 보이는 달</b>은? (아래 프레임 중 하나를 '정답 달'로 먼저 클릭 → 근거 추가 선택)`;}
function show(){if(i>=items.length){document.getElementById('q').innerText='끝. JSON 내려받기를 눌러 저장하세요.';return}
const it=items[i];sel=new Set();ans=null;t0=Date.now();document.getElementById('prog').innerText=`${i+1}/${items.length}`;document.getElementById('q').innerHTML=qtext(it);
const a=document.getElementById('ans');a.innerHTML='';(it.q=='Q1'?['yes','no']:['월 선택 후 저장']).forEach(v=>{const b=document.createElement('button');b.innerText=v;b.onclick=()=>{ans=v;[...a.children].forEach(c=>c.classList.remove('on'));b.classList.add('on')};a.appendChild(b)});
const g=document.getElementById('grid');g.innerHTML='';it.frames.forEach((f,k)=>{const d=document.createElement('div');d.className='fr'+(it.candidates.includes(it.months[k])?' cand':'');d.innerHTML=`<img src="${f}"><span>${it.months[k]}</span>`;
d.onclick=()=>{const m=it.months[k];if(sel.has(m))sel.delete(m);else sel.add(m);d.classList.toggle('sel');if(it.q=='Q2'&&!ans){ans=m;document.getElementById('q').innerHTML=qtext(it)+` <b style="color:#0f0">정답 달: ${m}</b>`}};
d.ondblclick=e=>{e.stopPropagation();document.getElementById('bigimg').src=f;document.getElementById('big').style.display='flex'};g.appendChild(d)})}
function rec(status){const it=items[i];out.push({id:it.id,q:it.q,annotator:document.getElementById('who').value,answer:ans,evidence:[...sel].sort(),status,seconds:Math.round((Date.now()-t0)/1000),ts:new Date().toISOString()});localStorage.setItem('pvg_out',JSON.stringify(out));i++;show()}
function save(){if(!ans){alert('답을 고르세요');return}if(sel.size<1){alert('근거 프레임을 1장 이상 고르세요');return}rec('ok')}
function skip(){ans=null;rec('unreadable')}
function dl(){const b=new Blob([JSON.stringify(out,null,1)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=`pvg_${document.getElementById('who').value||'anon'}.json`;a.click()}
</script></body></html>'''
(OUT/"index.html").write_text(html)
shutil.copy(Path(__file__).resolve(),OUT/"sn7_labeling_pack_v0.py")
n=sum(1 for _ in (OUT/"frames").rglob("*.jpg")); print("items",len(pub),"jpg",n); print("PACK DONE")
