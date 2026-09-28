'use strict';
(() => {
  const pilot = window.PILOT, $ = id => document.getElementById(id);
  if(pilot.synthetic_fixture_package){document.querySelector('h1').textContent='SYNTHETIC SOFTWARE TEST · 실제 판독 자료 아님';}
  const decisions = {target:'대상 · 양성 예시와 같음',counterexample:'혼동 · 혼동 예시와 같음',neither:'둘 다 아님',uncertain:'불확실 · 종류 판단 보류',unobservable:'관측 불가'};
  const issues = {seasonality:'계절·생육 시점',spectral:'색·분광 특성',shape:'형상·주변 맥락',cloud:'구름·그림자',resolution:'해상도 부족',dates:'필요 시점 부족',mixed:'후보 안에 여러 종류',examples:'예시 자체가 불명확',other:'그 외'};
  let state=null, key=null, current=null, running=false, lastTick=performance.now(), lastActivity=performance.now();
  const empty = caseId => ({case_id:caseId,status:'not_started',decision:null,evidence_frame_ids:[],reason:'',issues:[],confidence:null,intervals:[],active_seconds:0,first_started_at:null,completed_at:null,last_saved_at:null});
  const now = () => new Date().toISOString();
  const row = () => state && state.records[current];
  const datum = () => pilot.cases.find(c=>c.case_id===current);
  const save = () => {if(state){state.last_saved_at=now();if(row())row().last_saved_at=state.last_saved_at;localStorage.setItem(key,JSON.stringify(state));}};
  const escapeName = value => value.replace(/[^A-Za-z0-9_-]/g,'_');
  function tick(){
    const t=performance.now(), dt=Math.max(0,(t-lastTick)/1000);lastTick=t;
    if(running && !document.hidden && t-lastActivity<=60000){
      row().active_seconds+=dt;
      row().intervals.at(-1).active_seconds+=dt;
    }
    if(state){$('timer').textContent=`활성 ${Math.round(row().active_seconds)}초${running?' · 진행 중':' · 정지'}`;}
  }
  function pause(reason='manual'){
    tick();
    if(running){const interval=row().intervals.at(-1);interval.ended_at=now();interval.elapsed_seconds=Math.max(0,(Date.parse(interval.ended_at)-Date.parse(interval.started_at))/1000);interval.stop_reason=reason;running=false;save();}
    controls();
  }
  function controls(){
    if(!state)return;
    const available=datum().candidate_available;
    $('start').disabled=running||!available;
    $('pause').disabled=!running;$('observing').disabled=!running;
    $('answerFields').disabled=!running;
    document.querySelectorAll('.evidence').forEach(e=>e.disabled=!running);
    $('caseState').textContent=available?`${row().status==='complete'?'완료됨 · 재개하면 판독을 수정하고 다시 완료할 수 있습니다.':running?'판독 중':'시작 버튼을 누르면 영상을 볼 수 있습니다.'}`:datum().preparation_issue;
    $('taskContent').classList.toggle('hidden',row().status==='not_started'||!available);
  }
  function capture(){
    if(!state || !running)return;
    row().decision=document.querySelector('input[name=decision]:checked')?.value||null;
    row().evidence_frame_ids=Array.from(document.querySelectorAll('.evidence:checked')).map(e=>e.value);
    row().issues=Array.from(document.querySelectorAll('.issue:checked')).map(e=>e.value);
    row().reason=$('reason').value;row().confidence=$('confidence').value||null;save();
  }
  function drawFrames(kind){
    const container=$(kind+'Frames');container.replaceChildren();
    for(const f of datum().frames[kind]||[]){
      const fig=document.createElement('figure');fig.className='frame';
      const img=document.createElement('img');img.src=f[$('view').value];img.alt=`${f.date} 관측`;img.dataset.frameId=f.frame_id;
      img.onclick=()=>{if(!running)return;$('zoomImage').src=img.src;$('zoomDate').textContent=f.date;$('zoom').showModal();};
      const label=document.createElement('label'),box=document.createElement('input');box.type='checkbox';box.className='evidence';box.value=f.frame_id;box.checked=row().evidence_frame_ids.includes(f.frame_id);box.disabled=!running;box.onchange=capture;label.append(box,document.createTextNode(` ${f.date} · 근거`));fig.append(img,label);container.append(fig);
    }
  }
  function render(){
    $('caseTitle').textContent=current;
    for(const kind of ['query','positive','counterexample'])drawFrames(kind);
    document.querySelectorAll('input[name=decision]').forEach(e=>e.checked=e.value===row().decision);
    document.querySelectorAll('.issue').forEach(e=>e.checked=row().issues.includes(e.value));
    $('reason').value=row().reason;$('confidence').value=row().confidence||'';$('validation').textContent='';
    const nav=$('caseNav');nav.replaceChildren();
    for(const id of pilot.orders[state.assignment]){const button=document.createElement('button');button.textContent=id+(state.records[id].status==='complete'?' ✓':'');button.classList.toggle('complete',state.records[id].status==='complete');button.classList.toggle('current',id===current);button.onclick=()=>{capture();pause('case_switch');current=id;render();};nav.append(button);}
    $('progress').textContent=`완료 ${Object.values(state.records).filter(r=>r.status==='complete').length}/${pilot.case_count}`;controls();tick();
  }
  for(const [value,text] of Object.entries(decisions)){const label=document.createElement('label'),input=document.createElement('input');input.type='radio';input.name='decision';input.value=value;input.onchange=capture;label.append(input,document.createTextNode(' '+text));$('decisionChoices').append(label);}
  for(const [value,text] of Object.entries(issues)){const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.className='issue';input.value=value;input.onchange=capture;label.append(input,document.createTextNode(' '+text));$('issueChoices').append(label);}
  $('connect').onclick=()=>{
    const id=$('annotator').value.trim(),assignment=$('role').value;
    if(!/^[A-Za-z0-9_-]{1,64}$/.test(id)||!['A','B'].includes(assignment)){alert('영문·숫자·밑줄·하이픈으로 검수자 ID와 A/B 배정을 입력하세요.');return;}
    key=`${pilot.schema_version}:${pilot.package_id}:${assignment}:${id}`;
    const stored=localStorage.getItem(key);
    state=stored?JSON.parse(stored):{schema_version:pilot.schema_version,package_id:pilot.package_id,annotator_id:id,assignment,synthetic_fixture:!!pilot.synthetic_fixture_package,created_at:now(),last_saved_at:null,records:Object.fromEntries(pilot.cases.map(c=>[c.case_id,empty(c.case_id)]))};
    // Recover an interrupted interval only through its last persisted time, not through idle days.
    for(const record of Object.values(state.records)){const interval=record.intervals.at(-1);if(interval&&!interval.ended_at){interval.ended_at=record.last_saved_at||interval.started_at;interval.elapsed_seconds=Math.max(0,(Date.parse(interval.ended_at)-Date.parse(interval.started_at))/1000);interval.stop_reason='recovered_browser_interruption';}}
    current=pilot.orders[assignment].find(id=>state.records[id].status!=='complete')||pilot.orders[assignment][0];
    $('annotator').disabled=true;$('role').disabled=true;$('connect').disabled=true;$('export').disabled=false;$('workspace').classList.remove('hidden');
    $('identity').textContent=`검수자 ${id} · ${assignment} · 독립 판독. 다른 검수자의 답은 표시되지 않습니다.`;save();render();
  };
  $('start').onclick=()=>{if(!state||running)return;lastTick=performance.now();lastActivity=lastTick;running=true;const r=row();r.status='in_progress';r.completed_at=null;r.first_started_at=r.first_started_at||now();r.intervals.push({started_at:now(),ended_at:null,elapsed_seconds:null,active_seconds:0,stop_reason:null});save();render();};
  $('pause').onclick=()=>{capture();pause();};$('observing').onclick=()=>lastActivity=performance.now();
  $('complete').onclick=()=>{
    capture();const r=row();
    if(!r.decision||!r.confidence||!r.reason.trim()||!r.evidence_frame_ids.length){$('validation').textContent='판정·확신·이유·근거 날짜를 기록해 주세요.';return;}
    if(!r.evidence_frame_ids.some(x=>x.startsWith(current+'-q-'))){$('validation').textContent='판독 후보의 근거 날짜를 하나 이상 선택해 주세요.';return;}
    if(r.decision==='unobservable'&&!r.issues.some(x=>['cloud','resolution','dates','examples','other'].includes(x))){$('validation').textContent='관측 불가 원인을 선택해 주세요.';return;}
    pause('completed');r.status='complete';r.completed_at=now();save();render();
  };
  $('reason').oninput=capture;$('confidence').onchange=capture;
  $('view').onchange=()=>{for(const kind of ['query','positive','counterexample'])drawFrames(kind);};
  $('closeZoom').onclick=()=>$('zoom').close();
  $('export').onclick=()=>{
    capture();pause('export');
    const exported={...state,exported_at:now(),records:pilot.cases.map(c=>state.records[c.case_id]),timing_method:pilot.timing,
      independence_attestation:'Reviewer assigned to A or B; answers not exchanged through this interface. Independence still requires procedural verification.'};
    const link=document.createElement('a'),blob=new Blob([JSON.stringify(exported,null,2)],{type:'application/json'});link.href=URL.createObjectURL(blob);link.download=`${pilot.package_id}_${state.assignment}_${escapeName(state.annotator_id)}.json`;link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);
  };
  for(const event of ['pointerdown','pointermove','keydown','scroll'])window.addEventListener(event,()=>{lastActivity=performance.now();},{passive:true});
  document.addEventListener('visibilitychange',()=>{if(document.hidden){capture();pause('document_hidden');}});
  window.addEventListener('beforeunload',()=>{capture();pause('page_unload');});
  setInterval(()=>{tick();if(running)save();},1000);
})();
