const fs=require('fs'),vm=require('vm'),assert=require('assert');
class Node{
 constructor(tag,text){this.tag=tag;this.text=text||'';this.children=[];this.parentNode=null;this.root=false;}
 set textContent(value){this.text=String(value);this.replaceChildren();} get textContent(){return this.text+this.children.map(x=>x.textContent).join(' ')}
 append(...nodes){for(const n of nodes){n.parentNode=this;this.children.push(n)}}
 replaceChildren(...nodes){for(const n of this.children)n.parentNode=null;this.children=[];this.append(...nodes)}
 set innerHTML(value){throw new Error('Unsafe innerHTML write')} setAttribute(){} get isConnected(){return this.root||Boolean(this.parentNode?.isConnected)}
}
const html=fs.readFileSync(require('path').join(__dirname,'../code/eo_evidence_search_v0.html'),'utf8');
const code=html.slice(html.indexOf('let readerCaseVersion=0;'),html.indexOf('function label(s)'));
const detail=new Node('main');detail.root=true;const requests=[];
const context={console,encodeURIComponent,JSON,selected:null,searchVersion:0,$:()=>detail,researchNode:(tag,text,cls)=>{const n=new Node(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n},fetch:()=>new Promise(resolve=>requests.push(resolve))};
vm.createContext(context);vm.runInContext(code,context);
const snapshot=id=>({available:true,tile:id,checked_at:'2026-09-25T00:00:00Z',case:{tile:id,question_id:'<img src=x onerror=alert(1)>',run_id:'E2',source_role:'historical_model_output',dates:['2022-08-29','2022-09-10'],slots:['pre_2','post'],reference_label:'yes',reader_predictions:{1:'no',2:'yes',3:null},blind_predictions:{1:'no',2:'no',3:'no'},source_sha256:{code:'abc'},limitations:['<script>alert(1)</script>'],c1_membership:{quality_eligible:true,supported_stratum:false}}});
const response=data=>({ok:true,json:async()=>data});
const allArms=['real','earlier_only','later_only','repeat_earlier','repeat_later','no_delta','reverse'];
const descendants=n=>[n,...n.children.flatMap(descendants)];
const control=(c,arms)=>({run_id:'E3-PD-v1',question_id:c.question_id,source_role:'historical_model_output',source_gold:c.reference_label,predictions:Object.fromEntries(arms.map(arm=>[arm,{'1':'yes','2':'no','3':null}])),source_sha256:{'<img src=x onerror=alert(1)>':'a'.repeat(64)}});
const absent=snapshot('ks_none'),absentBox=new Node('section');context.renderReaderCase(absentBox,absent);assert(absentBox.textContent.includes('E3 · 이번 통제 미포함'));assert.equal(descendants(absentBox).filter(n=>n.tag==='table').length,1);
const full=snapshot('ks_full');full.case.e3_control=control(full.case,allArms);const before=JSON.stringify(full),fullBox=new Node('section');context.renderReaderCase(fullBox,full);assert.equal(JSON.stringify(full),before);const fullTables=descendants(fullBox).filter(n=>n.tag==='table');assert.equal(fullTables.length,2);assert.equal(descendants(fullTables[1]).filter(n=>n.tag==='tbody')[0].children.length,7);assert.equal(descendants(fullTables[1]).filter(n=>n.tag==='td'&&n.text==='미실행').length,0);assert.equal(descendants(fullTables[1]).filter(n=>n.tag==='td'&&n.text==='미확인').length,7);assert(fullBox.textContent.includes('변환된 장면의 새 정답이나 정확도를 뜻하지 않습니다.'));assert(fullBox.textContent.includes('원문항 참조 답 yes'));assert(!descendants(fullBox).some(n=>['img','script'].includes(n.tag)));
const hard=snapshot('ks_hard');hard.case.reference_label='no';hard.case.e3_control=control(hard.case,['real','later_only','repeat_later']);const hardBox=new Node('section');context.renderReaderCase(hardBox,hard);const hardTable=descendants(hardBox).filter(n=>n.tag==='table')[1];assert.equal(descendants(hardTable).filter(n=>n.tag==='td'&&n.text==='미실행').length,12);assert.equal(descendants(hardTable).filter(n=>n.tag==='td'&&n.text==='미확인').length,3);assert(hardBox.textContent.includes('원문항 참조 답 no'));
console.log('PASS: optional E3 absent, 7-row table, 3-arm hard negatives, unexecuted vs unparsed, original label/E2 preserved, text-only provenance');
const e4Arms=['real','delta_only','delta_sign_flip','delta_feature_permute'];
const e4Control=c=>({run_id:'E4-D-v0',question_id:c.question_id,source_role:'historical_model_output',source_gold:c.reference_label,predictions:Object.fromEntries(e4Arms.map(arm=>[arm,{'1':'yes','2':'no','3':null}])),source_sha256:{'<script>inert</script>':'b'.repeat(64)}});
assert(absentBox.textContent.includes('E4 · 이번 진단 미포함'));
const both=snapshot('ks_both');both.case.e3_control=control(both.case,allArms);both.case.e4_control=e4Control(both.case);const unchanged=JSON.stringify(both),bothBox=new Node('section');context.renderReaderCase(bothBox,both);assert.equal(JSON.stringify(both),unchanged);const bothTables=descendants(bothBox).filter(n=>n.tag==='table');assert.equal(bothTables.length,3);assert.equal(descendants(bothTables[0]).filter(n=>n.tag==='tbody')[0].children.length,3);assert.equal(descendants(bothTables[1]).filter(n=>n.tag==='tbody')[0].children.length,7);assert.equal(descendants(bothTables[2]).filter(n=>n.tag==='tbody')[0].children.length,4);assert.equal(descendants(bothTables[2]).filter(n=>n.tag==='td'&&n.text==='미확인').length,4);assert.equal(descendants(bothTables[2]).filter(n=>n.tag==='td'&&n.text==='미실행').length,0);assert(bothBox.textContent.includes('D=B−A는 앞·뒤 두 관측으로 계산한 차이입니다.'));assert(bothBox.textContent.includes('앞 관측이나 과거 정보가 불필요하다고 결론낼 수 없습니다.'));assert(bothBox.textContent.includes('변환된 장면의 새 정답이나 정확도를 뜻하지 않습니다.'));assert(!descendants(bothBox).some(n=>['img','script'].includes(n.tag)));
const e4Hard=snapshot('ks_e4_hard');e4Hard.case.reference_label='no';e4Hard.case.e4_control=e4Control(e4Hard.case);const e4HardBox=new Node('section');context.renderReaderCase(e4HardBox,e4Hard);assert.equal(descendants(e4HardBox).filter(n=>n.tag==='table').length,2);assert(e4HardBox.textContent.includes('원문항 참조 답 no'));assert(e4HardBox.textContent.includes('E3 · 이번 통제 미포함'));
// API rejects partial E4 blocks; the renderer still never maps a missing arm to no.
const partial=snapshot('ks_partial');partial.case.e4_control=e4Control(partial.case);delete partial.case.e4_control.predictions.delta_only;const partialBox=new Node('section');context.renderE4Control(partialBox,partial.case);assert.equal(descendants(partialBox).filter(n=>n.tag==='td'&&n.text==='미실행').length,3);assert.equal(descendants(partialBox).filter(n=>n.tag==='td'&&n.text==='미확인').length,3);
console.log('PASS: optional E4 absent, exact 4x3 table, E2/E3 unchanged, no fabricated missing predictions, both-observation/source-label caveats, text-safe provenance');
(async()=>{
 context.selected={id:'ks_A'};const first=context.loadReaderCase(context.selected);const boxA=detail.children[0];
 detail.replaceChildren();context.selected={id:'ks_B'};const second=context.loadReaderCase(context.selected);const boxB=detail.children[0];
 requests[0](response(snapshot('ks_A')));await first;assert(!boxA.textContent.includes('文')&&!boxA.textContent.includes('문항 참조 답'));assert(!boxB.textContent.includes('문항 참조 답'));
 requests[1](response(snapshot('ks_B')));await second;assert(boxB.textContent.includes('문항 참조 답: yes'));assert(boxB.textContent.includes('합성 날짜'));assert(boxB.textContent.includes('<script>alert(1)</script>'));assert(boxB.textContent.includes('미확인'));
 const tags=n=>[n.tag,...n.children.flatMap(tags)];assert(!tags(boxB).includes('script'));assert(!tags(boxB).includes('img'));assert(tags(boxB).includes('table'));
 detail.replaceChildren();context.selected={id:'ks_C'};const third=context.loadReaderCase(context.selected);const boxC=detail.children[0];context.searchVersion++;
 requests[2](response(snapshot('ks_C')));await third;assert(!boxC.textContent.includes('문항 참조 답'));
 const empty=new Node('section');context.renderReaderCase(empty,{available:false});assert(empty.textContent.includes('미확인'));assert(!tags(empty).includes('table'));
 detail.replaceChildren(new Node('p','원 참조 라벨 보존'));context.selected={id:'ks_D'};const fourth=context.loadReaderCase(context.selected);requests[3](response(snapshot('ks_WRONG')));await fourth;
 assert(detail.textContent.includes('원 참조 라벨 보존'));assert(detail.textContent.includes('확인하지 못했습니다'));assert(!detail.textContent.includes('문항 참조 답'));
 console.log('PASS: stale selection, stale search, text-only untrusted content, unknown state, mismatched response isolation');
})().catch(error=>{console.error(error);process.exitCode=1});
