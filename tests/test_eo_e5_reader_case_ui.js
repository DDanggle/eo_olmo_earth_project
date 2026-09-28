const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
class Node{
 constructor(tag,text){this.tag=tag;this.text=text||'';this.children=[];this.parentNode=null;this.root=false;}
 set textContent(value){this.text=String(value);this.replaceChildren();}get textContent(){return this.text+this.children.map(x=>x.textContent).join(' ')}
 append(...nodes){for(const n of nodes){n.parentNode=this;this.children.push(n)}}
 replaceChildren(...nodes){for(const n of this.children)n.parentNode=null;this.children=[];this.append(...nodes)}
 set innerHTML(value){throw new Error('Unsafe innerHTML')}setAttribute(){}get isConnected(){return this.root||Boolean(this.parentNode?.isConnected)}
}
const source=path.join(__dirname,'../code/eo_evidence_search_v0.html');
const html=fs.readFileSync(source,'utf8');
const code=html.slice(html.indexOf('let readerCaseVersion=0;'),html.indexOf('function label(s)'));
const context={console,encodeURIComponent,JSON,researchNode:(tag,text,cls)=>{const n=new Node(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n}};
vm.createContext(context);vm.runInContext(code,context);
const descendants=n=>[n,...n.children.flatMap(descendants)];
const blank={question_id:'synthetic_q1',reference_label:'yes',dates:['2020-01-01','2020-01-13'],slots:['pre_2','post'],reader_predictions:{1:'no',2:'no',3:'no'},blind_predictions:{1:'no',2:'no',3:'no'},limitations:[],c1_membership:{quality_eligible:true,supported_stratum:false},source_sha256:{}};
const saved={run_id:'E5-EB-v0',question_id:'synthetic_q1',source_role:'historical_model_output',source_gold:'yes',dates:blank.dates,slots:blank.slots,primary_membership:false,completed_at:'2026-09-25T07:47:56Z',source_sha256:{'<script>inert</script>':'b'.repeat(64)},predictions:{'full/native':{1:'no',2:'yes',3:null},'pair/native':{1:'yes',2:'no',3:'yes'},'later/native':{1:'no',2:'no',3:'yes'},'delta/native':{1:'yes',2:'yes',3:'no'},'full/full_no_delta':{1:'no',2:'no',3:'no'}}};
const absent=new Node('section');context.renderReaderCase(absent,{available:true,case:blank});assert(absent.textContent.includes('E5 · 연결된 저장 판독 없음'));assert.equal(descendants(absent).filter(x=>x.tag==='table').length,1);
const present={...blank,e5_control:saved},original=JSON.stringify(present),box=new Node('section');context.renderReaderCase(box,{available:true,case:present});assert.equal(JSON.stringify(present),original);
const tables=descendants(box).filter(x=>x.tag==='table');assert.equal(tables.length,2);const rows=descendants(tables[1]).find(x=>x.tag==='tbody').children;assert.equal(rows.length,5);
assert.deepEqual(rows.map(r=>r.children.slice(1).map(c=>c.text)),[['no','yes','미확인'],['yes','no','yes'],['no','no','yes'],['yes','yes','no'],['no','no','no']]);
assert(rows[4].textContent.includes('전체 입력 학습 후 D 제거'));assert(box.textContent.includes('첫 네 행은 입력별로 따로 학습'));assert(box.textContent.includes('E5 주분석: 제외'));assert(box.textContent.includes('문항 참조 답 yes'));assert(box.textContent.includes('개별 사례의 답이 전체 성능을 대표하지 않습니다.'));assert(!descendants(box).some(x=>['script','img'].includes(x.tag)));
const partial=JSON.parse(JSON.stringify(present));delete partial.e5_control.predictions['pair/native'];partial.e5_control.source_gold='no';partial.e5_control.primary_membership=true;
const partialBox=new Node('section');context.renderE5Control(partialBox,partial);assert.equal(descendants(partialBox).filter(x=>x.tag==='td'&&x.text==='저장 답 없음').length,3);assert(partialBox.textContent.includes('문항 참조 답 no'));assert(partialBox.textContent.includes('E5 주분석: 포함'));
console.log('PASS: E5 historical five-condition/three-seed answers preserve gold and distinguish retraining, intervention, missing response, membership and text-safe provenance.');
