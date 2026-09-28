"""Synthetic independent-auditor checks; never reads actual E5/E6 artifacts.

No production scorer/runner/head module imported. Fixtures are invented labels,
IDs, dates, answers and CPU tensors; no fitting or GPU discovery.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import audit_e6_results_v0 as audit


def fixture():
    items=[]
    def add(part,phen,tile,event,kind,special=False):
        row={'id':tile+':'+kind,'tile':tile,'partition':part,'phen':phen,'cluster':event,
             'kind':kind,'answer':'yes' if kind=='pos' else 'no',
             'dates':['2020-01-01','2020-01-13'] if kind!='neg' else ['2019-12-20','2020-01-01']}
        if special:row['dates']=['2021-02-01','2021-02-13']
        if phen=='flood':row.update(event=event,slots=['pre_1','pre_2'] if kind=='neg' else ['pre_2','post'])
        items.append(row)
    for phen,n,event_n in [('flood',1066,27),('landslide',518,7)]:
        for j in range(n):
            for kind in ('pos','neg'):add('train',phen,f'train.{phen}.{j}',f'train.{j%event_n}',kind)
            if phen=='flood':add('train','flood',f'train.hard.{j}',f'train.{j%event_n}','hard_neg')
    for event,n in enumerate([100,90,80,70,50,30,20,5,11,1]):
        for j in range(n):
            for kind in ('pos','neg'):add('test','flood',f'flood.{event}.{j}',str(event),kind,event==1 and j==0 and kind=='pos')
    for event,n in enumerate([1,2,3,4,5,6,7,429]):
        for j in range(n):add('test','flood',f'hard.{event}.{j}',str(event),'hard_neg',event==1 and j==0)
    for event,n in enumerate([191,1]):
        for j in range(n):
            for kind in ('pos','neg'):add('test','landslide',f'land.{event}.{j}',str(event),kind)
    test=[x for x in items if x['partition']=='test']
    sets={'all_test':[x['id'] for x in test],
          'primary_same_prompt':[x['id'] for x in test if x['phen']=='flood' and x['kind']!='neg' and int(x['cluster'])<8],
          'paired_flood':[x['id'] for x in test if x['phen']=='flood' and x['kind'] in ('pos','neg')],
          'hard_negative_flood':[x['id'] for x in test if x['kind']=='hard_neg'],
          'landslide':[x['id'] for x in test if x['phen']=='landslide'],
          'e3_subset':[x['id'] for x in test[:209]]}
    ordered={p:[x['id'] for x in items if x['partition']==p] for p in ('train','test')}
    batches={}
    for seed in (1,2,3):
        rng=np.random.default_rng(seed);epochs=[]
        for epoch in range(3):
            perm=rng.permutation(len(ordered['train']));ids=[ordered['train'][j] for j in perm]
            epochs.append([ids[j:j+8] for j in range(0,len(ids),8)])
        batches[str(seed)]=epochs
    index={x['id']:j for j,x in enumerate(items)}
    head=[];reference=[]
    for seed in (1,2,3):
        for item in test:
            base={k:item[k] for k in ('id','tile','cluster','phen','kind')}
            base.update(seed=seed,eval_arm='native',pair_index=index[item['id']],source_gold=item['answer'],transformed_gold=None)
            z=2. if item['answer']=='yes' else -2.
            head.append({**base,'model_arm':'full_head','logit':z,'probability':1/(1+math.exp(-z)),'prediction':item['answer']})
            reference.append({**base,'model_arm':'full','answer_raw':item['answer'].upper()+'.','parsed':item['answer']})
    return items,sets,ordered,batches,head,reference


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=fixture()
    def metrics(self,data=None):
        items,sets,ordered,batches,head,ref=data or self.data
        index,test,strata=audit.population(items,sets,ordered,batches)
        h,hp=audit.predictions(head,items,True);r,rp=audit.predictions(ref,items,False)
        return audit.recompute_metrics(test,strata,h,r),hp,rp
    def test_perfect_5989_population_and_two5265_systems(self):
        result,hp,rp=self.metrics()
        self.assertEqual(set(result),{'1','2','3'})
        for seed,metrics in result.items():
            primary=metrics['primary_same_prompt'];head=primary['evaluations']['full_head/native']
            self.assertEqual((head['n_items'],head['n_events'],head['n_strata']),(902,8,9))
            self.assertEqual(head['macro_ba'],1.)
            self.assertEqual(primary['contrasts']['full_minus_head']['ci95_delta'],[0.,0.])
            self.assertIsNone(metrics['paired_source']['landslide']['contrasts']['full_minus_head']['ci95_delta'])
        self.assertEqual(rp['1|flood']['n'],1371)
        self.assertEqual(hp['3|landslide']['n'],384)
    def test_stratum_then_event_equal_weight_not_sample_pool(self):
        data=copy.deepcopy(self.data)
        for row in data[4]:
            if row['id'] in ('flood.1.0:pos','hard.1.0:hard_neg'):
                row['logit']*=-1;row['probability']=1/(1+math.exp(-row['logit']));row['prediction']='yes' if row['logit']>=0 else 'no'
        result,_,_=self.metrics(data)
        for seed in ('1','2','3'):
            prim=result[seed]['primary_same_prompt']
            self.assertEqual(prim['evaluations']['full_head/native']['events']['1']['ba'],.5)
            self.assertEqual(prim['evaluations']['full_head/native']['macro_ba'],.9375)
            self.assertEqual(prim['contrasts']['full_minus_head']['delta'],.0625)
            self.assertEqual(prim['contrasts']['full_minus_head']['ci95_delta'],[0.,.1875])
    def test_eventsize_imbalance_and_seed_separation(self):
        data=copy.deepcopy(self.data)
        for row in data[4]:
            if row['seed']==3 and row['phen']=='landslide' and row['cluster']=='1':
                row['logit']*=-1;row['probability']=1/(1+math.exp(-row['logit']));row['prediction']='yes' if row['logit']>=0 else 'no'
        result,_,_=self.metrics(data)
        self.assertEqual(result['3']['paired_source']['landslide']['evaluations']['full_head/native']['macro_ba'],.5)
        self.assertEqual(result['1']['paired_source']['landslide']['evaluations']['full_head/native']['macro_ba'],1.)
        self.assertNotIn('mean',result)
    def test_unparsed_is_neither_true_negative_nor_false_positive(self):
        data=copy.deepcopy(self.data)
        row=next(x for x in data[5] if x['seed']==1 and x['id']=='hard.1.0:hard_neg')
        row.update(answer_raw='uncertain',parsed=None)
        result,_,rp=self.metrics(data)
        prim=result['1']['primary_same_prompt']['evaluations']['full/native']
        special=next(x for x in prim['events']['1']['strata'] if x['dates'][0]=='2021-02-01')
        self.assertEqual((special['specificity'],special['fpr'],special['ba']),(0.,0.,.5))
        self.assertEqual(prim['macro_ba'],.96875)
        self.assertEqual(rp['1|flood']['failed'],1)
    def test_parse_limit_by_seed_phenomenon(self):
        data=copy.deepcopy(self.data)
        rows=[r for r in data[5] if r['seed']==2 and r['phen']=='landslide']
        for row in rows[:3]:row.update(answer_raw='uncertain',parsed=None)
        audit.predictions(data[5],data[0],False)
        rows[3].update(answer_raw='uncertain',parsed=None)
        with self.assertRaisesRegex(ValueError,'parse failure'):audit.predictions(data[5],data[0],False)
    def test_exact_reference_identity_arm_gold_and_globalindex(self):
        for field,value in [('model_arm','pair'),('eval_arm','full_no_delta'),('seed',True),('id','train.flood.0:pos'),
                            ('pair_index',0),('source_gold','no'),('cluster','wrong'),('transformed_gold','yes')]:
            with self.subTest(field=field):
                rows=copy.deepcopy(self.data[5]);rows[0][field]=value
                with self.assertRaises(ValueError):audit.predictions(rows,self.data[0],False)
        rows=copy.deepcopy(self.data[5]);rows[-1]=rows[0]
        with self.assertRaisesRegex(ValueError,'Duplicate'):audit.predictions(rows,self.data[0],False)
    def test_logit_zero_and_probability_finiteness(self):
        rows=copy.deepcopy(self.data[4]);rows[0].update(logit=-0.,probability=.5,prediction='yes')
        audit.predictions(rows,self.data[0],True)
        for patch in [dict(prediction='no'),dict(probability=.5001),dict(logit=float('inf')),dict(logit=True)]:
            altered=copy.deepcopy(rows);altered[0].update(patch)
            with self.assertRaises(ValueError):audit.predictions(altered,self.data[0],True)
    def test_population_missingclass_trainleakage_and_order_rejected(self):
        for case in ('stratum','order','event','batch','primary'):
            data=copy.deepcopy(self.data)
            if case=='stratum':next(x for x in data[0] if x['id']=='hard.1.0:hard_neg')['dates']=['2030-01-01','2030-01-13']
            if case=='order':data[2]['train'].reverse()
            if case=='event':
                for x in data[0]:
                    if x['phen']=='flood' and x['partition']=='train' and x['cluster']=='train.0':x['cluster']='0';x['event']='0'
            if case=='batch':data[3]['1'][0][0][0]=data[2]['test'][0]
            if case=='primary':data[1]['primary_same_prompt'].pop()
            with self.subTest(case=case),self.assertRaises(ValueError):audit.population(*data[:4])
    def test_bootstrap_known_sign_and_support(self):
        self.assertEqual(audit.event_interval([-.5]*8),[-.5,-.5])
        self.assertEqual(audit.event_interval([0,.5,0,0,0,0,0,0]),[0.,.1875])
        with self.assertRaises(ValueError):audit.event_interval([0.]*24)
    def test_metadata_known_dates_float32_and_labels_excluded(self):
        item={'phen':'flood','dates':['2000-01-01','2100-12-31'],'answer':'unread','id':'unread'}
        actual=audit.metadata_values([item])
        np.testing.assert_array_equal(actual,np.asarray([[1,0,0,0,0,1,1,1]],dtype=np.float32))
        for dates in [['2020-01-02','2020-01-01'],['2020-1-1','2020-01-02'],['2021-02-29','2021-03-01']]:
            with self.assertRaises(ValueError):audit.metadata_values([dict(item,dates=dates)])
    def test_step1590_exposure12702_order_and_weighted_epoch_loss(self):
        items,sets,ordered,batches,_,_=self.data;index={x['id']:j for j,x in enumerate(items)}
        rows=[];step=0;seen=0
        for epoch,ep in enumerate(batches['1']):
            for batch_index,ids in enumerate(ep):
                step+=1;seen+=len(ids)
                rows.append(dict(seed=1,step=step,epoch=epoch,batch_index=batch_index,ids=ids,pair_indices=[index[x] for x in ids],batch_size=len(ids),exposures=seen,loss=3. if len(ids)==2 else 1.,loss_finite=True,gradients_finite=True,parameters_finite=True,elapsed_s=float(step)))
        values=audit.check_steps(rows,batches['1'],index,1)
        self.assertEqual(values,[(4232+6)/4234]*3)
        for field,value in [('exposures',0),('seed',True),('ids',['wrong']),('parameters_finite',False),('loss',float('nan'))]:
            bad=copy.deepcopy(rows);bad[1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):audit.check_steps(bad,batches['1'],index,1)
    def test_optimizer_fixed_values(self):
        group={'lr':1e-4,'weight_decay':.01,'betas':[.9,.999],'eps':1e-8,'foreach':False,'fused':False,'amsgrad':False,'maximize':False}
        audit.check_optimizer([group])
        with self.assertRaises(ValueError):audit.check_optimizer([dict(group,lr=.001)])
    def test_file_hash_pin_recheck_and_path_escape(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/'input.json';path.write_text('{"n":1}')
            files=audit.Files();pin=audit.digest(path);self.assertEqual(files.obj(path,pin),{'n':1})
            with self.assertRaises(ValueError):files.track(path,'0'*64)
            for name in ('../input.json','/etc/passwd'):
                with self.assertRaises(ValueError):audit.safe_file(root,name)
            path.write_text('{"n":2}')
            with self.assertRaises(ValueError):files.recheck()
    def test_duplicate_json_nonfinite_and_compare_shape_fail(self):
        for text in ('{"x":1,"x":2}','{"x":NaN}'):
            with self.assertRaises(ValueError):audit.decode(text)
        for expected,actual in [({'x':1},{'x':1,'y':2}),([1],[1,2]),(True,1),(1,True),(1.,float('nan'))]:
            with self.assertRaises(ValueError):audit.compare(expected,actual)
    def test_cpu_canonical_header_is_littleendian_sorted(self):
        import torch
        state={'z':torch.tensor([1.,2.],dtype=torch.float32),'a':torch.tensor([3],dtype=torch.int64)}
        expected=hashlib.sha256()
        for name,dtype,shape,raw in [('a','torch.int64',[1],struct.pack('<q',3)),('z','torch.float32',[2],struct.pack('<ff',1.,2.))]:
            header=json.dumps([name,dtype,shape],separators=(',',':')).encode()
            expected.update(struct.pack('<Q',len(header))+header+struct.pack('<Q',len(raw))+raw)
        self.assertEqual(audit.canonical_tensor_hash(state),expected.hexdigest())
        self.assertEqual(audit.canonical_tensor_hash(dict(reversed(list(state.items())))),expected.hexdigest())
        state['z'][0]=float('nan')
        with self.assertRaises(ValueError):audit.canonical_tensor_hash(state)
    def test_checkpoint_shape_parametercount_and_fixedbuffers(self):
        import torch
        schema=audit.checkpoint_schema()
        self.assertEqual(sum(math.prod(shape) for name,shape in schema.items() if name not in ('positions','token_types')),367361)
        state={name:torch.zeros(shape,dtype=torch.int64 if name=='token_types' else torch.float32) for name,shape in schema.items()}
        state['token_types']=torch.tensor([0]*64+[1]*64+[3]*64,dtype=torch.int64)
        grid=[]
        for row in range(8):
            for col in range(8):
                grid.append([v for coord in (row,col) for k in range(32) for v in (math.sin(coord/10000**(k/32)),math.cos(coord/10000**(k/32)))])
        state['positions']=torch.tensor(grid*3,dtype=torch.float32)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'head.pt';torch.save(state,path)
            pin=audit.canonical_tensor_hash(state);self.assertEqual(audit.check_checkpoint(path,pin,True),pin)
            state['token_types'][0]=2;torch.save(state,path)
            with self.assertRaisesRegex(ValueError,'token types'):audit.check_checkpoint(path,audit.canonical_tensor_hash(state),True)
            state['token_types'][0]=0;state['cls'][0]=1;torch.save(state,path)
            with self.assertRaisesRegex(ValueError,'CLS'):audit.check_checkpoint(path,audit.canonical_tensor_hash(state),True)
    def test_completed_audit_refuses_unpinned_or_failed_input_before_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.assertRaisesRegex(ValueError,'Explicit frozen'):audit.audit(root,'invalid')
            (root/'failure.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'failed artifact'):audit.audit(root,'0'*64)


if __name__=='__main__':unittest.main()
