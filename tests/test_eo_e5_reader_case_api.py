"""Optional saved E5 responses: same question, all conditions, no label replacement."""
import copy
from datetime import datetime
import unittest
import test_eo_e4_reader_case_api as previous

ARMS=('full/native','pair/native','later/native','delta/native','full/full_no_delta')

class E5ReaderCaseApiTests(previous.E4ReaderCaseApiTests):
    def attach_e5(self):
        c=self.payload['cases'][self.tile]
        c['e5_control']={'run_id':'E5-EB-v0','question_id':c['question_id'],
            'source_role':'historical_model_output','source_gold':c['reference_label'],
            'dates':list(c['dates']),'slots':list(c['slots']),
            'primary_membership':c['c1_membership']['supported_stratum'],
            'completed_at':'2026-09-25T07:47:56+00:00',
            'predictions':{arm:{'1':'yes','2':None,'3':'no'} for arm in ARMS},
            'source_sha256':{'audited_predictions':'a'*64,'independent_audit':'b'*64}}
        return c['e5_control']

    def test_e5_preserves_e2_e3_e4_source_gold_and_missing_parse(self):
        self.attach_e3();self.attach_e4()
        before=copy.deepcopy(self.payload['cases'][self.tile]);e=self.attach_e5()
        e['predictions']['full/native']['1']='no'
        e['predictions']['pair/native']['1']='yes'
        self.save();response,status=self.read();self.assertEqual(status,200)
        c=response['case'];self.assertEqual({k:v for k,v in c.items() if k!='e5_control'},before)
        self.assertEqual(c['e5_control'],e);self.assertEqual(c['reference_label'],'yes')
        self.assertIsNone(e['predictions']['full/native']['2'])

    def test_e5_negative_source_still_has_all_five_evaluations(self):
        c=self.payload['cases'][self.tile];c['reference_label']='no';c['question_id']=self.tile+'_q1_hard'
        e=self.attach_e5();self.save();response,status=self.read()
        self.assertEqual(status,200);self.assertEqual(set(response['case']['e5_control']['predictions']),set(ARMS))
        self.assertEqual(response['case']['reference_label'],'no')
        self.assertEqual(response['case']['e5_control']['predictions']['full/native']['1'],'yes')

    def test_e5_null_or_partial_structure_rejected(self):
        e=self.attach_e5();original=copy.deepcopy(self.payload)
        for value in (None,{},'absent'):
            self.payload=copy.deepcopy(original);self.payload['cases'][self.tile]['e5_control']=value;self.assert_invalid()
        for arm in ARMS:
            self.payload=copy.deepcopy(original);del self.payload['cases'][self.tile]['e5_control']['predictions'][arm];self.assert_invalid()

    def test_e5_identity_gold_pair_and_membership_must_match(self):
        self.attach_e5();original=copy.deepcopy(self.payload)
        for key,value in [('run_id','E5'),('source_role','live_prediction'),('source_gold','no'),
                          ('question_id','wrong'),('dates',['2020-01-01','2020-02-01']),
                          ('slots',['post','pre_2']),('primary_membership',True),
                          ('primary_membership',None),('primary_membership',0)]:
            with self.subTest(key=key,value=value):
                self.payload=copy.deepcopy(original);self.payload['cases'][self.tile]['e5_control'][key]=value;self.assert_invalid()

    def test_e5_included_primary_must_be_boolean_true(self):
        self.payload['cases'][self.tile]['c1_membership']['supported_stratum']=True
        e=self.attach_e5();self.save();self.assertEqual(self.read()[1],200)
        e['primary_membership']=1;self.assert_invalid()

    def test_e5_three_seeds_all_classes_are_strict(self):
        self.attach_e5();original=copy.deepcopy(self.payload)
        for values in ({'1':'yes','2':'no'},{'1':'yes','2':'no','4':'no'},
                       {'1':'YES','2':'no','3':None},{'1':0,'2':'no','3':None},
                       {'1':'yes','2':'no','3':'maybe'},None,['yes','no',None]):
            self.payload=copy.deepcopy(original);self.payload['cases'][self.tile]['e5_control']['predictions']['pair/native']=values;self.assert_invalid()

    def test_e5_completion_time_and_sha_required(self):
        self.attach_e5();original=copy.deepcopy(self.payload)
        for key,value in [('completed_at',None),('completed_at','2026-09-25'),('completed_at','bad'),
                          ('source_sha256',{}),('source_sha256',{'a':'z'*64}),('source_sha256',None)]:
            self.payload=copy.deepcopy(original);self.payload['cases'][self.tile]['e5_control'][key]=value;self.assert_invalid()

if __name__=='__main__':unittest.main()
