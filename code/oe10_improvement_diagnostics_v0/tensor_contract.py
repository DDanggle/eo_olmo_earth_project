"""Synthetic NumPy tensor contract only: no torch, EO/Qwen weights, raw imagery or score.

Checks dated/object support preservation and a minimal conditional matching
operator. Fixed coefficients below are arbitrary wiring-test constants, not
proposed trained parameters or claimed performance settings.
"""
import datetime as dt
import hashlib,json
from pathlib import Path
import unittest
import numpy as np


def keep_dates(native):
    if native.ndim != 6 or native.shape[0] != 1 or native.shape[1:3] != (32,32) or native.shape[4] != 1:
        raise ValueError('Expected B=1,H=32,W=32,T,bandset=1,D')
    return native[:,:,:,:,0,:]


def dated_prototype(grid, original_mask):
    if grid.ndim!=5 or grid.shape[0]!=1 or grid.shape[1:3]!=(32,32):raise ValueError('Dated grid shape')
    if original_mask.shape!=(128,128) or original_mask.dtype!=np.bool_:raise ValueError('Original boolean object mask')
    weight=original_mask.reshape(32,4,32,4).mean((1,3))
    if weight.sum()<=0:raise ValueError('Empty support object')
    return np.einsum('bhwtd,hw->btd',grid,weight)/weight.sum()


def phase(timestamps):
    # Existing native convention is [day, zero-based month, year], not y/m/d.
    if timestamps.ndim!=2 or timestamps.shape[1]!=3:raise ValueError('Timestamps shape')
    out=[]
    for day,month0,year in timestamps:
        date=dt.date(int(year),int(month0)+1,int(day))
        start=dt.date(int(year),1,1);days=(dt.date(int(year)+1,1,1)-start).days
        theta=2*np.pi*(date-start).days/days
        out.append((np.sin(theta),np.cos(theta)))
    return np.asarray(out)


def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def match(query,support,text,query_phase,support_phase,valid):
    # query[P,Tq,D]; support[2,K,Ts,D]; one content vector for each semantic role.
    if query.ndim!=3 or support.ndim!=4 or support.shape[0]!=2:raise ValueError('Feature contract')
    if query.shape[-1]!=support.shape[-1] or text.shape!=(2,query.shape[-1]):raise ValueError('Dimension mismatch')
    if valid.shape!=support.shape[:-1] or valid.dtype!=np.bool_:raise ValueError('Support validity contract')
    if not valid.reshape(2,-1).any(1).all():raise ValueError('Every role needs a valid observed support token')
    if query_phase.shape!=(query.shape[1],2) or support_phase.shape!=support.shape[:-1]+(2,):raise ValueError('Date contract')
    conditioned=unit(query[:,:,None,:]+.3*text[None,None,:,:])
    logits=np.einsum('ptrd,rksd->ptrks',conditioned,unit(support))
    logits=logits+.1*np.einsum('tf,rksf->trks',query_phase,support_phase)[None]
    logits=np.where(valid[None,None,:,:,:],logits,-np.inf)
    weight=np.exp(logits-np.max(logits,axis=(-2,-1),keepdims=True))
    weight/=weight.sum(axis=(-2,-1),keepdims=True)
    matched=np.einsum('ptrks,rksd->ptrd',weight,support)
    return matched,weight


class Contracts(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(280928)
    def data(self,k=8):
        q=self.rng.normal(size=(1024,2,16));s=self.rng.normal(size=(2,k,8,16));c=self.rng.normal(size=(2,16))
        qp=phase(np.array([[25,0,2019],[4,6,2019]]));sp=np.broadcast_to(phase(np.array([[1,i,2019] for i in range(8)])),(2,k,8,2)).copy()
        v=np.ones((2,k,8),dtype=bool)
        return q,s,c,qp,sp,v
    def test_old_date_pool_is_exact_reduction_of_new_grid(self):
        x=self.rng.normal(size=(1,32,32,2,1,16));new=keep_dates(x)
        self.assertEqual(new.shape,(1,32,32,2,16))
        np.testing.assert_allclose(new.mean(3),x.mean((3,4)),rtol=0,atol=0)
    def test_fractional_mask_perdate_prototype_reconstructs_old_average(self):
        x=self.rng.normal(size=(1,32,32,8,1,16));m=np.zeros((128,128),dtype=bool);m[3:69,5:86]=True
        w=m.reshape(32,4,32,4).mean((1,3));new=dated_prototype(keep_dates(x),m)
        old=np.einsum('bhwd,hw->bd',x.mean((3,4)),w)/w.sum()
        self.assertEqual(new.shape,(1,8,16));np.testing.assert_allclose(new.mean(1),old,rtol=1e-12,atol=1e-12)
    def test_flatten_preserves_role_object_date_identity(self):
        tagged=np.empty((2,8,8,1))
        for r in range(2):
            for k in range(8):
                for t in range(8):tagged[r,k,t,0]=100*r+10*k+t
        flat=tagged.reshape(2,64,1)
        for r in range(2):
            for k in range(8):
                for t in range(8):self.assertEqual(flat[r,k*8+t,0],100*r+10*k+t)
    def test_mean_can_hide_opposite_temporal_signatures(self):
        a=np.array([[1.,0.],[0.,1.]]);b=a[::-1]
        np.testing.assert_array_equal(a.mean(0),b.mean(0))
        self.assertFalse(np.array_equal(a[1]-a[0],b[1]-b[0]))
    def test_matching_uses_all_k_and_condition_changes_weights(self):
        for k in (1,2,4,8):
            q,s,c,qp,sp,v=self.data(k);out,w=match(q,s,c,qp,sp,v)
            self.assertEqual(out.shape,(1024,2,2,16));self.assertEqual(w.shape,(1024,2,2,k,8))
            np.testing.assert_allclose(w.sum((-2,-1)),1,rtol=1e-12)
            out0,w0=match(q,s,np.zeros_like(c),qp,sp,v)
            self.assertGreater(float(np.max(abs(out-out0))),1e-6)
            self.assertGreater(float(np.max(abs(w-w0))),1e-6)
    def test_object_permutation_consistent_with_metadata_is_invariant(self):
        q,s,c,qp,sp,v=self.data();perm=np.array([3,6,7,0,4,2,1,5])
        a,_=match(q,s,c,qp,sp,v);b,_=match(q,s[:,perm],c,qp,sp[:,perm],v[:,perm])
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
    def test_date_permutation_moves_feature_and_metadata_together(self):
        q,s,c,qp,sp,v=self.data();perm=np.array([3,6,7,0,4,2,1,5])
        a,_=match(q,s,c,qp,sp,v);b,_=match(q,s[:,:,perm],c,qp,sp[:,:,perm],v[:,:,perm])
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
        wrong,_=match(q,s[:,:,perm],c,qp,sp,v)
        self.assertGreater(float(np.max(abs(a-wrong))),1e-6)
    def test_role_swap_changes_role_output_not_crossmix(self):
        q,s,c,qp,sp,v=self.data();a,_=match(q,s,c,qp,sp,v);b,_=match(q,s[::-1],c[::-1],qp,sp[::-1],v[::-1])
        np.testing.assert_allclose(a[:,:,::-1],b,rtol=1e-12,atol=1e-12)
    def test_invalid_support_tokens_get_zero_weight_empty_role_rejected(self):
        q,s,c,qp,sp,v=self.data();v[0,0,:]=False;_,w=match(q,s,c,qp,sp,v)
        self.assertTrue(np.all(w[:,:,0,0,:]==0));v[0]=False
        with self.assertRaises(ValueError):match(q,s,c,qp,sp,v)
    def test_calendar_convention_leapyear_and_cyclic_boundary(self):
        p=phase(np.array([[1,0,2019],[31,11,2019]]));np.testing.assert_allclose(p[0],[0,1],atol=1e-12)
        self.assertLess(np.linalg.norm(p[0]-p[1]),.02)
        leap=phase(np.array([[29,1,2020]]));self.assertTrue(np.isfinite(leap).all())
        with self.assertRaises(ValueError):phase(np.array([[29,1,2019]]))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contracts)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    report={'scope':'Synthetic NumPy tensor/index contracts only; no model execution, actual imagery, gradients, learning or score measured',
      'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'query_dates':2,'support_dates_per_object':8,'max_k_pairs':8,'new_observations':0,
      'query_shape':['P=1024','Tq=2','D'],'support_shape':['role=2','K','Ts=8','D'],
      'max_attention_scores':1024*2*2*8*8,'max_attention_scores_fp32_bytes':1024*2*2*8*8*4,
      'limits':'Mean-erasure example is constructed features, not evidence native OlmoEarth loses its encoded temporal information.'}
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
