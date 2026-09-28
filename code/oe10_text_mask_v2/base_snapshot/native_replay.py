"""Pinned official v1.2 replay on existing native train64, not full CPT reproduction.

Factory imports model dependencies lazily. Full model state retains the frozen
EMA=(1,1) target and decoder/projector. Never substitutes a second online encoder.
"""
from __future__ import annotations
import ast
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import numpy as np

PINNED_PUBLIC_CONFIG_SHA256 = "0d531a67ad3e477e7011efabcceb01ed80f430aa0a0a3d344fe18cec0f229b8a"
PINNED_V12_RECIPE_SHA256 = "d88b23daca3d8a5657ede3ccbe98ce7e2ea6cdbf0e7b026b24fd15e61205f9ee"


def official_recipe_loss_configs(recipe_path, modality_cls):
    """Read exact loss dictionaries from the pinned official recipe without executing it."""
    tree = ast.parse(recipe_path.read_text())
    assignments = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
                   for target in node.targets if isinstance(target, ast.Name)}

    def literal(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.List, ast.Tuple)):
            values = [literal(child) for child in node.elts]
            return values if isinstance(node, ast.List) else tuple(values)
        if isinstance(node, ast.Dict):
            return {literal(k): literal(v) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.Name) and node.id == "ONLY_DECODE_MODALITIES":
            return literal(assignments[node.id])
        if (isinstance(node, ast.Attribute) and node.attr == "name" and
                isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name) and
                node.value.value.id == "Modality"):
            return getattr(modality_cls, node.value.attr).name
        raise ValueError(f"Unsupported official recipe expression: {ast.dump(node)}")

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_train_module_config"]
    if len(functions) != 1:
        raise ValueError("Expected exactly one official build_train_module_config")
    constructors = [node.value for node in functions[0].body if isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name) and node.value.func.id == "ContrastiveLatentMIMTrainModuleConfig"]
    if len(constructors) != 1:
        raise ValueError("Expected exact official train module config constructor")
    result = {}
    for keyword in constructors[0].keywords:
        if keyword.arg not in ["loss_config", "contrastive_config"]:
            continue
        call = keyword.value
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "LossConfig":
            raise ValueError("Expected official LossConfig constructor")
        payloads = [k.value for k in call.keywords if k.arg == "loss_config"]
        if len(payloads) != 1:
            raise ValueError("Expected one official loss_config payload")
        result[keyword.arg] = literal(payloads[0])
    if set(result) != {"loss_config", "contrastive_config"}:
        raise ValueError("Incomplete official loss recipe")
    return result


def bridge_public_loss_registry_keys(train_config, config_sha256, source_root, modality_cls):
    """Restore only serialized-away registry keys with an exact pinned-source match.

    Official LossConfig.build mutates its input with pop('type'), and the official
    experiment writes config after constructing the train module. _CLASS_ names
    the Config dataclass, not the missing registered loss type.
    """
    recipe_path = source_root / "scripts/official/v1_2/base.py"
    recipe_hash = sha256(recipe_path)
    if recipe_hash != PINNED_V12_RECIPE_SHA256:
        raise ValueError("Public loss bridge requires the reviewed pinned v1.2 recipe SHA")
    expected = official_recipe_loss_configs(recipe_path, modality_cls)
    normalized = copy.deepcopy(train_config)
    changes = []
    for role in ["loss_config", "contrastive_config"]:
        observed = normalized[role]["loss_config"]
        if "type" not in observed:
            if config_sha256 != PINNED_PUBLIC_CONFIG_SHA256:
                raise ValueError("Missing loss registry key in an unreviewed checkpoint config; refusing inference")
            expected_parameters = {k: v for k, v in expected[role].items() if k != "type"}
            if observed != expected_parameters:
                raise ValueError(f"Public {role} parameters differ from pinned official recipe")
            observed["type"] = expected[role]["type"]
            changes.append({"path": f"train_module.{role}.loss_config.type", "from": "absent",
                            "to": expected[role]["type"], "evidence": str(recipe_path)})
        elif observed != expected[role]:
            raise ValueError(f"Explicit {role} differs from this pinned v1.2 objective probe")
    return normalized, {"source_recipe_sha256": recipe_hash, "checkpoint_config_sha256": config_sha256,
                        "changes": changes, "official_recipe_loss_configs": expected,
                        "scope": "Memory-only restore of registry keys; public config and weights are not edited",
                        "cause": "Official LossConfig.build pops type before experiment serializes the built train configuration"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


PINNED_MANIFEST_SHA256 = '54d12227d7319aec7aa0eefccc3d65e07039b6784134fe47d27f69b6b1b25e4e'
PINNED_WEIGHTS_SHA256 = '57f7b66faf206db1307670673839e639d3a19c305f6ad968c62392ad3e88deec'
PINNED_DATA_ROOT = Path('/home/work/data/olmoearth/oe4_native_v12_v0/data/official_subset_1k')
PINNED_SOURCE_HASHES = {
    'olmoearth_pretrain/model_loader.py': '2ff982cda47fa53e8bb3a9f4ff3879816e4297770ed6e6c6cf49f5e323562ba7',
    'olmoearth_pretrain/train/masking.py': '4a29589d46fd0920912dfef8e46c8cf2c614545417841325c567ca55a5e27dba',
    'olmoearth_pretrain/train/loss.py': '7f378e156eeb65434fddaf3c63300bb0c426b4532d6b987d753365ab762997b5',
    'scripts/official/v1_2/base.py': PINNED_V12_RECIPE_SHA256,
}
BRIDGE_REUSED_FROM_RUNTIME_SHA256 = 'b737a7254c2d66518e3acc8b8314918c8529d21d33e04347b49be6029cac0b2c'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_train_manifest(path):
    require(sha256(path) == PINNED_MANIFEST_SHA256, 'native manifest is not pinned train64/dev8 receipt')
    data = json.loads(Path(path).read_text())
    selected = data['selected']
    require(len(selected) == 72, 'expected native train64 + held-out dev8')
    require(len({r['file'] for r in selected}) == 72
            and len({r['raw_crop_sha256'] for r in selected}) == 72, 'duplicate native file/raw crop')
    train = [r for r in selected if r['split'] == 'train_diagnostic']
    dev = [r for r in selected if r['split'] == 'dev_diagnostic']
    require(len(train) == 64 and len(dev) == 8, 'native role count mismatch')
    for row in selected:
        path = Path(row['file']).resolve()
        require(path.is_relative_to(PINNED_DATA_ROOT.resolve()) and path.suffix == '.h5', 'native path outside pinned data root')
        require(row['crop_yxhw'][2:] == [32,32] and row['compact_timesteps'] == [0,1]
                and len(row['timestamp_indices']) == 2, 'native crop/date contract changed')
        require(row['normalization'] == {'sentinel2_l2a':'computed','worldcover':'computed'}, 'native normalizer policy changed')
    return copy.deepcopy(train)


def batch_rows(train_rows, step):
    require(type(step) is int and step >= 0, 'step must be a nonnegative integer')
    require(len(train_rows) == 64 and all(r['split'] == 'train_diagnostic' for r in train_rows),
            'only native train64 can enter replay')
    start = (step * 2) % 64
    return train_rows[start:start+2]


def read_native_sample(row, h5py, normalizer, modality_cls, missing_value):
    """Read only the manifest's crop/compact-date positions, then verify raw identity."""
    require(row['split'] == 'train_diagnostic', 'dev row cannot enter replay reader')
    path = Path(row['file']).resolve()
    require(path.is_relative_to(PINNED_DATA_ROOT.resolve()), 'H5 path escapes pinned data')
    require(path.stat().st_size == row['file_bytes'], 'H5 size differs from frozen receipt')
    with h5py.File(path,'r') as f:
        s2=f['sentinel2_l2a'];wc=f['worldcover'];timestamps=np.asarray(f['timestamps'][()])
        require(list(s2.shape) == row['source_s2_shape'] and list(wc.shape) == row['source_worldcover_shape'],
                'native source shape mismatch')
        require(timestamps.ndim == 2 and timestamps.shape[1] == 3
                and np.isfinite(timestamps).all() and np.equal(timestamps,np.round(timestamps)).all(), 'timestamp values')
        require(((timestamps[:,0] >= 1)&(timestamps[:,0] <= 31)).all()
                and ((timestamps[:,1] >= 0)&(timestamps[:,1] <= 11)).all(), 'timestamp convention')
        if 'missing_timesteps_masks/sentinel2_l2a' in f:
            present=np.asarray(f['missing_timesteps_masks/sentinel2_l2a'][()]).reshape(-1)
            require(len(present) == len(timestamps) and np.isin(present,[0,1]).all(), 'presence mask')
            present_indices=np.flatnonzero(present)
        else:
            require(s2.shape[2] == len(timestamps), 'compact dates require presence mask')
            present_indices=np.arange(len(timestamps))
        require(len(present_indices) == s2.shape[2], 'compact presence count')
        compact=row['compact_timesteps'];indices=np.asarray(row['timestamp_indices'])
        require(compact == [0,1] and np.array_equal(present_indices[compact],indices), 'compact/original date index mismatch')
        y,x,h,w=row['crop_yxhw']
        require([h,w] == [32,32] and y >= 0 and x >= 0
                and y+h <= s2.shape[0] and x+w <= s2.shape[1], 'native crop bounds')
        raw_s2=np.asarray(s2[y:y+h,x:x+w,:2,:],dtype=np.float32)
        raw_wc=np.asarray(wc[y:y+h,x:x+w],dtype=np.float32)
        if raw_wc.ndim == 3 and raw_wc.shape[-1] == 1: raw_wc=raw_wc[:,:,None,:]
        require(raw_s2.shape == (32,32,2,12) and raw_wc.shape == (32,32,1,1), 'selected native array shape')
        ts=timestamps[indices].astype(np.int64)
        require(ts.tolist() == row['timestamps_day_month0_year'], 'selected native dates mismatch')
        raw_hash=hashlib.sha256(raw_s2.tobytes()+raw_wc.tobytes()+ts.tobytes()).hexdigest()
        require(raw_hash == row['raw_crop_sha256'], 'native raw crop hash mismatch')
        require(np.isfinite(raw_s2).all() and np.isfinite(raw_wc).all()
                and not (raw_s2 == missing_value).any() and not (raw_wc == missing_value).any(), 'missing/nonfinite native crop')
        require((raw_s2 != 0).any(), 'all-zero native crop')
        if row.get('latlon') is not None:
            require('latlon' in f and np.array_equal(np.asarray(f['latlon'][()]),row['latlon']), 'native coordinate mismatch')
    sample={'sentinel2_l2a':np.asarray(normalizer.normalize(modality_cls.get('sentinel2_l2a'),raw_s2),dtype=np.float32),
            'worldcover':np.asarray(normalizer.normalize(modality_cls.get('worldcover'),raw_wc),dtype=np.float32),
            'timestamps':ts}
    require(all(np.isfinite(a).all() for a in sample.values()), 'nonfinite normalized native input')
    return sample


class NativeReplay:
    def __init__(self, model, train_rows, samples, masker, base_loss, contrastive,
                 train_config, device, torch, sample_cls, mask_value_cls, identity):
        self.model=model
        self.encoder=model.encoder
        self.train_rows=train_rows
        self._samples=samples
        self._masker=masker
        self._base_loss=base_loss
        self._contrastive=contrastive
        self._train_config=train_config
        self.device=device
        self._torch=torch
        self._sample_cls=sample_cls
        self._mask_value_cls=mask_value_cls
        self.identity=identity
        self.calls=0

    def state_dict(self):
        # Full official state, including fixed target + all trainable decoders/projectors.
        return self.model.state_dict()

    def load_state_dict(self, state, strict=True):
        return self.model.load_state_dict(state,strict=strict)

    def loss(self, step, seed):
        torch=self._torch
        require(type(seed) is int and 0 <= seed < 2**32, 'seed must be uint32')
        require(self.encoder is self.model.encoder, 'replay online encoder identity changed')
        require(not any(p.requires_grad for p in self.model.target_encoder.parameters()), 'official fixed target became trainable')
        require(self.model.target_encoder.patch_embeddings.band_dropout_rate == 0, 'target band dropout must stay disabled')
        selected=batch_rows(self.train_rows,step)
        indices=[(step*2)%64,(step*2)%64+1]
        b=self._sample_cls(**{k:torch.from_numpy(np.stack([self._samples[i][k] for i in indices])).to(self.device)
                             for k in ('sentinel2_l2a','worldcover','timestamps')})
        modes=[(m,m.training) for m in self.model.modules()]
        python_rng=random.getstate();numpy_rng=np.random.get_state()
        cuda_devices=[] if self.device.type != 'cuda' else [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        try:
            # Local replay masking/dropout RNG does not perturb the caller's training stream.
            with torch.random.fork_rng(devices=cuda_devices), torch.autocast(device_type=self.device.type,enabled=False):
                torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
                self.model.train()
                masked=[self._masker.apply_mask(b,patch_size=4) for _ in range(2)]
                losses,pooled,counts=[],[],[]
                for view in masked:
                    count={}
                    for name in ('sentinel2_l2a','worldcover'):
                        mask=getattr(view,name+'_mask')
                        count[name]={str(int(k)):int(n) for k,n in zip(*torch.unique(mask,return_counts=True))}
                    require(int((view.sentinel2_l2a_mask == self._mask_value_cls.ONLINE_ENCODER.value).sum()) > 0,
                            'native mask has no online S2 tokens')
                    require(int((view.worldcover_mask == self._mask_value_cls.ONLINE_ENCODER.value).sum()) == 0,
                            'decode-only WorldCover leaked into online input')
                    _,decoded,projected,_,_=self.model(view,4)
                    with torch.no_grad():
                        target=self.model.target_encoder(view.unmask(),patch_size=4,
                            token_exit_cfg=self._train_config['token_exit_cfg'])['tokens_and_masks']
                    with torch.autocast(device_type=self.device.type,enabled=False):
                        losses.append(self._base_loss.compute(decoded,target))
                    pooled.append(projected);counts.append(count)
                with torch.autocast(device_type=self.device.type,enabled=False):
                    base=(losses[0]+losses[1])/2
                    contrastive=self._contrastive.compute(pooled[0],pooled[1])
                    total=base+contrastive
                require(bool(torch.isfinite(total)), 'nonfinite native objective')
        finally:
            random.setstate(python_rng);np.random.set_state(numpy_rng)
            for module,mode in modes:module.training=mode
        self.calls+=1
        encoder_trainable=any(p.requires_grad for p in self.encoder.parameters())
        return total, {'total':float(total.detach()),'base':float(base.detach()),
            'contrastive':float(contrastive.detach()),'mask_counts':counts,
            'step':step,'seed':seed,'sample_indices':indices,
            'files':[r['file'] for r in selected],'raw_crop_sha256':[r['raw_crop_sha256'] for r in selected],
            'native_train_samples':2,'native_dev_samples':0,'native_model_forward_calls':2,
            'explicit_target_forward_calls':2,'online_patch_date_instances':8,
            'explicit_target_patch_date_instances':8,'batch_size':2,'crop_size':32,'timesteps':2,
            'encoder_trainable':encoder_trainable,'all_native_frozen':not any(p.requires_grad for p in self.model.parameters()),
            'loss_requires_grad':bool(total.requires_grad),'native_encoder_update_possible':encoder_trainable and bool(total.requires_grad),
            'fixed_target_ema_decay':[1.0,1.0], 'interpretation':'native objective replay; zero components retained, not repaired'}


def make_replay(source_root, deps_root, checkpoint_dir, manifest_path, device):
    source_root=Path(source_root).resolve();checkpoint_dir=Path(checkpoint_dir).resolve()
    source_hashes={p:sha256(source_root/p) for p in PINNED_SOURCE_HASHES}
    require(source_hashes == PINNED_SOURCE_HASHES, 'official objective source hashes changed')
    checkpoint_hashes={p:sha256(checkpoint_dir/p) for p in ('config.json','weights.pth')}
    require(checkpoint_hashes == {'config.json':PINNED_PUBLIC_CONFIG_SHA256,'weights.pth':PINNED_WEIGHTS_SHA256},
            'replay must initialize from pinned original public checkpoint')
    train_rows=load_train_manifest(manifest_path)
    if deps_root is not None:sys.path.insert(0,str(Path(deps_root).resolve()))
    sys.path.insert(0,str(source_root))
    import torch
    import h5py
    import hdf5plugin  # noqa: F401 -- required official H5 compression registration
    import olmoearth_pretrain
    from olmoearth_pretrain.data.constants import MISSING_VALUE,Modality
    from olmoearth_pretrain.data.normalize import Normalizer,Strategy
    from olmoearth_pretrain.datatypes import OlmoEarthSample,MaskValue
    from olmoearth_pretrain.model_loader import load_model_from_path
    from olmoearth_pretrain.train.loss import LossConfig
    from olmoearth_pretrain.train.masking import MaskingConfig
    require(Path(olmoearth_pretrain.__file__).resolve().is_relative_to(source_root),'wrong model package import')
    config=json.loads((checkpoint_dir/'config.json').read_text())
    tc,bridge=bridge_public_loss_registry_keys(config['train_module'],checkpoint_hashes['config.json'],source_root,Modality)
    require(tc.get('ema_decay') in ([1.0,1.0],(1.0,1.0)), 'only official fixed EMA target supported')
    require(not tc.get('mae_loss_config') and not tc.get('regularizer_config')
            and all(v == 0 for v in tc['token_exit_cfg'].values()), 'unsupported objective configuration')
    base=LossConfig(loss_config=copy.deepcopy(tc['loss_config']['loss_config'])).build()
    contrastive=LossConfig(loss_config=copy.deepcopy(tc['contrastive_config']['loss_config'])).build()
    normalizer=Normalizer(Strategy.COMPUTED)
    samples=[read_native_sample(r,h5py,normalizer,Modality,MISSING_VALUE) for r in train_rows]
    device=torch.device(device)
    model=load_model_from_path(str(checkpoint_dir)).to(device,dtype=torch.float32)
    disabled=[]
    for name,module in model.named_modules():
        if getattr(module,'use_flash_attn',False):module.use_flash_attn=False;disabled.append(name)
    model.encoder.enable_band_dropout()
    require(model.encoder.tokenization_config.get_num_bandsets('sentinel2_l2a') == 1,'S2 bandsets')
    require(not any(p.requires_grad for p in model.target_encoder.parameters()),'official target not frozen')
    require(model.target_encoder.patch_embeddings.band_dropout_rate == 0,'target band dropout')
    masker=MaskingConfig(strategy_config=copy.deepcopy(tc['masking_config']['strategy_config']),
                         tokenization_config=model.encoder.tokenization_config).build()
    identity={'manifest_sha256':PINNED_MANIFEST_SHA256,'source_hashes':source_hashes,
              'checkpoint_hashes':checkpoint_hashes,'public_loss_registry_bridge':bridge,
              'bridge_runtime_source_sha256':BRIDGE_REUSED_FROM_RUNTIME_SHA256,
              'train_files_loaded_and_raw_hash_verified':64,'dev_files_loaded':0,
              'normalization':'official computed once, S2 + WorldCover',
              'samples_cpu_cached':True,'cached_array_bytes':sum(a.nbytes for s in samples for a in s.values()),
              'precision':'FP32 native forward and losses; outer autocast explicitly disabled',
              'flash_attn_disabled_modules':disabled,'ema_decay':[1.0,1.0],
              'scope':'small native replay auxiliary; not full official pretraining/CPT reproduction',
              'geographic_independence_certified':False,'pretraining_exposure_unknown':True}
    return NativeReplay(model,train_rows,samples,masker,base,contrastive,tc,device,torch,OlmoEarthSample,MaskValue,identity)
