"""Local pinned weights + train-only OE11 packets; no download or generation."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import torch
from training_core import require, sha


def verify_tree(root, expected):
    root = Path(root).resolve()
    require(type(expected) is dict and bool(expected), 'Missing model/source identity')
    actual = {}
    for name, digest in expected.items():
        path = (root / name).resolve()
        require(path.is_relative_to(root), 'Model identity path escapes root')
        actual[name] = sha(path)
    require(actual == expected, 'Actual model/source files differ from pinned identity')
    return actual


class ActualBackend:
    def __init__(self, protocol, cases, arm):
        paths = {name: Path(value) for name, value in protocol['paths'].items()}
        expected = json.loads(paths['model_identity'].read_text())
        qwen = paths['qwen_weights']; eo = paths['eo_weights']
        discovered = {str(p.relative_to(qwen)) for p in qwen.rglob('*') if p.is_file()
                      and p.suffix in {'.json', '.jinja', '.safetensors', '.txt'}}
        require(discovered == set(expected['reader_files_sha256']), 'Reader/tokenizer identity coverage mismatch')
        reader_identity = {'files': verify_tree(qwen, expected['reader_files_sha256'])}
        source_identity = verify_tree(paths['eo_source'], expected['eo_source_files_sha256'])
        require({'olmoearth_pretrain/model_loader.py', 'olmoearth_pretrain/data/normalize.py',
                 'olmoearth_pretrain/data/constants.py'} <= set(source_identity), 'EO normalization source pins missing')
        eo_identity = {'files': verify_tree(eo, expected['eo_files_sha256']), 'source': source_identity,
                       'prepared_manifest_sha256': sha(paths['prepared'] / 'manifest.jsonl'),
                       'prepared_normalization': 'Pinned precomputed arrays checked by loader'}
        sys.path[:0] = [str(paths['matching_code']), str(paths['loader_code']),
                        str(paths['eo_source']), str(paths['deps'])]
        from episode_adapter import make_episode_model_class
        from loader_adapter import make_loader_class
        Model = make_episode_model_class(paths['text_mask_code'])
        from contracts import load_base, validate_context
        native = load_base('native_replay')
        require(expected['eo_files_sha256'] == {'config.json': native.PINNED_PUBLIC_CONFIG_SHA256,
                                                'weights.pth': native.PINNED_WEIGHTS_SHA256},
                'Unexpected original OlmoEarth checkpoint')
        Loader = make_loader_class(paths['loader_code'] / 'frozen_episode_loader.py')
        self.loader = Loader(paths['prepared'], paths['episodes'], 'train', support_mode='pooled')
        self.catalog = self.loader.episodes
        self.contexts = json.loads(paths['contexts'].read_text())
        self.allowed = set(cases['episode_ids'])
        for eid in self.allowed:
            require(eid in self.catalog and self.catalog[eid]['split'] == 'train', 'Case outside train catalog')
            for condition in ('names_only', 'matched_knowledge'):
                validate_context(self.contexts[self.catalog[eid]['pair_id'] + ':' + condition])
        from olmoearth_pretrain.model_loader import load_model_from_path
        official = load_model_from_path(str(eo)); encoder = official.encoder; del official
        self.model = Model(encoder, arm, 'cuda:0', qwen, gradient_checkpointing=True,
                           reader_identity=reader_identity, eo_identity=eo_identity,
                           max_text_tokens=protocol['max_text_tokens'],
                           max_text_cache_entries=protocol['max_text_cache_entries'])
        self.identity = dict(reader=reader_identity, eo=eo_identity, loader=self.loader.adapter_audit())

    def fetch(self, eid, condition):
        require(eid in self.allowed, 'Worker cannot fetch an unregistered case')
        require(condition in ('names_only', 'matched_knowledge'), 'Unknown context condition')
        episode = self.catalog[eid]
        item = self.loader.load(eid, acquired_positions=[2, 5])
        target = self.loader.load_target(eid, purpose='training', training=True)
        context = self.contexts[episode['pair_id'] + ':' + condition]
        return item['model_input'], target, context, episode['k_pairs']
