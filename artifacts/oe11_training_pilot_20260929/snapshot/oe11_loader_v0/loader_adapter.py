"""Read-only OE11 adapter over the immutable OE10 loader.

Two explicit validation aliases only: calibration -> dev_query, and the
partitioned support-mask path -> the old flat-name spelling. The original
manifest/catalog remain in memory and on disk for every actual read/hash.
"""
from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

PINNED_LOADER_SHA256 = '536de03b77bcd5bfedf439b5456d9e0fe60a3789c5f9c1a53c5511ded05d07f6'
SOURCE_PARENTS = frozenset(('t31tfj', 't31tfm', 't32ulu'))
PARTITIONS = {'train_pool': ('train', True), 'source_bank': ('train', False),
              'calibration': ('development', False)}


def make_loader_class(base_loader_path):
    base_loader_path = Path(base_loader_path).resolve()
    if hashlib.sha256(base_loader_path.read_bytes()).hexdigest() != PINNED_LOADER_SHA256:
        raise ValueError('Immutable base loader SHA mismatch')
    spec = importlib.util.spec_from_file_location('_oe11_frozen_input_loader', base_loader_path)
    base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(base)

    class CalibrationEpisodeLoader(base.EpisodeLoader):
        """Constructor-only validation view; no global guard or path monkeypatch.

        Cross-parent mode constrains TRAIN supports. Development uses the common
        global source bank in both modes; it is calibration, not an unseen region.
        """
        def __init__(self, prepared_root, episodes_root, split, *, support_mode=None):
            self.contract = json.loads((Path(episodes_root) / 'episode_contract.json').read_text())
            mode = self.contract.get('train_support_parent_policy')
            base.require(mode in ('pooled', 'cross_parent'), 'unknown support-parent policy')
            base.require(support_mode is None or support_mode == mode, 'requested/catalog mode mismatch')
            self.support_mode = mode
            base.require(Path(self.contract['prepared_input_reference_root']).resolve() ==
                         Path(prepared_root).resolve(), 'prepared reference root mismatch')
            base.require(Path(self.contract['support_mask_reference_root']).resolve() ==
                         Path(episodes_root).resolve(), 'mask reference root mismatch')
            base.require(set(self.contract['allowed_source_parents']) == SOURCE_PARENTS,
                         'source parent scope mismatch')
            base.require(self.contract['selected_classes'] == [1, 3, 8, 14], 'four-class scope mismatch')
            base.require(self.contract['final_holdout_t30uxv_included'] is False,
                         'held-out region forbidden')
            base.require(self.contract['bank_and_calibration_labels_allowed_for_encoder_gradient_training']
                         is False, 'target training boundary mismatch')
            self.validation_alias_count = 0
            super().__init__(prepared_root, episodes_root, split)
            # Validate even rows not referenced by the selected public catalog.
            for row in self._manifest.values():
                self._validate_manifest_row(row)

        @staticmethod
        def _validate_manifest_row(row):
            base.require(row['parent_tile'] in SOURCE_PARENTS, 'unknown/held-out manifest parent')
            partition = row['training_partition']
            base.require(partition in PARTITIONS, 'unexpected manifest partition')
            role, can_train = PARTITIONS[partition]
            base.require(row['role'] == role and row['supervised_training_allowed'] is can_train,
                         'original manifest role/training flag mismatch')

        def _validate_episode(self, episode):
            qid = episode['query_patch_id']
            base.require(qid in self._manifest, 'unknown query patch')
            original_query = self._manifest[qid]
            self._validate_manifest_row(original_query)
            expected_query = 'train_pool' if self.split == 'train' else 'calibration'
            expected_support = 'train_pool' if self.split == 'train' else 'source_bank'
            base.require(original_query['training_partition'] == expected_query,
                         'OE11 query partition violation')
            validation_episode = copy.deepcopy(episode)
            for actual_pair, validation_pair in zip(episode['support_pairs'],
                                                    validation_episode['support_pairs']):
                base.require(set(actual_pair) == {'positive', 'counterexample'}, 'support role keys')
                for role, support in actual_pair.items():
                    base.require(support['patch_id'] in self._manifest, 'unknown support patch')
                    source = self._manifest[support['patch_id']]
                    self._validate_manifest_row(source)
                    base.require(source['training_partition'] == expected_support,
                                 'OE11 support partition violation')
                    if self.split == 'train' and self.support_mode == 'cross_parent':
                        base.require(source['parent_tile'] != original_query['parent_tile'],
                                     'cross-parent TRAIN support violation')
                    leaf = support['object_key'].replace(':', '_') + '.npz'
                    expected = f'support_masks/{expected_support}/{leaf}'
                    base.require(support['mask_npz'] == expected, 'partitioned mask lineage mismatch')
                    base.safe_path(self.episodes_root, support['mask_npz'], 'support_masks')
                    validation_pair[role]['mask_npz'] = 'support_masks/' + leaf
            original_manifest = self._manifest
            validation_manifest = dict(original_manifest)
            if self.split == 'development':
                validation_manifest[qid] = dict(original_query, training_partition='dev_query')
            try:
                self._manifest = validation_manifest
                super()._validate_episode(validation_episode)
            finally:
                self._manifest = original_manifest
            self.validation_alias_count += 1

        def _target(self, episode, purpose):
            # Deliberately lazy: inference construction/load never reads scoring
            # or query label files. The base then checks each join and label SHA.
            scoring_path = self.episodes_root / 'scoring' / f'scoring_{self.split}.jsonl'
            expected = self.contract['scoring_file_sha256'][self.split]
            base.require(base.digest(scoring_path) == expected, 'scoring file hash mismatch')
            return super()._target(episode, purpose)

        def load_target(self, episode, *, purpose, training=False):
            if purpose == 'training':
                return self.training_target(episode, training=training)
            base.require(purpose == 'evaluation' and training is False,
                         'explicit training/evaluation target purpose required')
            return self.evaluation_target(episode)

        def adapter_audit(self):
            return dict(adapter='oe11_calibration_loader_v0', base_sha256=PINNED_LOADER_SHA256,
                        train_support_parent_policy=self.support_mode,
                        development_support_policy='global_source_bank',
                        validation_alias_count=self.validation_alias_count,
                        manifest_sha256=base.digest(self.prepared_root / 'manifest.jsonl'),
                        catalog_sha256=base.digest(self.episodes_root / f'episodes_{self.split}.jsonl'),
                        calibration_is_unseen_geographic_region=False,
                        scoring_loaded=self._scoring is not None,
                        model_acquisition_support='Loader allows 2/4/8; existing encoder wrapper accepts 2 only')

    return CalibrationEpisodeLoader
