import copy
import unittest

from eo_query_core_v0 import query_catalog


def record(sid, overlap=None, *, event='2021-07-01', unknown=0, valid=1000):
    row = {'id': sid, 'dataset': 'kurosiwo', 'aoi_id': '1', 'region_id': '1',
           'split': 'test', 'event_type': 'flood', 'status': 'reference_flood',
           'event_date': event, 'point': [126.0, 37.0],
           'flood_fraction_pct': 20.0, 'evidence': []}
    if overlap is not None:
        row['land_cover_overlap'] = {
            'status': 'measured',
            'baseline': {'dataset': 'ESA WorldCover 2020', 'reference_end': '2020-12-31'},
            'classes': {'cropland': {'class_code': 40, 'flood_overlap_m2': overlap},
                        'forest': {'class_code': 10, 'flood_overlap_m2': overlap / 2}},
            'joint_valid_area_m2': valid,
            'unknown_area_m2': unknown,
        }
    return row


class PartialCoverageTests(unittest.TestCase):
    def query(self, rows, **kwargs):
        return query_catalog({'records': rows}, {'dataset': 'kurosiwo', 'land_cover': 'cropland', **kwargs})

    def test_one_of_7000_never_counts_as_complete_search(self):
        result = self.query([record('computed', 100)] + [record(f'unknown{i}') for i in range(6999)])
        self.assertEqual(result['status'], 'partial_coverage')
        self.assertEqual(result['matched_count'], 1)
        self.assertEqual(result['coverage']['scope_count'], 7000)
        self.assertEqual(result['coverage']['assessed_count'], 1)
        self.assertEqual(result['coverage']['unassessed_count'], 6999)
        self.assertFalse(result['coverage_complete'])

    def test_known_zero_with_unknown_is_not_no_matches(self):
        result = self.query([record('zero', 0), record('unknown')])
        self.assertEqual(result['status'], 'partial_coverage')
        self.assertEqual(result['matched_count'], 0)
        self.assertEqual(result['coverage']['observed_zero_count'], 1)
        self.assertEqual(result['coverage']['fully_observed_zero_count'], 1)
        self.assertEqual(result['coverage']['unassessed_count'], 1)

    def test_no_measurements_needs_data(self):
        result = self.query([record('unknown')])
        self.assertEqual(result['status'], 'needs_data')
        self.assertEqual(result['coverage']['unassessed_count'], 1)
        self.assertEqual(result['blockers'], ['not_computed'])

    def test_all_known_zeros_are_no_observed_overlap(self):
        result = self.query([record('zero1', 0), record('zero2', 0)])
        self.assertEqual(result['status'], 'no_matches')
        self.assertTrue(result['coverage_complete'])
        self.assertEqual(result['coverage']['fully_observed_zero_count'], 2)

    def test_scope_is_after_date_and_before_limit(self):
        result = self.query([record('yes1', 20), record('yes2', 30), record('out', event='2019-01-01')],
                            start='2021-01-01', limit=1)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['coverage']['scope_count'], 2)
        self.assertEqual(result['matched_count'], 2)
        self.assertEqual(result['returned_count'], 1)
        self.assertEqual(result['records'][0]['id'], 'yes2')

    def test_scope_respects_geographic_filter(self):
        outside = record('outside')
        outside['point'] = [1.0, 1.0]
        result = self.query([record('measured', 12), outside], bbox=[125, 36, 127, 38])
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['coverage']['scope_count'], 1)

    def test_2020_annual_map_cannot_be_pre_event_for_2020_event(self):
        for event in ['2019-01-01', '2020-12-31']:
            with self.subTest(event=event):
                result = self.query([record('case', 100, event=event)])
                self.assertEqual(result['status'], 'needs_data')
                self.assertEqual(result['coverage']['unassessed_reasons'], {'baseline_not_strictly_pre_event': 1})

    def test_partial_pixels_stay_partial_even_without_uncomputed_chips(self):
        for overlap in (0, 100):
            with self.subTest(overlap=overlap):
                result = self.query([record('case', overlap, unknown=50)])
                self.assertEqual(result['status'], 'partial_coverage')
                self.assertEqual(result['coverage']['partial_pixel_count'], 1)
                self.assertEqual(result['coverage']['unassessed_count'], 0)
                self.assertEqual(result['coverage']['fully_observed_zero_count'], 0)

    def test_no_joint_valid_pixels_are_unknown(self):
        result = self.query([record('case', 0, valid=0, unknown=1000)])
        self.assertEqual(result['status'], 'needs_data')
        self.assertEqual(result['blockers'], ['no_joint_valid_pixels'])

    def test_missing_class_is_unknown(self):
        row = record('case', 12)
        del row['land_cover_overlap']['classes']['cropland']
        result = self.query([row])
        self.assertEqual(result['status'], 'needs_data')
        self.assertEqual(result['blockers'], ['requested_class_not_computed'])

    def test_missing_coverage_metadata_is_unknown(self):
        row = record('case', 12)
        del row['land_cover_overlap']['unknown_area_m2']
        result = self.query([row])
        self.assertEqual(result['status'], 'needs_data')
        self.assertEqual(result['blockers'], ['coverage_metadata_unavailable'])

    def test_tree_cover_is_not_renamed_forest_inventory(self):
        result = self.query([record('case', 100)], land_cover='forest')
        match = result['records'][0]['land_cover_match']
        self.assertEqual(match['source_class_code'], 10)
        self.assertEqual(match['source_class_name'], 'Tree cover')
        self.assertEqual(match['flood_overlap_m2'], 50)

    def test_malformed_area_or_class_rejected(self):
        invalid = [record('too_big', 1001), record('nan', float('nan')), record('negative', -1)]
        bad_class = record('wrong_class', 12)
        bad_class['land_cover_overlap']['classes']['cropland']['class_code'] = 10
        invalid.append(bad_class)
        for row in invalid:
            with self.subTest(row=row['id']), self.assertRaises(ValueError):
                self.query([row])

    def test_land_all_does_not_require_measurements(self):
        result = self.query([record('case')], land_cover='all')
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['matched_count'], 1)
        self.assertNotIn('coverage', result)

    def test_empty_other_filter_scope_not_global_needs_data(self):
        result = self.query([record('unknown')], aoi_id='absent')
        self.assertEqual(result['status'], 'no_matches')
        self.assertEqual(result['coverage']['scope_count'], 0)

    def test_source_records_never_mutated(self):
        rows = [record('case', 10)]
        before = copy.deepcopy(rows)
        result = self.query(rows)
        result['records'][0]['land_cover_match']['baseline']['dataset'] = 'changed output copy'
        self.assertEqual(rows, before)


if __name__ == '__main__':
    unittest.main()
