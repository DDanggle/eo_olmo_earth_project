"""Independent scientific-unit checks; no network, raster opening, or repo edits.

Loads only pure definitions by AST. Local NumPy runs measure checks. Optional
pyproj adds actual ellipsoidal EPSG:3857 checks without requiring rasterio.
Set EO_OVERLAP_SOURCE to choose a server copy of eo_flood_overlap_v0.py.
"""
import ast
import hashlib
import os
from pathlib import Path
import unittest

import numpy as np

SOURCE = Path(os.environ.get('EO_OVERLAP_SOURCE', str(Path(__file__).resolve().parents[1] / 'code/eo_flood_overlap_v0.py')))
tree = ast.parse(SOURCE.read_text())
nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in {'measure', 'pixel_areas'}) or
         (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'WC_CLASSES' for t in n.targets))]
ns = {'np': np}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), ns)
measure = ns['measure']
pixel_areas = ns['pixel_areas']
try:
    from pyproj import Geod, Transformer
    ns.update(Geod=Geod, Transformer=Transformer)
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False


class Affine:
    """Minimal affine protocol used by the pure area helper."""
    def __init__(self, a, b, c, d, e, f):
        self.a, self.b, self.c, self.d, self.e, self.f = a, b, c, d, e, f
    def __mul__(self, point):
        col, row = point
        return self.a*col+self.b*row+self.c, self.d*col+self.e*row+self.f


class MaskContract(unittest.TestCase):
    def test_semantic_nonwater_zero_kept_permanent_water_excluded(self):
        mask = np.array([[0, 1, 2, 3]], dtype=np.uint8)
        result = measure(mask, np.ones_like(mask, bool), np.full_like(mask, 40),
                         np.ones_like(mask, bool), np.array([[7., 11., 13., 17.]]))
        self.assertEqual(result['joint_valid_area_m2'], 31)
        self.assertEqual(result['unknown_area_m2'], 17)
        self.assertEqual(result['reference_flood_area_m2'], 13)
        self.assertEqual(result['classes']['cropland']['flood_overlap_m2'], 13)
        self.assertEqual(result['classes']['cropland']['observed_class_area_m2'], 31)
        self.assertEqual(result['classes']['cropland']['flood_share_of_observed_class'], 13/31)

    def test_explicit_invalid_flag_excludes_apparent_flood(self):
        mask = np.array([[2, 2]], dtype=np.uint8)
        result = measure(mask, np.array([[True, False]]), np.array([[40, 40]]),
                         np.ones_like(mask, bool), np.array([[10., 20.]]))
        self.assertEqual(result['reference_flood_area_m2'], 10)
        self.assertEqual(result['classes']['cropland']['flood_overlap_m2'], 10)
        self.assertEqual(result['unknown_area_m2'], 20)

    def test_unknown_land_is_not_zero_or_nonwater(self):
        mask = np.full((1, 4), 2)
        result = measure(mask, np.ones_like(mask, bool), np.array([[40, 0, 250, 10]]),
                         np.array([[True, True, True, False]]), np.array([[5., 7., 11., 13.]]))
        self.assertEqual(result['joint_valid_area_m2'], 5)
        self.assertEqual(result['unknown_area_m2'], 31)
        self.assertEqual(result['reference_flood_area_m2'], 36)
        self.assertEqual(result['flood_unknown_landcover_area_m2'], 31)
        self.assertEqual(result['classes']['forest']['flood_overlap_m2'], 0)
        self.assertIsNone(result['classes']['forest']['flood_share_of_observed_class'])

    def test_tree_and_cropland_are_disjoint_weighted_intersections(self):
        mask = np.array([[2, 2, 1, 0]])
        result = measure(mask, np.ones_like(mask, bool), np.array([[40, 10, 10, 40]]),
                         np.ones_like(mask, bool), np.array([[2., 3., 5., 7.]]))
        self.assertEqual(result['classes']['cropland']['flood_overlap_m2'], 2)
        self.assertEqual(result['classes']['forest']['flood_overlap_m2'], 3)
        self.assertEqual(result['classes']['cropland']['observed_class_area_m2'], 9)
        self.assertEqual(result['classes']['forest']['observed_class_area_m2'], 8)
        self.assertEqual(result['classes']['forest']['source_class_name'], 'Tree cover')

    def test_area_partition_and_unknown_no_double_count(self):
        mask = np.array([[0, 1, 2], [3, 2, 2]])
        valid = np.array([[True, True, True], [False, True, False]])
        land = np.array([[40, 10, 0], [0, 40, 10]])
        lv = np.ones_like(mask, bool)
        areas = np.arange(1, 7).reshape(2, 3).astype(float)
        result = measure(mask, valid, land, lv, areas)
        self.assertEqual(result['joint_valid_area_m2'] + result['unknown_area_m2'], result['footprint_area_m2'])
        self.assertEqual(result['joint_valid_pixels'] + result['unknown_pixels'], mask.size)
        self.assertEqual(result['flood_unknown_landcover_area_m2'], 3)
        self.assertEqual(result['classes']['cropland']['flood_overlap_m2'], 5)
        self.assertEqual(result['unknown_area_m2'], 3 + 4 + 6)

    def test_all_invalid_preserves_unknown_without_nan_rate(self):
        mask = np.array([[3, 2]])
        result = measure(mask, np.zeros_like(mask, bool), np.array([[0, 40]]),
                         np.ones_like(mask, bool), np.array([[10., 20.]]))
        self.assertEqual(result['joint_valid_area_m2'], 0)
        self.assertEqual(result['unknown_area_m2'], 30)
        self.assertIsNone(result['classes']['cropland']['flood_share_of_observed_class'])

    def test_reject_wrong_grid_class_or_area(self):
        mask = np.array([[2, 2]])
        valid = np.ones_like(mask, bool)
        land = np.full_like(mask, 40)
        for area in (np.array([[1., 0.]]), np.array([[1., -1.]]), np.array([[1., np.nan]]), np.ones((2, 2))):
            with self.subTest(area=area), self.assertRaises(ValueError):
                measure(mask, valid, land, valid, area)
        with self.assertRaisesRegex(ValueError, 'Unexpected flood class'):
            measure(np.array([[2, 4]]), valid, land, valid, np.ones_like(mask, float))


@unittest.skipUnless(HAS_PYPROJ, 'pyproj unavailable locally; run in server geobench env')
class GroundAreaContract(unittest.TestCase):
    def test_actual_case_webmercator_is_not_100m2_per_pixel(self):
        affine = Affine(10, 0, 7534615, 0, -10, 3244315)
        areas = pixel_areas(affine, 'EPSG:3857', 224, 224)
        # The recovered case is ~28 deg N. Geodesic area is well below 100 m².
        self.assertTrue(np.all((areas > 75) & (areas < 82)))
        self.assertGreater(100 * 224 * 224 / areas.sum(), 1.20)
        self.assertLess(100 * 224 * 224 / areas.sum(), 1.35)
        self.assertGreater(abs(float(areas[0, 0] - areas[-1, 0])), 0)

    def test_same_row_fast_path_matches_direct_four_corner_geodesic(self):
        affine = Affine(10, 0, 7534615, 0, -10, 3244315)
        areas = pixel_areas(affine, 'EPSG:3857', 224, 224)
        tr = Transformer.from_crs('EPSG:3857', 'EPSG:4326', always_xy=True)
        geod = Geod(ellps='WGS84')
        for row, col in ((0, 0), (0, 223), (100, 121), (223, 223)):
            xy = [affine * point for point in ((col, row), (col+1, row), (col+1, row+1), (col, row+1))]
            lon, lat = tr.transform(*zip(*xy))
            direct = abs(geod.polygon_area_perimeter(lon, lat)[0])
            self.assertAlmostEqual(areas[row, col], direct, delta=1e-4)

    def test_rotated_affine_uses_all_columns(self):
        affine = Affine(10, 2, 7534615, 1, -10, 3244315)
        areas = pixel_areas(affine, 'EPSG:3857', 2, 2)
        self.assertTrue(np.all(areas > 0))
        self.assertFalse(np.array_equal(areas[:, 0], areas[:, 1]))


if __name__ == '__main__':
    print('source_sha256', hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    unittest.main(verbosity=2)
