import copy,json,unittest
from measurement_contract import assess_measurement_contract,require_measurement_contract,resampled_affine,pixel_area_coordinate_units


class MeasurementContractTests(unittest.TestCase):
    def good(self):
        return ({'geotransform':[1,10,0,2,0,-10],'source_tiff_geotransform':[1,10,0,2,0,-10],
                 'crs_coordinate_unit':'metre','crs_unit_verified':True},
                {'scope':'partial_mask','mask_asset_id':'mask-v1','source_id':'survey-1','valid_support_asset_id':'support-v1','observation_alignment_verified':True})

    def test_presence_is_never_area_gold(self):
        m,r=self.good();r['scope']='scene_presence'
        with self.assertRaisesRegex(ValueError,'reference_not_spatial_mask'):require_measurement_contract(m,r)

    def test_conflicting_grids_even_with_valid_mask(self):
        m,r=self.good();m['source_tiff_geotransform']=[1,60,0,2,0,-60]
        s=assess_measurement_contract(m,r)
        self.assertFalse(s['measurement_eligible']);self.assertEqual(s['inner_to_outer_pixel_area_ratio'],36)

    def test_partial_labels_need_support(self):
        m,r=self.good();r.pop('valid_support_asset_id')
        self.assertIn('missing_labeled_support_region',assess_measurement_contract(m,r)['blockers'])

    def test_degrees_not_square_metres(self):
        m,r=self.good();m['crs_coordinate_unit']='degree'
        self.assertFalse(assess_measurement_contract(m,r)['measurement_eligible'])

    def test_missing_alignment_does_not_pass(self):
        m,r=self.good();r['observation_alignment_verified']=False
        self.assertFalse(assess_measurement_contract(m,r)['measurement_eligible'])

    def test_verified_partial_reference_passes_without_mutation(self):
        m,r=self.good();old=copy.deepcopy((m,r))
        self.assertTrue(require_measurement_contract(m,r)['measurement_eligible']);self.assertEqual((m,r),old)

    def test_footprint_preserved_under_resampling(self):
        src=[100,60,0,200,0,-60];dst=resampled_affine(src,20,20,120,120)
        self.assertEqual(dst,[100,10,0,200,0,-10])
        self.assertAlmostEqual(20*20*pixel_area_coordinate_units(src),120*120*pixel_area_coordinate_units(dst))

    def test_rotated_anisotropic_grid(self):
        src=[1,3,4,2,5,6];dst=resampled_affine(src,20,30,40,90)
        # Pixel-to-world footprint corners must remain unchanged.
        def corner(t,x,y):return(t[0]+x*t[1]+y*t[2],t[3]+x*t[4]+y*t[5])
        for x,y in [(0,0),(30,0),(0,20),(30,20)]:
            for a,b in zip(corner(src,x,y),corner(dst,x*3,y*2)):self.assertAlmostEqual(a,b)

    def test_nonfinite_and_singular_rejected(self):
        m,r=self.good();m['geotransform'][1]=float('nan')
        self.assertIn('invalid_or_missing_grid',assess_measurement_contract(m,r)['blockers'])
        with self.assertRaises(ValueError):pixel_area_coordinate_units([0,1,1,0,1,1])

    def test_verified_flags_require_boolean_true(self):
        for value in ('false','true',1,0,False,None,[],{'verified':True}):
            for field,blocker,target in (
                ('crs_unit_verified','metric_crs_not_verified','metadata'),
                ('observation_alignment_verified','observation_label_alignment_unverified','reference'),
            ):
                with self.subTest(field=field,value=value):
                    m,r=self.good()
                    (m if target=='metadata' else r)[field]=value
                    result=assess_measurement_contract(m,r)
                    self.assertFalse(result['measurement_eligible'])
                    self.assertIn(blocker,result['blockers'])

    def test_finite_coefficients_cannot_overflow_determinant(self):
        # Finite inputs can produce infinity or inf-inf (NaN).
        for grid in ([0,1e308,0,0,0,-1e308],
                     [0,1e308,1e308,0,1e308,1e308]):
            with self.subTest(grid=grid):
                with self.assertRaises(ValueError):pixel_area_coordinate_units(grid)
                m,r=self.good();m['geotransform']=grid
                result=assess_measurement_contract(m,r)
                self.assertFalse(result['measurement_eligible'])
                self.assertIn('invalid_or_missing_grid',result['blockers'])
                json.dumps(result,allow_nan=False)

    def test_integer_conversion_overflow_fails_closed(self):
        m,r=self.good();m['geotransform'][1]=10**1000
        result=assess_measurement_contract(m,r)
        self.assertFalse(result['measurement_eligible'])
        self.assertIn('invalid_or_missing_grid',result['blockers'])

    def test_area_ratio_overflow_and_underflow_fail_closed(self):
        tiny=[0,1e-154,0,0,0,-1e-154]
        huge=[0,1e154,0,0,0,-1e154]
        for outer,inner in ((tiny,huge),(huge,tiny)):
            with self.subTest(outer=outer):
                # Each determinant is finite and positive; only the ratio fails.
                self.assertGreater(pixel_area_coordinate_units(outer),0)
                self.assertGreater(pixel_area_coordinate_units(inner),0)
                m,r=self.good();m['geotransform']=outer;m['source_tiff_geotransform']=inner
                result=assess_measurement_contract(m,r)
                self.assertFalse(result['measurement_eligible'])
                self.assertIn('invalid_or_missing_grid',result['blockers'])
                self.assertIsNone(result['inner_to_outer_pixel_area_ratio'])
                json.dumps(result,allow_nan=False)


if __name__=='__main__':unittest.main()
