"""Stop scene labels or conflicting raster grids from becoming area ground truth.

This is a data-contract check, not independent verification of supplied masks.
Original OE1 artifacts must not be overwritten to satisfy these checks.
"""
from __future__ import annotations
import math


def affine6(values):
    if not isinstance(values, (list, tuple)) or len(values) != 6:
        raise ValueError('Expected six GDAL affine coefficients')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in values):
        raise ValueError('Nonfinite/non-numeric affine coefficient')
    try:
        coefficients = tuple(float(x) for x in values)
    except OverflowError as exc:
        raise ValueError('Affine coefficient exceeds finite float range') from exc
    if any(not math.isfinite(x) for x in coefficients):
        raise ValueError('Nonfinite/non-numeric affine coefficient')
    return coefficients


def pixel_area_coordinate_units(transform):
    """Area in squared CRS coordinate units, never implicitly square metres."""
    _, a, b, _, d, e = affine6(transform)
    area = abs(a*e-b*d)
    if not math.isfinite(area) or area <= 0:
        raise ValueError('Nonfinite or degenerate affine grid area')
    return area


def resampled_affine(transform, source_height, source_width, output_height, output_width):
    """Preserve the footprint when resampling; includes rotation/shear terms."""
    sizes = (source_height, source_width, output_height, output_width)
    if any(type(x) is not int or x <= 0 for x in sizes):
        raise ValueError('Raster dimensions must be positive integers')
    x0,a,b,y0,d,e = affine6(transform)
    sx,sy = source_width/output_width,source_height/output_height
    return [x0,a*sx,b*sy,y0,d*sx,e*sy]


def assess_measurement_contract(metadata, reference):
    """Reports every known blocker; never upgrades presence labels to masks.

    A dense/partial mask reference must be separately constructed and audited.
    Partial labels require an explicit support region: the rest is not negative.
    """
    blockers=[]
    outer,inner=metadata.get('geotransform'),metadata.get('source_tiff_geotransform')
    ratio=None
    try:
        oa=pixel_area_coordinate_units(outer)
        if inner is not None:
            ia=pixel_area_coordinate_units(inner)
            candidate_ratio=ia/oa
            if not math.isfinite(candidate_ratio) or candidate_ratio <= 0:
                raise ValueError('Nonfinite or underflowed pixel area ratio')
            ratio=candidate_ratio
            if any(not math.isclose(x,y,rel_tol=1e-12,abs_tol=1e-8) for x,y in zip(affine6(outer),affine6(inner))):
                blockers.append('conflicting_geotransforms')
    except (TypeError,ValueError,OverflowError):
        blockers.append('invalid_or_missing_grid')
    if metadata.get('crs_coordinate_unit') != 'metre' or metadata.get('crs_unit_verified') is not True:
        blockers.append('metric_crs_not_verified')
    scope=reference.get('scope')
    if scope not in ('dense_mask','partial_mask'):
        blockers.append('reference_not_spatial_mask')
    if not reference.get('mask_asset_id') or not reference.get('source_id'):
        blockers.append('missing_spatial_reference_provenance')
    if not reference.get('valid_support_asset_id'):
        blockers.append('missing_labeled_support_region')
    if reference.get('observation_alignment_verified') is not True:
        blockers.append('observation_label_alignment_unverified')
    return {'measurement_eligible':not blockers,'blockers':blockers,
            'inner_to_outer_pixel_area_ratio':ratio,
            'label_scope':scope,'changes_presence_task_eligibility':False}


def require_measurement_contract(metadata,reference):
    result=assess_measurement_contract(metadata,reference)
    if not result['measurement_eligible']:
        raise ValueError('Cannot produce area ground truth: '+', '.join(result['blockers']))
    return result
