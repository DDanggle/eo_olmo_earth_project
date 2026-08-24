"""Small rslearn compositor adapters used by the Jeju v7 smoke test."""

from rasterio.enums import Resampling
from rslearn.dataset.sentinel2_scl import Sentinel2SCLBestClear


class Sentinel2SCLBestClearNearest(Sentinel2SCLBestClear):
    """Score categorical SCL with nearest-neighbor, then materialize normally.

    The upstream compositor forwards the reflectance layer's resampling method to
    SCL scoring. Reflectance uses bilinear resampling here, but SCL is categorical;
    interpolating its class IDs would corrupt equality tests such as ``scl == 9``.
    This adapter changes only the scoring read. The selected reflectance item still
    uses the layer's configured bilinear resampling in the inherited compositor.
    """

    def _score_item(self, item, tile_store, projection, bounds, resampling_method):
        return super()._score_item(
            item=item,
            tile_store=tile_store,
            projection=projection,
            bounds=bounds,
            resampling_method=Resampling.nearest,
        )
