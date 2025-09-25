from sentinel_tiles import sentinel_tiles
from ETtoolbox.VIIRS_GEOS5FP import VIIRS_GEOS5FP

tile = "11SPS"
geometry = sentinel_tiles.grid(tile)

VIIRS_GEOS5FP(
    target_date="2025-05-22",
    geometry=geometry,
    target=tile
)
