from sentinel_tiles import sentinel_tiles
from ETtoolbox.VIIRS_GFS_forecast import VIIRS_GFS_forecast

tile = "11SPS"
geometry = sentinel_tiles.grid(tile)

VIIRS_GFS_forecast(
    target_date="2025-05-28",
    geometry=geometry,
    target=tile
)
