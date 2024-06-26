from LandsatL2C2 import LandsatL2C2
from sentinel import sentinel_tile_grid

tile = "11SPS"
geometry = sentinel_tile_grid.grid(tile, 30)

landsat = LandsatL2C2(working_directory="~/data/landsat_testing")

ST_C = landsat.product("2022-11-01", "ST_C", geometry=geometry, target_name=tile)
