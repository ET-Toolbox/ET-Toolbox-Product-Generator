from VIIRS.VNP43MA4 import VNP43MA4
from sentinel import sentinel_tile_grid

date_UTC = "2022-07-01"
geometry = sentinel_tile_grid.grid("11SPS", cell_size=1000)
viirs = VNP43MA4(working_directory="~/data/VNP43MA4_download_testing")
viirs.NDVI(date_UTC=date_UTC, geometry=geometry)
