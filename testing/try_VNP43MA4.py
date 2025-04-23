from ETtoolbox.VIIRS import VNP43MA4
from sentinel_tiles import sentinel_tiles

date_UTC = "2022-07-01"
geometry = sentinel_tiles.grid("11SPS", cell_size=1000)
viirs = VNP43MA4(working_directory="~/data/VNP43MA4_download_testing")
viirs.NDVI(date_UTC=date_UTC, geometry=geometry)
