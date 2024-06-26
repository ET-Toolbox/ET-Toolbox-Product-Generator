# https://webmap.ornl.gov/ogcdown/dataset.jsp?dg_id=10012_1
from os.path import join, abspath, dirname

from rasters import Raster, RasterGeometry

KOPPEN_GEIEGER_FILENAME = join(abspath(dirname(__file__)), "KG_simplified_uint8.tif")

def load_koppen_geiger(geometry: RasterGeometry = None):
    koppen_geiger = Raster.open(KOPPEN_GEIEGER_FILENAME)

    if geometry is not None:
        koppen_geiger = koppen_geiger.to_geometry(geometry)

    return koppen_geiger
