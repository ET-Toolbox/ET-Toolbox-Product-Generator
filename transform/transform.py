import warnings
import numpy as np
import geopandas as gpd
import pyproj
import rasterio
from pyproj import Proj, Transformer
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry

WGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"


def get_proj4(projection: Proj or str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        if isinstance(projection, Proj):
            result = projection.crs.to_proj4()

        elif isinstance(projection, rasterio.crs.CRS):
            result = projection.to_proj4()

        elif isinstance(projection, pyproj.crs.crs.CRS):
            result = projection.to_proj4()

        elif isinstance(projection, str):
            # result = pycrs.parse.from_unknown_text(projection).to_proj4()
            result = pyproj.CRS(projection).to_proj4()

        else:
            raise ValueError(f"projection not recognized ({type(projection)})")

    result = result.replace("+init=epsg:", "epsg:")

    return result


def transform_xy(
        x: float or np.ndarray,
        y: float or np.ndarray,
        source_projection: Proj or str = None,
        target_projection: Proj or str = None,
        transformer: Transformer = None):
    if transformer is None:
        if source_projection is None or target_projection is None:
            raise ValueError("no source and target projection")

        source_projection = get_proj4(source_projection)
        target_projection = get_proj4(target_projection)
        transformer = Transformer.from_crs(source_projection, target_projection)

    projected_x, projected_y = transformer.transform(x, y)

    return projected_x, projected_y


def transform_point(
        point: Point,
        source_projection: Proj or str = None,
        target_projection: Proj or str = None,
        transformer: Transformer = None):
    if transformer is None:
        if source_projection is None or target_projection is None:
            raise ValueError("no source and target projection")

        source_projection = get_proj4(source_projection)
        target_projection = get_proj4(target_projection)
        transformer = Transformer.from_crs(source_projection, target_projection)

    projected_point = Point(*transform_xy(point.x, point.y, transformer=transformer))

    return projected_point


def transform_shape(
        shape: BaseGeometry,
        source_projection: Proj or str,
        target_projection: Proj or str) -> BaseGeometry:
    if not isinstance(shape, BaseGeometry):
        raise ValueError("invalid shape")

    # TODO need to stop relying on deprecated proj4 as common projection encoding
    source_projection = get_proj4(source_projection)
    target_projection = get_proj4(target_projection)
    projected_shape = gpd.GeoDataFrame({}, geometry=[shape], crs=source_projection).to_crs(target_projection).geometry[
        0]

    return projected_shape


def center_aeqd_proj4(center_coord):
    """
    Generate Azimuthal Equal Area projection centered at given lat/lon.
    :param center_coord: shapely.geometry.Point object containing latitute and longitude point of center of projection
    :return: pyproj.Proj object of centered projection
    """
    return Proj('+proj=aeqd +lat_0=%f +lon_0=%f' % (
        center_coord.y,
        center_coord.x
    ))

def is_proj_geographic(projection):
    if not isinstance(projection, Proj):
        projection = Proj(str(projection))

    if hasattr(projection, "is_latlong"):
        return projection.is_latlong()
    elif hasattr(projection, "crs"):
        return projection.crs.is_geographic
    else:
        raise ValueError("invalid projection object")

