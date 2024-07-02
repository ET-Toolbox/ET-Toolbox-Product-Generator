import warnings
from os.path import join, abspath, dirname
import numpy as np
import geopandas as gpd
import shapely
from shapely.geometry import Polygon, Point
import shapely.wkt
from ETtoolbox.transform.UTM import UTM_proj4_from_latlon
import rasters as rt

def pathdotrow(tile: str) -> str:
    tile = str(tile)

    if len(tile) == 6:
        tile = f"{tile[:3]}.{tile[4:]}"
    elif len(tile) == 7 and tile[4] == "/":
        tile = tile.replace("/", ".")

    if not len(tile) == 7 and tile[4] == ".":
        raise ValueError(f"unrecognized tile: {tile}")

    return tile


class WRS2Descending:
    POLYGONS_FILENAME = join(abspath(dirname(__file__)), "WRS2_descending_polygons.geojson")

    def __init__(self, *args, **kwargs):
        self._polygons = None
        self._centroids = None

    @property
    def polygons(self):
        if self._polygons is None:
            self._polygons = gpd.read_file(self.POLYGONS_FILENAME)

        return self._polygons

    def polygon(self, tile: str) -> Polygon:
        tile = pathdotrow(tile)
        geometry = self.polygons[self.polygons.tile == tile].iloc[0].geometry

        return geometry

    def centroid(self, tile: str) -> Point:
        return self.polygon(tile).centroid

    def tiles(
            self,
            geometry: shapely.geometry.shape or gpd.GeoDataFrame,
            calculate_area: bool = False,
            eliminate_redundancy: bool = False) -> gpd.GeoDataFrame:
        if isinstance(geometry, str):
            geometry = shapely.wkt.loads(geometry)

        if isinstance(geometry, (shapely.geometry.point.Point, shapely.geometry.polygon.Polygon)):
            geometry = gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:4326")
        
        if isinstance(geometry, (rt.Point, rt.Polygon)):
            geometry = gpd.GeoDataFrame(geometry=[geometry.geometry], crs=geometry.crs.to_wkt())

        if not isinstance(geometry, gpd.GeoDataFrame):
            raise ValueError(f"invalid target geometry {type(geometry)}")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            matches = self.polygons[
                self.polygons.intersects(geometry.to_crs(self.polygons.crs).unary_union)]
            matches.rename(columns={"Name": "tile"}, inplace=True)
            tiles = matches[["tile", "geometry"]]

            if calculate_area or eliminate_redundancy:
                centroid = geometry.to_crs("EPSG:4326").unary_union.centroid
                lon = centroid.x
                lat = centroid.y
                projection = UTM_proj4_from_latlon(lat, lon)
                tiles_UTM = tiles.to_crs(projection)
                target_UTM = geometry.to_crs(projection)
                overlap = gpd.overlay(tiles_UTM, target_UTM)
                area = overlap.geometry.area

                if eliminate_redundancy:
                    tiles_UTM["area"] = np.array(area)
                    tiles_UTM.sort_values(by="area", ascending=False, inplace=True)
                    tiles_UTM.reset_index(inplace=True)
                    tiles_UTM = tiles_UTM[["tile", "area", "geometry"]]
                    remaining_target = target_UTM.unary_union
                    remaining_target_area = remaining_target.area
                    indices = []

                    for i, (tile, area, geometry) in tiles_UTM.iterrows():
                        remaining_target = remaining_target - geometry
                        previous_area = remaining_target_area
                        remaining_target_area = remaining_target.area
                        change_in_area = remaining_target_area - previous_area

                        if change_in_area != 0:
                            indices.append(i)

                        if remaining_target_area == 0:
                            break

                    tiles_UTM = tiles_UTM.iloc[indices, :]
                    tiles = tiles_UTM.to_crs(tiles.crs)
                    tiles.sort_values(by="tile", ascending=True, inplace=True)
                    tiles = tiles[["tile", "area", "geometry"]]
                else:
                    tiles["area"] = np.array(area)
                    tiles = tiles[["tile", "area", "geometry"]]

            return tiles
