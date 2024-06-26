import os
import time
from os.path import abspath, expanduser, exists
from typing import Union
import ephem
import json
import base64
import pytz
from spacetrack import SpaceTrackClient
from datetime import datetime, timedelta, date
import itertools
from dateutil import parser
import ephem
from shapely.geometry import Point
from datetime import datetime, date, timedelta
from dateutil import parser
import shapely.ops
import pyproj
import numpy as np
from shapely.geometry import Polygon, LineString
import geopandas as gpd
import geopandas as gpd
from datetime import datetime
from dateutil import parser
import logging
import shapely.ops
from rasters import RasterGeometry

from spacetrack_credentials import get_spacetrack_credentials
import cl

logger = logging.getLogger(__name__)


def get_TLE(datetime_UTC: datetime = None, username: str = None, password: str = None, spacetrack_credentials_filename: str = None) -> str:
    if datetime_UTC is None:
        datetime_UTC = datetime.now()
    
    if isinstance(datetime_UTC, str):
        datetime_UTC = parser.parse(datetime_UTC)

    if username is None or password is None:
        credentials = get_spacetrack_credentials(filename=spacetrack_credentials_filename)
        username = credentials.username
        password = credentials.password

    norad_cat_id = 37849

    filename = abspath(expanduser("~/.VIIRS_TLE"))

    if not exists(filename) or (time.time() - os.path.getmtime(abspath(expanduser(filename)))) > 86400:
        # connect to Space Track
        st = SpaceTrackClient(
            identity=username,
            password=password
        )

        epoch = f">{datetime_UTC - timedelta(hours=12)}"

        # print(f"searching TLE for NORAD ID {norad_cat_id} at epoch: {epoch}")

        # query all TLEs for satellite
        tle_query_text = st.tle(
            norad_cat_id=norad_cat_id,
            epoch=epoch,
            orderby='epoch',
            format='3le'
        )

        with open(filename, "w") as file:
            file.write(tle_query_text)
    else:
        with open(filename, "r") as file:
            tle_query_text = file.read()

    lines = tle_query_text.split('\n')
    records = []

    for one, two, three in zip(*[itertools.islice(lines, i, None, 3) for i in range(3)]):
        date_text = two.split()[3]
        year = 2000 + int(date_text[:2])
        doy = int(date_text[2:5])
        d = date(year, 1, 1) + timedelta(days=(doy - 1))
        day_fraction = float(date_text[5:])
        dt = datetime(d.year, d.month, d.day) + timedelta(days=day_fraction)
        tle = '\n'.join((one, two, three))
        records.append((dt, tle))

    record = records[0]
    
    return record

def get_satellite_position(datetime_UTC: date = None, TLE: str = None, spacetrack_credentials_filename: str = None) -> Point:
    if datetime_UTC is None:
        datetime_UTC = datetime.utcnow()
    
    if isinstance(datetime_UTC, str):
        datetime_UTC = parser.parse(datetime_UTC)

    if TLE is None:
        TLE_datetime_UTC, TLE = get_TLE(datetime_UTC - timedelta(hours=3), spacetrack_credentials_filename=spacetrack_credentials_filename)
    
    ephemeris = ephem.readtle(*TLE.split("\n"))
    ephemeris.compute(datetime_UTC)
    lon = ephemeris.sublong * 57.2958
    lat = ephemeris.sublat * 57.2958
    point = Point(lon, lat)

    return point

def center_aeqd_proj4(center_coord: Point) -> str:
    return '+proj=aeqd +lat_0=%f +lon_0=%f' % (
        center_coord.y,
        center_coord.x
    )

def UTC_to_solar(time_UTC: datetime, lon: float) -> datetime:
    return time_UTC + timedelta(hours=(np.radians(lon) / np.pi * 12))

def solar_to_UTC(time_solar: datetime, lon: float) -> datetime:
    return time_solar - timedelta(hours=(np.radians(lon) / np.pi * 12))

def is_day(datetime_UTC: datetime, point_latlon: Point) -> bool:
    solar_time = datetime_UTC + timedelta(hours=(np.radians(point_latlon.x) / np.pi * 12))
    solar_hour = solar_time.hour + solar_time.minute / 3600.0 + solar_time.second / 86400.0

    day_of_year = datetime_UTC.timetuple().tm_yday
    day_angle_rad = (2 * np.pi * (day_of_year - 1)) / 365
    solar_declination_deg = (0.006918 - 0.399912 * np.cos(day_angle_rad) + 0.070257 * np.sin(
        day_angle_rad) - 0.006758 * np.cos(2 * day_angle_rad) + 0.000907 * np.sin(2 * day_angle_rad) - 0.002697 * np.cos(
        3 * day_angle_rad) + 0.00148 * np.sin(3 * day_angle_rad)) * (180 / np.pi)
    sunrise_cosine = -np.tan(np.pi * point_latlon.y / 180) * np.tan(np.pi * solar_declination_deg / 180)

    if sunrise_cosine >= 1:
        sha = 0
    elif sunrise_cosine <= -1:
        sha = 180
    else:
        sha = np.arccos(sunrise_cosine) * (180 / np.pi)

    sunrise_hour = 12 - (sha / 15)
    daylight_hours = (2.0 / 15.0) * sha

    return sunrise_hour < solar_hour < sunrise_hour + daylight_hours

def split_geometry(geometry):
    def to_polar(lon, lat):
        phi = np.pi / 180. * lon
        radius = np.pi / 180. * (90. - sign * lat)

        # nudge points at +/- 180 out of the way so they don't intersect the testing wedge
        phi = np.sign(phi) * np.where(np.abs(phi) > np.pi - 1.5 * epsilon, np.pi - 1.5 * epsilon, np.abs(phi))
        radius = np.where(radius < 1.5 * epsilon, 1.5 * epsilon, radius)

        x = radius * np.sin(phi)
        y = radius * np.cos(phi)
        if (isinstance(lon, list)):
            x = x.tolist()
            y = y.tolist()
        elif (isinstance(lon, tuple)):
            x = tuple(x)
            y = tuple(y)

        return (x, y)

    def from_polar(x, y):
        radius = np.sqrt(np.array(x) ** 2 + np.array(y) ** 2)
        phi = np.arctan2(x, y)

        # close up the tiny gap
        radius = np.where(radius < 2 * epsilon, 0., radius)
        phi = np.sign(phi) * np.where(np.abs(phi) > np.pi - 2 * epsilon, np.pi, np.abs(phi))

        lon = 180. / np.pi * phi
        lat = sign * (90. - 180. / np.pi * radius)

        if (isinstance(x, list)):
            lon = lon.tolist()
            lat = lat.tolist()
        elif (isinstance(x, tuple)):
            lon = tuple(lon)
            lat = tuple(lat)
        return (lon, lat)

    epsilon = 1e-14

    # logger.verbose('forming anti-meridian wedge')

    antimeridian_wedge = shapely.geometry.Polygon([
        (epsilon, -np.pi),
        (epsilon ** 2, -epsilon),
        (0, epsilon),
        (-epsilon ** 2, -epsilon),
        (-epsilon, -np.pi),
        (epsilon, -np.pi)
    ])

    feature_shape = shapely.geometry.shape(geometry)
    sign = 2. * (0.5 * (feature_shape.bounds[1] + feature_shape.bounds[3]) >= 0.) - 1.
    polar_shape = shapely.ops.transform(to_polar, feature_shape)

    if not polar_shape.intersects(antimeridian_wedge):
        # logger.verbose('geometry does not cross the anti-meridian')
        return geometry

    else:
        pass
        # logger.verbose('geometry crosses the anti-meridian')

    difference = polar_shape.difference(antimeridian_wedge)
    output_shape = shapely.ops.transform(from_polar, difference)

    return output_shape

def get_swaths(
        start_datetime_UTC: datetime = None,
        end_datetime_UTC: datetime = None, 
        target: Polygon = None, 
        TLE: str = None,
        spacetrack_credentials_filename: str = None,
        swath_duration_minutes: int = 6,
        day_only: bool = True,
        filter_geometry: bool = True,
        filter_poles: bool = True) -> gpd.GeoDataFrame:
    if start_datetime_UTC is None:
        start_datetime_UTC = datetime.utcnow().date()
    
    if isinstance(start_datetime_UTC, str):
        start_datetime_UTC = parser.parse(start_datetime_UTC)
    
    if end_datetime_UTC is None:
        end_datetime_UTC = datetime.utcnow().date() + timedelta(minutes=swath_duration_minutes)
    
    if isinstance(end_datetime_UTC, str):
        end_datetime_UTC = parser.parse(end_datetime_UTC)

    if TLE is None:
        _, TLE = get_TLE(start_datetime_UTC)

    SWATH_WIDTH = 3000000
    radius = SWATH_WIDTH / 2.0

    times = []
    times_solar = []
    names = []
    daytimes = []
    polygons = []
    satellite_positions = {}

    datetime_UTC = start_datetime_UTC

    while datetime_UTC < end_datetime_UTC:
        times.append(datetime_UTC)
        
        if datetime_UTC not in satellite_positions:
            back_point = get_satellite_position(datetime_UTC, TLE=TLE, spacetrack_credentials_filename=spacetrack_credentials_filename)
            satellite_positions[datetime_UTC] = back_point
        else:
            back_point = satellite_positions[datetime_UTC]

        back_crs = center_aeqd_proj4(back_point)
        back_projected = shapely.ops.transform(pyproj.Transformer.from_crs("EPSG:4326", back_crs, always_xy=True).transform, back_point)
        front_datetime_UTC = datetime_UTC + timedelta(minutes=swath_duration_minutes)

        if front_datetime_UTC not in satellite_positions:
            front_point = get_satellite_position(front_datetime_UTC, TLE=TLE, spacetrack_credentials_filename=spacetrack_credentials_filename)
            satellite_positions[front_datetime_UTC] = front_point
        else:
            front_point = satellite_positions[front_datetime_UTC]

        front_projected = shapely.ops.transform(pyproj.Transformer.from_crs("EPSG:4326", back_crs, always_xy=True).transform, front_point)
        back_direction = np.arctan2(front_projected.y, front_projected.x)
        back_right_direction = back_direction - np.pi / 2
        back_left_direction = back_direction + np.pi / 2
        back_right_projected = Point(radius * np.cos(back_right_direction), radius * np.sin(back_right_direction))
        back_right_point = shapely.ops.transform(pyproj.Transformer.from_crs(back_crs, "EPSG:4326", always_xy=True).transform, back_right_projected)
        back_left_projected = Point(radius * np.cos(back_left_direction), radius * np.sin(back_left_direction))
        back_left_point = shapely.ops.transform(pyproj.Transformer.from_crs(back_crs, "EPSG:4326", always_xy=True).transform, back_left_projected)
        next_datetime_UTC = datetime_UTC + timedelta(minutes=10)

        if next_datetime_UTC not in satellite_positions:
            next_point = get_satellite_position(next_datetime_UTC, TLE=TLE)
            satellite_positions[next_datetime_UTC] = next_point
        else:
            next_point = satellite_positions[next_datetime_UTC]

        front_crs = center_aeqd_proj4(front_point)
        next_projected = shapely.ops.transform(pyproj.Transformer.from_crs("EPSG:4326", front_crs, always_xy=True).transform, next_point)
        front_direction = np.arctan2(next_projected.y, next_projected.x)
        front_right_direction = front_direction - np.pi / 2
        front_left_direction = front_direction + np.pi / 2
        front_right_projected = Point(radius * np.cos(front_right_direction), radius * np.sin(front_right_direction))
        front_right_point = shapely.ops.transform(pyproj.Transformer.from_crs(front_crs, "EPSG:4326", always_xy=True).transform, front_right_projected)
        front_left_projected = Point(radius * np.cos(front_left_direction), radius * np.sin(front_left_direction))
        front_left_point = shapely.ops.transform(pyproj.Transformer.from_crs(front_crs, "EPSG:4326", always_xy=True).transform, front_left_projected)
        polygon = split_geometry(Polygon([back_left_point, back_point, back_right_point, front_right_point, front_point, front_left_point]))
        polygons.append(polygon)
        name = datetime_UTC.strftime("%H%M")
        names.append(name)
        daytime = is_day(datetime_UTC, back_point)
        daytimes.append(daytime)
        datetime_solar = UTC_to_solar(datetime_UTC, back_point.x)
        times_solar.append(datetime_solar)
        datetime_UTC = datetime_UTC + timedelta(minutes=swath_duration_minutes)
    
    gdf = gpd.GeoDataFrame({"time_UTC": times, "time_solar": times_solar, "name": names, "daytime": daytimes}, geometry=polygons)

    if day_only:
        gdf = gdf[gdf["daytime"]]
        gdf = gdf.drop(columns=["daytime"])

    if filter_poles:
        gdf = gdf[~gdf.intersects(LineString([(-180, 90), (180, 90)]))]
        gdf = gdf[~gdf.intersects(LineString([(-180, -90), (180, -90)]))]

    if target is not None:
        gdf["target"] = gdf.intersects(target)

        if filter_geometry:
            gdf = gdf[gdf["target"]]
            gdf = gdf.drop(columns=["target"])

    return gdf

def find_VIIRS_swaths(date_solar: Union[date, str], geometry: Union[Polygon, RasterGeometry] = None, filter_geometry: bool = True, spacetrack_credentials_filename: str = None):
    if isinstance(date_solar, str):
        date_solar = parser.parse(date_solar)

    if isinstance(geometry, RasterGeometry):
        geometry = geometry.boundary_latlon.geometry

    radius_minutes = 60 * 2
    datetime_solar = datetime(date_solar.year, date_solar.month, date_solar.day, 13, 30)
    datetime_UTC = solar_to_UTC(datetime_solar, geometry.centroid.x)
    datetime_UTC = datetime(datetime_UTC.year, datetime_UTC.month, datetime_UTC.day, datetime_UTC.hour, int(datetime_UTC.minute / 6) * 6)
    start_datetime_UTC = datetime_UTC - timedelta(minutes=radius_minutes)
    end_datetime_UTC = datetime_UTC + timedelta(minutes=radius_minutes)
    swaths = get_swaths(start_datetime_UTC, end_datetime_UTC, target=geometry, filter_geometry=filter_geometry, spacetrack_credentials_filename=spacetrack_credentials_filename)

    return swaths