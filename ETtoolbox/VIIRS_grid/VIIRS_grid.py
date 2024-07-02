import logging

import numpy as np
from affine import Affine
from shapely.geometry import Polygon, MultiPolygon, Point, mapping
import rasters
from rasters import RasterGrid, Raster, RasterGeometry

SINUSOIDAL_PROJECTION = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 +b=6371007.181 +units=m +no_defs"
MODLAND_TILE_SIZES = {
    250: 4800,
    500: 2400,
    1000: 1200,
    5000: 240,
    10000: 120
}

def calculate_MODLAND_affine(h, v, tile_size):
    # boundaries of sinusodial projection
    UPPER_LEFT_X_METERS = -20015109.355798
    UPPER_LEFT_Y_METERS = 10007554.677899
    LOWER_RIGHT_X_METERS = 20015109.355798
    LOWER_RIGHT_Y_METERS = -10007554.677899

    # size across (width or height) of any equal-area sinusoidal target
    TILE_SIZE_METERS = 1111950.5197665554

    # boundaries of MODIS land grid
    TOTAL_ROWS = 18
    TOTAL_COLUMNS = 36

    y_max = LOWER_RIGHT_Y_METERS + (TOTAL_ROWS - v) * TILE_SIZE_METERS
    x_min = UPPER_LEFT_X_METERS + int(h) * TILE_SIZE_METERS

    cell_size = TILE_SIZE_METERS / tile_size

    # width of pixel
    a = cell_size
    # row rotation
    b = 0.0
    # x-coordinate of upper-left corner of upper-left pixel
    c = x_min
    # column rotation
    d = 0.0
    # height of pixel
    e = -1.0 * cell_size
    # y-coordinate of the upper-left corner of upper-left pixel
    f = y_max
    affine = Affine(a, b, c, d, e, f)

    return affine


def calculate_global_MODLAND_affine(spatial_resolution):
    tile_size = MODLAND_TILE_SIZES[spatial_resolution]
    affine = calculate_MODLAND_affine(0, 0, tile_size)

    return affine


def generate_MODLAND_grid(h, v, tile_size):
    affine = calculate_MODLAND_affine(h, v, tile_size)
    grid = RasterGrid.from_affine(affine, tile_size, tile_size, crs=SINUSOIDAL_PROJECTION)

    return grid


def parsehv(region_name):
    h = int(region_name[1:3])
    v = int(region_name[4:6])

    return h, v
