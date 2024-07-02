import warnings
from os.path import join, basename, splitext
from glob import glob
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import rasters as rt

locations = pd.read_csv("litvak_locations.csv")
image_directory = "/project/sandbox/ETtoolbox_testing/VIIRS_GEOS5FP_output"
image_filenames = sorted(
    glob(join(image_directory, "**", "*.tif"), recursive=True))

table_rows = []

for i, image_filename in enumerate(image_filenames):
    # print(basename(image_filename))
    print(f"({i + 1} / {len(image_filenames)}) ")
    variable = splitext(basename(image_filename))[0].split("_")[-1]
    time_UTC = datetime.strptime(splitext(basename(image_filename))[
                                 0].split("_")[-3], "%Y.%m.%d.%H.%M.%S")

    image_geometry = rt.RasterGrid.open(image_filename)

    for j, (tower, lat, lon) in locations.iterrows():
        time_solar = time_UTC + timedelta(hours=(np.radians(lon) / np.pi * 12))
        tower_point_latlon = rt.Point(lon, lat)

        if not image_geometry.intersects(tower_point_latlon.to_crs(image_geometry.crs)):
            # print(f"image does not intersect {tower} tower: {image_filename}")
            continue

        tower_row, tower_col = image_geometry.index_point(
            tower_point_latlon.to_crs(image_geometry.crs))
        rows, cols = image_geometry.shape

        if tower_row > rows or tower_row < 0 or tower_col > cols or tower_col < 0:
            # print(f"image does not intersect {tower} tower: {image_filename}")
            continue

        # print(f"extracting tower {tower} at lat {lat} lon {lon} from image: {image_filename}")

        # print(f"tower cell index: [{tower_row}, {tower_col}]")
        # print(f"tower shape: [{rows}, {cols}]")

        min_row = max(tower_row - 1, 0)
        max_row = min(tower_row + 2, rows - 1)
        min_col = max(tower_col - 1, 0)
        max_col = min(tower_col + 2, cols - 1)
        # print(f"3x3 subset index: [{min_row}:{max_row}, {min_col}:{max_col}]")
        subset_3x3 = image_geometry[min_row:max_row, min_col:max_col]
        subset_image = rt.Raster.open(image_filename, geometry=subset_3x3)
        # print(subset_image)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            median_value = np.nanmedian(subset_image)

        # print(f"median value: {median_value}")

        if np.isnan(median_value):
            continue

        print(tower, lat, lon, time_UTC, time_solar, variable, median_value)
        table_rows.append([tower, lat, lon, time_UTC,
                          time_solar, variable, median_value])

df = pd.DataFrame(table_rows, columns=[
                  "tower", "lat", "lon", "time_UTC", "time_solar", "variable", "med3x3"])
df.to_csv("calval.csv", index=False)
df
