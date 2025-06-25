# Math imports
import shutil
import time

import numpy as np
import xarray as xr
import pandas as pd

# GIS imports
import rasterio.crs
import rasterio
import rioxarray
from rioxarray import merge
import geopandas as gpd
from pyogrio import set_gdal_config_options
from shapely.geometry import mapping

# General imports
import os, glob, shutil
from multiprocessing import Pool

# Set GDAL configuration
set_gdal_config_options({'SHAPE_RESTORE_SHX': 'YES'})


def process_date(s_target_directory: str, s_date: str, o_shapefile: object, s_output_directory: str):
    """
    Processes the GFS based ET forecasts for a specific date across all calculated tiles. It returns a pandas dataframe that will be combined with other dates

    Parameters
    ----------
    s_target_directory: str
        Path to the GFS output
    s_date: str
        Date in ET Toolbox foramt
    o_shapefile: object
        Shapefile for the area being summarized
    s_output_directory: str
        Path to copy the output rater files into

    Returns
    -------
    df_output: pd.DataFrame
        Contains the output for the daily summary

    """


    # Create a dataframe to hold the data
    df_output = pd.DataFrame(index=[s_date], columns=['count', 'area', 'minimum', 'maximum', 'range', 'mean', 'std', 'percentile_90'])

    # Create a working path
    s_working_date_path = os.path.join(s_target_directory, s_date)

    # Search the folder structure using glob for ET tiles
    sl_working_date_et_tiles = glob.glob(os.path.join(s_working_date_path, '*', '*_ET.tif'), recursive=True)

    # Attempt to process the data
    try:
        # Read the datasets
        ol_datasets = []
        for i_entry_file in range(0, len(sl_working_date_et_tiles), 1):
            o_dataset = xr.open_dataset(os.path.join('outputs', sl_working_date_et_tiles[i_entry_file]), engine='rasterio')
            ol_datasets.append(o_dataset['band_data'])

        # Merge the files
        o_merged_dataset_sum = merge.merge_arrays(ol_datasets, method='sum')
        o_merged_dataset_counts = merge.merge_arrays(ol_datasets, method='count')
        o_merged_dataset = o_merged_dataset_sum / o_merged_dataset_counts

        # Make sure the CRS is set correctly on the merged dataset
        o_merged_dataset.rio.write_crs(32613, inplace=True)

        # Clip the raster to the polygon
        o_clipped_dataset = o_merged_dataset.rio.clip(o_shapefile.geometry.apply(mapping), o_shapefile.crs)

        # Perform math
        df_output.loc[s_date, 'count'] = np.sum(~np.isnan(o_clipped_dataset)).values
        df_output.loc[s_date, 'area'] = o_shapefile.area.values
        df_output.loc[s_date, 'minimum'] = np.nanmin(o_clipped_dataset)
        df_output.loc[s_date, 'maximum'] = np.nanmax(o_clipped_dataset)
        df_output.loc[s_date, 'range'] = df_output.loc[s_date]['maximum'] - df_output.loc[s_date]['minimum']
        df_output.loc[s_date, 'mean'] = np.nanmean(o_clipped_dataset)
        df_output.loc[s_date, 'std'] = np.nanstd(o_clipped_dataset)
        df_output.loc[s_date, 'percentile_90'] = np.percentile(o_clipped_dataset.values.flatten()[~np.isnan(o_clipped_dataset.values.flatten())], 90)

    except:
        # Something went wrong in the data processing. Return the empty frame or the current calculations
        pass

    # Attempt to move the rasters
    try:
        # Create the output folder
        s_daily_output = os.path.join(s_output_directory, s_date)

        if not os.path.isdir(s_daily_output):
            os.makedirs(s_daily_output)

        # Copy the ET files to the output directory
        for i_entry_file in range(0, len(sl_working_date_et_tiles), 1):
            # Create the new filename
            s_new_path = os.path.join(s_daily_output, os.path.basename(sl_working_date_et_tiles[i_entry_file]))

            # Copy the file
            shutil.copyfile(sl_working_date_et_tiles[i_entry_file], s_new_path)

    except:
        pass

    # Return the dataframe
    return df_output


if __name__ == '__main__':

    ### Set the target directory ###
    # Set the input directories
    s_target_directory = 'GFS_output'
    s_target_shapefile = os.path.join("mrg_shapefile", "Projects.shp")

    # Set the output directory
    s_output_directory = '/mnt/export/et_rasters'

    # Find the dates in the folder
    sl_dates = os.listdir(s_target_directory)

    ### Open the shapefile ###
    o_shapefile = gpd.read_file(s_target_shapefile)
    o_shapefile = o_shapefile.to_crs('EPSG:32613')

    ### Create the pandas dataframe to hold the output ###
    df_output = pd.DataFrame(index=sl_dates, columns=['count', 'area', 'minimum', 'maximum', 'range', 'mean', 'std', 'percentile_90'])

    ### Make sure the output directory exists ###
    if not os.path.isdir(s_output_directory):
        os.makedirs(s_output_directory)

    else:
        # Remove the existing folder
        shutil.rmtree(s_output_directory)

        # Pause to allow the delete
        time.sleep(5)

        # Remake the directory
        os.makedirs(s_output_directory)

    ### Loop and process each date ###
    # Open the compute pool
    o_pool = Pool()

    # Create the tasks
    ol_tasks = [o_pool.apply_async(process_date, args=(s_target_directory, x, o_shapefile, s_output_directory)) for x in sl_dates]

    # Get the tasks
    ol_tasks = [x.get() for x in ol_tasks]

    # Close the compute pool
    o_pool.close()

    ### Post process the output ###
    # Remove Nones from the output
    df_output = [x for x in df_output if x is not None]

    try:
        # Concatenate the output
        df_output = pd.concat(ol_tasks, axis=0)

        # Save the dataframe to csv for later use
        df_output.to_csv('summary.csv')

        # Copy the summary file to the drive
        shutil.copyfile('summary.csv', os.path.join(s_output_directory, 'summary.csv'))

    except:
        # Error with the output. Just pass and not create the file
        pass