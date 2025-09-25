import logging, os
from datetime import datetime, timedelta, date, timezone
from typing import List, Callable, Union

import numpy as np

import colored_logging
import rasters as rt
from gedi_canopy_height import GEDICanopyHeight
from GEOS5FP import GEOS5FP

from global_forecasting_system import forecast_Ta_C, forecast_RH, get_GFS_listing, forecast_SWin
from harmonized_landsat_sentinel import HLS2Connection
from ETtoolbox.LANCE import retrieve_vnp43ma4n, retrieve_vnp43ia4n, retrieve_vnp21nrt_emissivity, available_LANCE_dates
from ETtoolbox.LANCE_GEOS5FP_NRT import LANCE_GEOS5FP_NRT, LANCENotAvailableError, GEOS5FPNotAvailableError, retrieve_vnp21nrt_st, \
    check_LANCE_already_processed, DEFAULT_LANCE_OUTPUT_DIRECTORY, load_LANCE
from ETtoolbox.LANCE_GFS_forecast import LANCE_GFS_forecast
from ETtoolbox.LandsatL2C2 import LandsatL2C2
from MODISCI import MODISCI
from ETtoolbox.SRTM import SRTM
from soil_capacity_wilting import SoilGrids
from solar_apparent_time import solar_to_UTC
from PTJPL import PTJPL
from GEOS5FP.downscaling import bias_correct, downscale_soil_moisture, downscale_air_temperature, downscale_vapor_pressure_deficit, downscale_relative_humidity
from GEOS5FP.downscaling import linear_downscale
from rasters import Raster, RasterGrid
from sentinel_tiles import sentinel_tiles

from .credentials import *
from .daterange import date_range
from .constants import *

logger = logging.getLogger(__name__)

ET_MODEL_NAME = "PTJPLSM"
SWIN_MODEL_NAME = "GEOS5FP"
RN_MODEL_NAME = "Verma"

DOWNSCALE_AIR = True
DOWNSCALE_HUMIDITY = True
DOWNSCALE_MOISTURE = True
FLOOR_TOPT = True

STATIC_DIRECTORY = "PTJPL_static"
SRTM_DIRECTORY = "SRTM_download_directory"
LANCE_DIRECTORY = "LANCE_download_directory"
GEOS5FP_DIRECTORY = "GEOS5FP_download_directory"
GFS_DIRECTORY = "GFS_download_directory"

HLS_CELL_SIZE = 30
I_CELL_SIZE = 500
M_CELL_SIZE = 1000
GEOS5FP_CELL_SIZE = 27375
GFS_CELL_SIZE = 27375
LANDSAT_INITIALIZATION_DAYS = 16
HLS_INITIALIZATION_DAYS = 10

TARGET_VARIABLES = ["Rn", "LE", "ET", "ESI", "SM", "ST", "Ta", "RH", "SWin"]


def check_distribution(o_image: Raster, s_variable: str, s_date_utc: date or str, s_target: str):
    """
    Checks the properties of the raster and logs messages appropriately

    Parameters
    ----------
    o_image: object
        Opened raster image to be tested
    s_variable: str
        Variable to analyze within the raster setup
    s_date_utc: date or str
        Date of the file
    s_target: str
        Data sources

    Returns
    -------
    None. Values are written into the log file.

    """

    ### Perform math checks on the raster ###
    # Get the unique values
    da_unique = np.unique(o_image)

    # Calculate the nan fraction. This is used as a proxy for the data quality
    d_nan_proportion = np.count_nonzero(np.isnan(o_image)) / np.size(o_image)

    ### Log the quality of the raster ###
    if len(da_unique) < 10:
        # The number of unique values is low. Log the data as problematic and continue
        # Log the file and variable information
        logger.info("variable " + colored_logging.name(s_variable) + " on " + colored_logging.time(f"{s_date_utc:%Y-%m-%d}") + " at " + colored_logging.place(s_target))

        # Count occurences of each unique value
        for d_value in da_unique:
            # Count the occurences of the specific value
            i_count = np.count_nonzero(o_image == d_value)

            # Log based on the trial value
            if d_value == 0:
                # Log the occurences of a zero value in the raster
                logger.info(f"* {colored_logging.colored(d_value, 'red')}: {colored_logging.colored(i_count, 'red')}")

            else:
                # Log the occurence of any other value
                logger.info(f"* {colored_logging.val(d_value)}: {colored_logging.val(i_count)}")

    else:
        # There are many unique values in the dataset. Calculate the dataset quality based on the minimum/maximum
        ## Process the raster minimum ##
        # Calculate the minimum value
        d_minimum = np.nanmin(o_image)

        # Convert to a string based on the sign of the value
        if d_minimum < 0:
            # The minimum value is less than zero. This is not expected, so log it as an angry red
            s_minimum_string = colored_logging.colored(f"{d_minimum:0.3f}", "red")

        else:
            # The minimum value is zero or greater. Log normally.
            s_minimum_string = colored_logging.val(f"{d_minimum:0.3f}")

        ## Process the raster maximum ##
        # Calculate the raster maximum
        d_maximum = np.nanmax(o_image)

        # Convert to a string based on the sign of the value
        if d_maximum <= 0:
            # The maximum values is less than zero. This is not expected, so log it as an angry red
            s_maximum_string = colored_logging.colored(f"{d_maximum:0.3f}", "red")

        else:
            # The maximum is greater than zero. Log normally.
            s_maximum_string = colored_logging.val(f"{d_maximum:0.3f}")

        ## Process the fraction of nans ##
        # Convert to a sting based on the fraction of nans
        if d_nan_proportion == 1:
            # All of the dataset is nans. Log it as angry red.
            s_nan_proportion_string = colored_logging.colored(f"{(d_nan_proportion * 100):0.2f}%", "red")

        elif d_nan_proportion > 0.5:
            # More than half the dataset is nans. This is likely concerning. Log it as unhappy yellow.
            s_nan_proportion_string = colored_logging.colored(f"{(d_nan_proportion * 100):0.2f}%", "yellow")

        else:
            # Less than half the dataset is nans. Lot it as normal.
            s_nan_proportion_string = colored_logging.val(f"{(d_nan_proportion * 100):0.2f}%")

        ## Output the message ##
        # Construct the log message
        s_message = "variable " + colored_logging.name(s_variable) + \
                  " on " + colored_logging.time(f"{s_date_utc:%Y-%m-%d}") + \
                  " at " + colored_logging.place(s_target) + \
                  " min: " + s_minimum_string + \
                  " mean: " + colored_logging.val(f"{np.nanmean(o_image):0.3f}") + \
                  " max: " + s_maximum_string + \
                  " nan: " + s_nan_proportion_string + f" ({colored_logging.val(o_image.nodata)})"

        # Add a zero check message onto the end of the log and determine log type
        if np.all(o_image == 0):
            # The whole datsaet is zeros.
            # Append a message
            s_message += " all zeros"

            # Log as a warning
            logger.warning(s_message)

        else:
            # The dataset has nonzero values. Log normally.
            logger.info(s_message)

    ### Perform an nan check ###
    if d_nan_proportion == 1:
        # The whole dataset is nans. Log it.
        logger.error(f"variable {s_variable} on {s_date_utc:%Y-%m-%d} at {s_target} is a blank image")


def ET_toolbox_hindcast_forecast_tile(
        tile: str,
        o_present_date: Union[date, str] = None,
        water: Raster = None,
        model: PTJPL = None,
        ET_model_name: str = ET_MODEL_NAME,
        SWin_model_name: str = SWIN_MODEL_NAME,
        Rn_model_name: str = RN_MODEL_NAME,
        s_working_directory: str = None,
        s_static_directory: str = None,
        HLS_download: str = None,
        HLS_initialization_days: int = HLS_INITIALIZATION_DAYS,
        landsat_download: str = None,
        landsat_initialization_days: int = LANDSAT_INITIALIZATION_DAYS,
        s_gfs_download_directory: str = None,
        s_lance_download_directory: str = None,
        s_lance_output_directory: str = None,
        o_srtm_connection: SRTM = None,
        s_srtm_download_directory: str = None,
        o_geos5fp_connection: GEOS5FP = None,
        s_geos5fp_download_directory: str = None,
        GEOS5FP_products: str = None,
        GEDI_connection: GEDICanopyHeight = None,
        GEDI_download: str = None,
        ORNL_connection: MODISCI = None,
        CI_directory: str = None,
        soil_grids_connection: SoilGrids = None,
        soil_grids_download: str = None,
        intermediate_directory: str = None,
        ANN_model: Callable = None,
        ANN_model_filename: str = None,
        resampling: str = DEFAULT_RESAMPLING,
        o_hls_geometry: RasterGrid = None,
        HLS_cell_size: float = HLS_CELL_SIZE,
        o_i_geometry: RasterGrid = None,
        I_cell_size: float = I_CELL_SIZE,
        o_m_geometry: RasterGrid = None,
        M_cell_size: float = M_CELL_SIZE,
        o_geos5fp_geometry: RasterGrid = None,
        GEOS5FP_cell_size: float = GEOS5FP_CELL_SIZE,
        o_gfs_geometry: RasterGrid = None,
        GFS_cell_size: float = GFS_CELL_SIZE,
        downscale_air: bool = DOWNSCALE_AIR,
        downscale_humidity: bool = DOWNSCALE_HUMIDITY,
        downscale_moisture: bool = DOWNSCALE_MOISTURE,
        floor_Topt: bool = FLOOR_TOPT,
        apply_GEOS5FP_GFS_bias_correction: bool = True,
        save_intermediate: bool = False,
        show_distribution: bool = False,
        load_previous: bool = True,
        sl_target_variables: List[str] = None):

    ### Authenticate on to the necessary systems ###


    ### Get the current date ###
    # Check if a date was provided
    if o_present_date is None:
        # No date was provided. Use the system time in UTC
        o_present_date = datetime.now(timezone.utc)

    # Log the current date being used
    logger.info(f"generating ET Toolbox hindcast and forecast at tile {colored_logging.place(tile)} centered on present date: {colored_logging.time(o_present_date)}")

    ### Setup the directory structure ###
    ## Working directory ##
    # Check if a working directory is provided
    if s_working_directory is None:
        # A working directory is not provided. Assume the local directory.
        s_working_directory = "~/data/ET_toolbox"

    # Get the absolute path of the working directory
    s_working_directory = os.path.abspath(os.path.expanduser(s_working_directory))

    # Log the working direcotry
    logger.info(f"working directory: {s_working_directory}")

    ## Static directory ##
    # Check if a static directory is provided
    if s_static_directory is None:
        # A static directory is not provided. Assume the global default.
        s_static_directory = os.path.join(s_working_directory, STATIC_DIRECTORY)

    # Log the static directory
    logger.info(f"static directory: {s_static_directory}")

    ## SRTM directory ##
    # Check if a srtm directory is provided
    if s_srtm_download_directory is None:
        # A srtm directory is not provided. Assume the global default
        s_srtm_download_directory = os.path.join(s_working_directory, SRTM_DIRECTORY)

    # Log the srtm directory
    logger.info(f"SRTM directory: {s_srtm_download_directory}")

    ## LANCE directory ##
    # Check if a lance directory is provided
    if s_lance_download_directory is None:
        # A lance directory is not provided. Assume the global default.
        s_lance_download_directory = os.path.join(s_working_directory, LANCE_DIRECTORY)

    # Log the lance directory
    logger.info(f"LANCE directory: {s_lance_download_directory}")

    ## GEOS5FP directory ##
    # Check if a geos5fp directory is provided
    if s_geos5fp_download_directory is None:
        # A geos5fp directory is not provided. Assume the global default
        s_geos5fp_download_directory = os.path.join(s_working_directory, GEOS5FP_DIRECTORY)

    # Log the geos5fp directory
    logger.info(f"GEOS-5 FP directory: {s_geos5fp_download_directory}")

    ## GFS directory ##
    # Check if a gfs directory is provided
    if s_gfs_download_directory is None:
        # A gfs directory is not provided. Assume the global default.
        s_gfs_download_directory = os.path.join(s_working_directory, GFS_DIRECTORY)

    ## LANCE output directory ##
    # Check if a lance output directory is provided
    if s_lance_output_directory is None:
        # A lance directory is not provided. Assume the global default.
        s_lance_output_directory = os.path.join(s_working_directory, DEFAULT_LANCE_OUTPUT_DIRECTORY)

    ### Geometry checks ###
    ## Lat/long boundary ##
    # Parse the hls geometry from sentinel if geometry is not provided
    if o_hls_geometry is None:
        # Geometry is not provided. Use sentinel to set the values
        # Log the cell size
        logger.info(f"HLS cell size: {colored_logging.val(HLS_cell_size)}m")

        # Read the geometry
        o_hls_geometry = sentinel_tiles.grid(tile, cell_size=HLS_cell_size)

    # Get the lat/long from the geometry boundary
    o_hls_polygon_latlon = o_hls_geometry.boundary_latlon.geometry

    ## I geometry ##
    # Parse the sentinel tiles for the I geometry if not provided
    if o_i_geometry is None:
        # Geometry is not provided. Calculate from sentinel
        # Log the cell size
        logger.info(f"I-band cell size: {colored_logging.val(I_cell_size)}m")

        # Get the geometry information
        o_i_geometry = sentinel_tiles.grid(tile, cell_size=I_cell_size)

    ## M geometry ##
    # Parse the sentinel tiles for the M geometry if not provided
    if o_m_geometry is None:
        # M geomtry is not provided. Calculated from sentinel
        # Log the cell size
        logger.info(f"I-band cell size: {colored_logging.val(M_cell_size)}m")

        # Get the geometry information
        o_m_geometry = sentinel_tiles.grid(tile, cell_size=M_cell_size)

    ## GOES5FP geometry ##
    # Parse the sentinel files for the GEOES5FP geometry if not provided
    if o_geos5fp_geometry is None:
        # GEOS5FP geometry is not provided. Calculate from sentinel
        # Log the cell size
        logger.info(f"GEOS-5 FP cell size: {colored_logging.val(GEOS5FP_cell_size)}m")

        # Get the geometry information
        o_geos5fp_geometry = sentinel_tiles.grid(tile, cell_size=GEOS5FP_cell_size)

    ## GFS geometry ##
    # Parse the sentinel tiles for the GFS geometry if not provided
    if o_gfs_geometry is None:
        # GFS geometry is not provided. Calculate from sentinel
        # Log the cell size
        logger.info(f"GFS cell size: {colored_logging.val(GFS_cell_size)}m")

        # Get the geometry information
        o_gfs_geometry = sentinel_tiles.grid(tile, cell_size=GFS_cell_size)

    ### Set the target variables ##
    # Determine the target variables if not provided
    if sl_target_variables is None:
        # Target variables are not provided. Use the default values.
        sl_target_variables = TARGET_VARIABLES

    ### Setup the data connections ###
    ## Create the GEOS5FP connection ##
    if o_geos5fp_connection is None:
        # Connection is not provided. Create a new connection object.
        o_geos5fp_connection = GEOS5FP(download_directory=s_geos5fp_download_directory)

    ## Create the SRTM connection ##
    if o_srtm_connection is None:
        # Connection is not provided. Create a new connection object
        o_srtm_connection = SRTM(working_directory=s_working_directory, download_directory=s_srtm_download_directory, offline_ok=True)

    ## Create the Landsat/Sentinel connection ##
    o_hls = HLS2Connection(working_directory=s_working_directory, download_directory=HLS_download, target_resolution=int(HLS_cell_size))

    ## Get the geometry mask where elevation is available ##
    bm_water_hls = o_srtm_connection.swb(o_hls_geometry)
    bm_water_m = o_srtm_connection.swb(o_m_geometry)
    bm_water_i = o_srtm_connection.swb(o_i_geometry)

    ### Begin LANCE processing ###
    ## Find the available lance dates ##
    # Log status update
    logger.info("listing available LANCE dates")

    # Request available dates on the VNP43MA4N product
    ol_lance_dates = available_LANCE_dates("VNP43MA4N", archive="5200")

    # Get the first and last dates fromt he list
    o_earliest_lance_date = ol_lance_dates[0]
    o_latest_lance_date = ol_lance_dates[-1]

    # Log the date range
    logger.info(f"LANCE is available from {colored_logging.time(o_earliest_lance_date)} to {colored_logging.time(o_latest_lance_date)}")

    # Create the storage variables to determine what lance dates are available
    e_lance_dates_processed = set()

    # Loop over the last week of data and determine if the data is available
    for i_entry_relative_days in range(-7, 0):
        # Construct the target date and log vhe value
        o_target_date = o_present_date + timedelta(days=i_entry_relative_days)
        logger.info(f"LANCE GEOS-5 FP target date: {colored_logging.time(o_target_date)} ({colored_logging.time(i_entry_relative_days)} days)")

        # Relate the date back to UTC time
        o_time_solar = datetime(o_target_date.year, o_target_date.month, o_target_date.day, 13, 30)
        logger.info(f"LANCE target time solar: {colored_logging.time(o_time_solar)}")
        o_time_utc = solar_to_UTC(o_time_solar, o_hls_geometry.centroid.latlon.x)

        # Determine if the date is within the valid lance range. Otherwise skip to the next date.
        if o_target_date.date() < o_earliest_lance_date:
            # Date is earlier than the earliest available lance date
            logger.info(f"target date {o_target_date} is before earliest available LANCE {o_earliest_lance_date}")
            continue

        elif o_target_date.date() > o_latest_lance_date:
            # Date is later than the latest available lance date
            logger.info(f"target date {o_target_date} is after latest available LANCE {o_latest_lance_date}")
            continue

        # Determine if the data is already processed
        b_lance_already_processed = check_LANCE_already_processed(LANCE_output_directory=s_lance_output_directory, target_date=o_target_date, time_UTC=o_time_utc,
                                                                  target=tile, products=sl_target_variables)

        # If data is already processed, add it to the set to prevent recalculation
        if b_lance_already_processed:
            logger.info(f"LANCE GEOS-5 FP already processed at tile {colored_logging.place(tile)} for date {o_target_date}")
            e_lance_dates_processed |= {o_target_date}
            continue

    # Set and log the hls target range
    o_hls_start = o_earliest_lance_date - timedelta(days=HLS_initialization_days)
    o_hls_end = o_earliest_lance_date - timedelta(days=1)
    logger.info(f"forming HLS NDVI composite from {colored_logging.time(o_hls_start)} to {colored_logging.time(o_hls_end)}")

    ### Create the HLS NDVI images ###
    # Create a list to hold the objects
    ol_ndvi_images = []

    # Loop on the date range and attempt to append the data
    for o_entry_hls_date in date_range(o_hls_start, o_hls_end):
        try:
            ol_ndvi_images.append(o_hls.NDVI(tile=tile, date_UTC=o_entry_hls_date).to_geometry(o_hls_geometry))
        except Exception as e:
            logger.warning(e)
            continue

    # Create hls the rasters and swap with the prior
    o_ndvi_hls_initial = Raster(np.nanmedian(np.stack(ol_ndvi_images), axis=0), geometry=o_hls_geometry)
    o_ndvi_hls_initial = rt.where(bm_water_hls, np.nan, o_ndvi_hls_initial)
    o_ndvi_hls_prior = o_ndvi_hls_initial

    ## Parse the albedo rasters ##
    # Create the list to hold the rasters
    ol_albedo_images = []

    # Loop on the dates
    for o_entry_hls_date in date_range(o_hls_start, o_hls_end):
        # Attempt to create the raster, otherwise log an error
        try:
            ol_albedo_images.append(o_hls.albedo(tile=tile, date_UTC=o_entry_hls_date).to_geometry(o_hls_geometry))
        except Exception as e:
            logger.warning(e)
            continue

    # Set the initial raster maps to use
    o_albedo_hls_initial = Raster(np.nanmedian(np.stack(ol_albedo_images), axis=0), geometry=o_hls_geometry)
    o_albedo_HLS_prior = o_albedo_hls_initial

    ## Setup the landsat connection ##
    # Create the initial connection
    o_landsat = LandsatL2C2(working_directory=s_working_directory, download_directory=landsat_download)

    # Find the available scenes between the LANCE dates
    o_landsat_start = o_earliest_lance_date - timedelta(days=landsat_initialization_days)
    o_landsat_end = o_earliest_lance_date - timedelta(days=1)
    o_landsat_listing = o_landsat.scene_search(start=o_landsat_start, end=o_landsat_end, target_geometry=o_hls_polygon_latlon)

    # Extract the images into a list
    ol_landsat_scene_images = []
    for date_UTC in sorted(set(o_landsat_listing.date_UTC)):
        # Loop across the days, attempting to find a scene associated with a day
        try:
            o_landsat_scene_st_celsius = o_landsat.product(acquisition_date=date_UTC, product="ST_C", geometry=o_hls_geometry, target_name=tile)
            ol_landsat_scene_images.append(o_landsat_scene_st_celsius)

        except Exception as e:
            logger.warning(e)
            continue

    # Concatenate the scenes together
    o_landsat_scene_initial_stc = None
    o_landsat_scene_prior_stc = None            # todo: This will fail if no information is available. Should fail back to climatology

    if len(ol_landsat_scene_images) > 0:
        o_landsat_scene_initial_stc = Raster(np.nanmedian(np.stack(ol_landsat_scene_images), axis=0), geometry=o_hls_geometry)
        o_landsat_scene_prior_stc = o_landsat_scene_initial_stc

    ### Process the hindcast information ###
    # Construct a list to hold the missing dates
    ol_missing_dates = []

    ## Loop back on the last week to create estimates ##
    for i_entry_relative_days in range(-7, 0):
        # Get and log the current target date
        o_target_date = o_present_date + timedelta(days=i_entry_relative_days)
        logger.info(f"LANCE GEOS-5 FP target date: {colored_logging.time(o_target_date)} ({colored_logging.time(i_entry_relative_days)} days)")

        # Convert the date to UTC format
        o_time_solar = datetime(o_target_date.year, o_target_date.month, o_target_date.day, 13, 30)
        logger.info(f"LANCE target time solar: {colored_logging.time(o_time_solar)}")
        o_time_utc = solar_to_UTC(o_time_solar, o_hls_geometry.centroid.latlon.x)

        # Check that the date is within the availability of the LANCE range
        if o_target_date.date() < o_earliest_lance_date:
            # Date is before the earliest LANCE date. Skip the day.
            logger.info(f"target date {o_target_date} is before earliest available LANCE {o_earliest_lance_date}")
            continue

        if o_target_date.date() > o_latest_lance_date:
            # Date is after the latest LANCE date. Skip the day.
            logger.info(f"target date {o_target_date} is after latest available LANCE {o_latest_lance_date}")
            continue

        # The day possibly has data. Attempt to process it.
        try:
            # Check if the date has already been processed
            b_lance_already_processed = check_LANCE_already_processed(LANCE_output_directory=s_lance_output_directory, target_date=o_target_date, time_UTC=o_time_utc,
                                                                      target=tile, products=sl_target_variables)

            # Date has been processed. Skip it.
            if b_lance_already_processed:
                logger.info(f"LANCE GEOS-5 FP already processed at tile {colored_logging.place(tile)} for date {o_target_date}")
                continue


            # Advance the current ST_C variable from the previous date
            try:
                # Attempt to get an update of the variable
                o_new_landsat_stc = o_landsat.product(acquisition_date=o_target_date, product="ST_C", geometry=o_hls_geometry, target_name=tile)

                # Update the mask
                o_landsat_stc = rt.where(np.isnan(o_new_landsat_stc), o_landsat_scene_prior_stc, o_new_landsat_stc)

            except Exception as e:
                # Error in the request or data is not available. Keep the existing estimate from the previous day.
                o_landsat_stc = o_landsat_scene_prior_stc

            # Swap the prior value for the current value
            o_landsat_scene_prior_stc = o_landsat_stc


            # Get the LANCE VNP21 data
            # Log the attempt to the console
            logger.info(f"retrieving LANCE VNP21 ST for tile {colored_logging.place(tile)} on date {colored_logging.time(o_target_date)} at "
                        f"{colored_logging.val(M_cell_size)}m resolution")

            # Request the data from the server
            o_vnp21nrt_st_kelvin_mband, o_valid_date_utc = retrieve_vnp21nrt_st(o_geometry=o_m_geometry, o_date_solar=o_target_date, s_directory=s_lance_download_directory,
                                                                                s_resampling="cubic")

            # Strip timezone from value
            o_valid_date_utc = o_valid_date_utc.replace(tzinfo=None)

            # Log a difference in date
            if o_target_date != o_valid_date_utc:
                logger.info(f"Attempted data retrieval for {o_target_date}. Using {o_valid_date_utc} based on data availablity")

            # Perform initial unity conversions and regridding
            if o_vnp21nrt_st_kelvin_mband is not None:
                o_vnp21nrt_st_celsius_mband = o_vnp21nrt_st_kelvin_mband - 273.15
                o_vnp21nrt_st_celsius_mband_smooth = o_geos5fp_connection.Ts_K(time_UTC=o_valid_date_utc, geometry=o_m_geometry, resampling="cubic") - 273.15
                o_vnp21nrt_st_celsius_mband = rt.where(np.isnan(o_vnp21nrt_st_celsius_mband), o_vnp21nrt_st_celsius_mband_smooth, o_vnp21nrt_st_celsius_mband)

            else:
                o_vnp21nrt_st_celsius_mband = None

            # Get the Landsat data update
            try:
                # Request the tile update
                o_ndvi_hls = o_hls.NDVI(tile=tile, date_UTC=o_target_date).to_geometry(o_hls_geometry)

                # Update the boolean mask
                o_ndvi_hls = rt.where(np.isnan(o_ndvi_hls), o_ndvi_hls_prior, o_ndvi_hls)

            except Exception as e:
                # Error in the request or data is not available. Keep the existing estimate from the previous day.
                o_ndvi_hls = o_ndvi_hls_prior

            # Swap the prior value for the current value
            o_ndvi_hls_prior = o_ndvi_hls


            # Get the LANCE data
            # Log the mband attempt
            logger.info(f"retrieving LANCE VIIRS M-band NDVI for tile {colored_logging.place(tile)} on date {colored_logging.time(o_target_date)} at "
                        f"{colored_logging.val(M_cell_size)}m resolution")

            # Request the mband from the server
            o_ndvi_mband = retrieve_vnp43ma4n(o_geometry=o_m_geometry, o_date_utc=o_target_date, s_variable="NDVI", s_directory=s_lance_download_directory, s_resampling="cubic")

            # Resample the mband data
            o_ndvi_mband_smooth = o_geos5fp_connection.NDVI(time_UTC=o_time_utc, geometry=o_m_geometry, resampling="cubic")
            o_ndvi_mband = rt.where(np.isnan(o_ndvi_mband), o_ndvi_mband_smooth, o_ndvi_mband)
            o_ndvi_mband = rt.where(bm_water_m, np.nan, o_ndvi_mband)

            # Log the iband attempt
            logger.info(f"retrieving LANCE VIIRS I-band NDVI for tile {colored_logging.place(tile)} on date {colored_logging.time(o_target_date)} at "
                        f"{colored_logging.val(I_cell_size)}m resolution")

            # Request the iband from the server
            o_ndvi_iband = retrieve_vnp43ia4n(o_geometry=o_i_geometry, o_date_utc=o_target_date, s_variable="NDVI", s_directory=s_lance_download_directory, s_resampling="cubic")

            # Resample the iband
            o_ndvi_iband_smooth = o_geos5fp_connection.NDVI(time_UTC=o_time_utc, geometry=o_i_geometry, resampling="cubic")
            o_ndvi_iband = rt.where(np.isnan(o_ndvi_iband), o_ndvi_iband_smooth, o_ndvi_iband)
            o_ndvi_iband = rt.where(bm_water_i, np.nan, o_ndvi_iband)

            # Bias correct the bands
            logger.info(f"down-scaling I-band NDVI to HLS composite for tile {colored_logging.place(tile)} on date {colored_logging.time(o_target_date)} from "
                        f"{colored_logging.val(I_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")
            o_ndvi = bias_correct(coarse_image=o_ndvi_iband, fine_image=o_ndvi_hls)

            # Smooth the bias corrected products, mask, and clip to the domain
            o_ndvi_smooth = o_geos5fp_connection.NDVI(time_UTC=o_time_utc, geometry=o_hls_geometry, resampling="cubic")
            o_ndvi = rt.where(np.isnan(o_ndvi), o_ndvi_smooth, o_ndvi)
            o_ndvi = rt.where(bm_water_hls, np.nan, o_ndvi)
            o_ndvi = rt.clip(o_ndvi, 0, 1)
            check_distribution(o_ndvi, "NDVI", o_target_date, tile)

            # Log the emissivity attempt
            logger.info(f"retrieving LANCE VNP21 emissivity for tile {colored_logging.place(tile)} on date {colored_logging.time(o_target_date)} at "
                        f"{colored_logging.val(M_cell_size)}m resolution")

            # Request the emissivity portion of the m band from the server
            o_emissivity_mband, o_valid_date_utc = retrieve_vnp21nrt_emissivity(o_geometry=o_m_geometry, o_date_solar=o_target_date, s_directory=s_lance_download_directory, s_resampling="cubic")

            # Strip timezone from value
            o_valid_date_utc = o_valid_date_utc.replace(tzinfo=None)

            # Log a difference in date
            if o_target_date != o_valid_date_utc:
                logger.info(f"Attempted data retrieval for {o_target_date}. Using {o_valid_date_utc} based on data availablity")

            # Process the data based on availability
            if o_emissivity_mband is not None:
                # Data is available. Convert the values.
                o_emissivity_mband = rt.where(bm_water_m, 0.96, o_emissivity_mband)
                o_emissivity_mband = rt.where(np.isnan(o_emissivity_mband), 1.0094 + 0.047 * np.log(rt.clip(o_ndvi_mband, 0.01, 1)), o_emissivity_mband)
                o_emissivity_estimate = 1.0094 + 0.047 * np.log(rt.clip(o_ndvi, 0.01, 1))

                # Bias correct the emissivity
                logger.info(f"down-scaling VNP21 emissivity to HLS composite for tile {colored_logging.place(tile)} on date {colored_logging.time(o_target_date)} from "
                            f"{colored_logging.val(M_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")
                o_emissivity = bias_correct(coarse_image=o_emissivity_mband, fine_image=o_emissivity_estimate)

                # Smooth and clip the raster to the domain
                o_emissivity = rt.where(bm_water_hls, 0.96, o_emissivity)
                o_emissivity = rt.clip(o_emissivity, 0, 1)
                check_distribution(o_emissivity, "emissivity", o_target_date, tile)

            else:
                # Data is not available. Set the values to none to indicate the lack of data avialability. # todo: this should be replaced by climatology
                o_emissivity = None

            # Get the albedo data update
            try:
                # Request the tile update
                o_albedo_hls = o_hls.albedo(tile=tile, date_UTC=o_target_date).to_geometry(o_hls_geometry)

                # Update the boolean mask
                o_albedo_hls = rt.where(np.isnan(o_albedo_hls), o_albedo_HLS_prior, o_albedo_hls)

            except Exception as e:
                # Error in the request or data is not available. Keep the existing estimate from the previous day.
                o_albedo_hls = o_albedo_HLS_prior

            # Swap the prior value for the current value
            o_albedo_HLS_prior = o_albedo_hls

            # Log the albedo attempt
            logger.info(f"retrieving LANCE VIIRS M-band albedo for tile {colored_logging.place(tile)} on date {colored_logging.time(o_target_date)} at "
                        f"{colored_logging.val(M_cell_size)}m resolution")

            # Request the albedo portion of the m band from the server
            o_albedo_mband = retrieve_vnp43ma4n(o_geometry=o_m_geometry, o_date_utc=o_target_date, s_variable="albedo", s_directory=s_lance_download_directory, s_resampling="cubic")

            # Smooth the albedo and update the mask
            o_albedo_mband_smooth = o_geos5fp_connection.ALBEDO(time_UTC=o_time_utc, geometry=o_m_geometry, resampling="cubic")
            o_albedo_mband = rt.where(np.isnan(o_albedo_mband), o_albedo_mband_smooth, o_albedo_mband)

            # Bias correct the albedo
            logger.info(f"down-scaling M-band albedo to HLS composite for tile {colored_logging.place(tile)} on date {colored_logging.time(o_target_date)} from "
                        f"{colored_logging.val(M_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")
            o_albedo = bias_correct(coarse_image=o_albedo_mband, fine_image=o_albedo_hls)

            # Smooth and clip the raster to the domain
            o_albedo_smooth = o_geos5fp_connection.ALBEDO(time_UTC=o_time_utc, geometry=o_hls_geometry, resampling="cubic")
            o_albedo = rt.where(np.isnan(o_albedo), o_albedo_smooth, o_albedo)
            o_albedo = rt.clip(o_albedo, 0, 1)
            check_distribution(o_albedo, "albedo", o_target_date, tile)


            # Downscale the datasets to match landsat
            # Log the attempt
            logger.info(f"down-scaling VNP21 ST to Landsat 8/9 for tile {colored_logging.place(tile)} on date {colored_logging.time(o_target_date)} from "
                        f"{colored_logging.val(M_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")

            # Bias correct the landsat scene
            if o_landsat_stc is not None and o_vnp21nrt_st_celsius_mband is not None:
                # Data is available to bias correct. Make the bias correction and performan operations
                o_landsat_scene_st_celsius = bias_correct(coarse_image=o_vnp21nrt_st_celsius_mband, fine_image=o_landsat_stc)

                # Smooth and convert units on the datasets
                o_st_celsius_smooth = o_geos5fp_connection.Ts_K(time_UTC=o_time_utc, geometry=o_hls_geometry, resampling="cubic") - 273.15
                o_landsat_scene_st_celsius = rt.where(np.isnan(o_landsat_scene_st_celsius), o_st_celsius_smooth, o_landsat_scene_st_celsius)
                o_st_kelvin = o_landsat_scene_st_celsius + 273.15
                check_distribution(o_landsat_scene_st_celsius, "ST_C", o_target_date, tile)

            else:
                # A fine image is not available. Set values to None to flag issues. # todo: this should be replaced with climatology
                o_landsat_scene_st_celsius = None
                o_st_kelvin = None

            # Acquire soil moisture data
            if downscale_moisture and o_st_kelvin is not None:
                # Attempt to use downscaling to match the spatial soil moisture spatial resolution
                # Log the attempt
                logger.info(f"down-scaling GEOS-5 FP soil moisture to HLS composite for tile {colored_logging.place(tile)} on date {colored_logging.time(o_target_date)} from "
                            f"{colored_logging.val(GEOS5FP_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")

                # Get soil moisture at both the coarse and fine geometries
                o_soil_moisture_coarse_geos5fp = o_geos5fp_connection.SFMC(time_UTC=o_time_utc, geometry=o_geos5fp_geometry, resampling="cubic")
                o_soil_moisture_smooth_geos5fp = o_geos5fp_connection.SFMC(time_UTC=o_time_utc, geometry=o_hls_geometry, resampling="cubic")

                # Downscale the soil moisture
                o_soil_moisture = downscale_soil_moisture(time_UTC=o_time_utc, fine_geometry=o_hls_geometry, coarse_geometry=o_geos5fp_geometry,
                                                          SM_coarse=o_soil_moisture_coarse_geos5fp,SM_resampled=o_soil_moisture_smooth_geos5fp, ST_fine=o_st_kelvin,
                                                          NDVI_fine=o_ndvi, water=bm_water_hls)
            else:
                # Downscaling is not necessary
                # Log the attempt
                logger.info(f"down-sampling GEOS-5 FP soil moisture for tile {colored_logging.place(tile)} on date {colored_logging.time(o_target_date)} from "
                            f"{colored_logging.val(GEOS5FP_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")

                # Request the soil moisture data at the landsat resolution
                o_soil_moisture = o_geos5fp_connection.SFMC(time_UTC=o_time_utc, geometry=o_hls_geometry, resampling="cubic")

            # Check the distribution of the data
            check_distribution(o_soil_moisture, "SM", o_target_date, tile)
  

            # Acquire air temperature data
            if downscale_air and o_st_kelvin is not None:
                # Attempt to use downscaling to match the air temperature spatial resolution
                # Log the attempt
                logger.info(f"down-scaling GEOS-5 FP air temperature to HLS composite for tile {colored_logging.place(tile)} on date {colored_logging.time(o_target_date)} "
                            f"from {colored_logging.val(GEOS5FP_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")

                # Get air temperature at both the coarse and fine resolutions
                o_air_temperature_kelvin_coarse = o_geos5fp_connection.Ta_K(time_UTC=o_time_utc, geometry=o_geos5fp_geometry, resampling="cubic")
                o_air_temperature_kelvin_smooth = o_geos5fp_connection.Ta_K(time_UTC=o_time_utc, geometry=o_hls_geometry, resampling="cubic")

                # Downscale the air temperature
                o_air_temperature_kelvin = downscale_air_temperature(time_UTC=o_time_utc, Ta_K_coarse=o_air_temperature_kelvin_coarse, ST_K=o_st_kelvin,
                                                                     fine_geometry=o_hls_geometry, coarse_geometry=o_geos5fp_geometry)

                # Update the boolean mask
                o_air_temperature_kelvin = rt.where(np.isnan(o_air_temperature_kelvin), o_air_temperature_kelvin_smooth, o_air_temperature_kelvin)

            else:
                # Downscaling is not necessary.
                # Log the attempt
                logger.info(f"down-sampling GEOS-5 FP air temperature for tile {colored_logging.place(tile)} on date {colored_logging.time(o_target_date)} from "
                            f"{colored_logging.val(GEOS5FP_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")

                # Request the air temperature data at the landsat resolution
                o_air_temperature_kelvin = o_geos5fp_connection.Ta_K(time_UTC=o_time_utc, geometry=o_hls_geometry, resampling="cubic")

            # Adjust from kelvin to celsius
            o_air_temperature_celsius = o_air_temperature_kelvin - 273.15

            # Check the distribution
            check_distribution(o_air_temperature_celsius, "Ta_C", o_target_date, tile)


            # Acquire relative humidity data
            if downscale_humidity and o_st_kelvin is not None:
                # Attempt to use downscaling to match the relative humidity spatial resolution
                # Log the attempt
                logger.info(f"down-scaling GEOS-5 FP humidity to HLS composite for tile {colored_logging.place(tile)} on date {colored_logging.time(o_target_date)} from "
                            f"{colored_logging.val(GEOS5FP_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")

                # Get vapor pressure at both the coarse and fine spatial resolutions
                o_vapor_pressure_deficit_pascals_coarse = o_geos5fp_connection.VPD_Pa(time_UTC=o_time_utc, geometry=o_geos5fp_geometry, resampling="cubic")
                o_vapor_pressure_deficit_pascals_smooth = o_geos5fp_connection.VPD_Pa(time_UTC=o_time_utc, geometry=o_hls_geometry, resampling="cubic")

                # Downscale the vapor pressure deficit
                o_vapor_pressure_deficit_pascals = downscale_vapor_pressure_deficit(time_UTC=o_time_utc, VPD_Pa_coarse=o_vapor_pressure_deficit_pascals_coarse,
                                                                                    ST_K=o_st_kelvin, fine_geometry=o_hls_geometry,coarse_geometry=o_geos5fp_geometry)

                # Update the boolean mask
                o_vapor_pressure_deficit_pascals = rt.where(np.isnan(o_vapor_pressure_deficit_pascals), o_vapor_pressure_deficit_pascals_smooth,
                                                            o_vapor_pressure_deficit_pascals)

                # Convert from pascals to kilo pascals
                o_vapor_pressure_deficit_kilopascals = o_vapor_pressure_deficit_pascals / 1000

                # Request the relative humidity at both the coarse and fine resolutions
                o_relative_humidity_coarse = o_geos5fp_connection.RH(time_UTC=o_time_utc, geometry=o_geos5fp_geometry, resampling="cubic")
                o_relative_humidity_smooth = o_geos5fp_connection.RH(time_UTC=o_time_utc, geometry=o_hls_geometry, resampling="cubic")

                # Downscale the relative humidity
                o_relative_humidity = downscale_relative_humidity(time_UTC=o_time_utc, RH_coarse=o_relative_humidity_coarse, SM=o_soil_moisture, ST_K=o_st_kelvin,
                                                                  VPD_kPa=o_vapor_pressure_deficit_kilopascals, water=bm_water_hls, fine_geometry=o_hls_geometry,
                                                                  coarse_geometry=o_geos5fp_geometry)

                # Update the boolean mask
                o_relative_humidity = rt.where(np.isnan(o_relative_humidity), o_relative_humidity_smooth, o_relative_humidity)

            else:
                # Lot the attempt
                logger.info(f"down-sampling GEOS-5 FP relative humidity for tile {colored_logging.place(tile)} on date {colored_logging.time(o_target_date)} from "
                            f"{colored_logging.val(GEOS5FP_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")

                # Request relative humidity at the landsat resolution
                o_relative_humidity = o_geos5fp_connection.RH(time_UTC=o_time_utc, geometry=o_hls_geometry, resampling="cubic")

            # Check the relative humidity distribution
            check_distribution(o_relative_humidity, "RH", o_target_date, tile)

            # Calculate the ET estimate
            LANCE_GEOS5FP_NRT(target_date=o_target_date.date(), o_geometry=o_hls_geometry, target=tile, working_directory=s_working_directory, static_directory=s_static_directory,
                              SRTM_connection=o_srtm_connection, SRTM_download=s_srtm_download_directory,
                              GEDI_connection=GEDI_connection, GEDI_download=GEDI_download,
                              ORNL_connection=ORNL_connection,
                              CI_directory=CI_directory,
                              soil_grids_connection=soil_grids_connection, soil_grids_download=soil_grids_download,
                              s_lance_download_directory=s_lance_download_directory, LANCE_output_directory=s_lance_output_directory,
                              intermediate_directory=intermediate_directory,
                              ANN_model=ANN_model, ANN_model_filename=ANN_model_filename, o_model=model, s_model_name=ET_model_name,
                              o_st_band_celsius=o_landsat_scene_st_celsius,
                              o_emissivity=o_emissivity,
                              o_ndvi=o_ndvi,
                              o_albedo=o_albedo,
                              SM=o_soil_moisture,
                              o_air_temperature_celsius=o_air_temperature_celsius,
                              RH=o_relative_humidity,
                              SWin=SWin_model_name,
                              Rn=Rn_model_name,
                              o_water=bm_water_hls,
                              coarse_cell_size=GEOS5FP_cell_size,
                              target_variables=sl_target_variables,
                              downscale_air=downscale_air, downscale_humidity=downscale_humidity, downscale_moisture=downscale_moisture,
                              floor_Topt=floor_Topt,
                              resampling=resampling,
                              show_distribution=show_distribution,
                              load_previous=load_previous,
                              save_intermediate=save_intermediate)

            # Add the date to the processed set
            e_lance_dates_processed |= {o_target_date}

        except (LANCENotAvailableError, GEOS5FPNotAvailableError) as e:
            # Data is not available to process the date. Log and continue to the next date
            logger.warning(e)
            logger.warning(f"LANCE GEOS-5 FP cannot be processed for date: {o_target_date}")
            ol_missing_dates.append(o_target_date)
            continue
        except Exception as e:
            # A different exception handle that is not expected. Stop the code and handle it
            logger.exception(e)
            logger.warning(f"LANCE GEOS-5 FP cannot be processed for date: {o_target_date}")
            ol_missing_dates.append(o_target_date)
            continue

    # Log the missing dates
    logger.info("missing LANCE GEOS-5 FP dates: " + ", " .join(colored_logging.time(d) for d in ol_missing_dates))

    ### Begin processing the forecast dates ###
    # Construct a list of the dates to test with the forecast
    ol_forecast_dates = ol_missing_dates + [o_present_date + timedelta(days=d) for d in range(8)]

    # Get the range of dates for LANCE
    o_earliest_lance_date = min(e_lance_dates_processed)
    o_latest_lance_date = max(e_lance_dates_processed)

    # Get the available GFS forecats
    logger.info("getting GFS listing")
    of_gfs_listing = get_GFS_listing(o_present_date)

    ## Loop and process each forecast date ##
    for o_target_date in ol_forecast_dates:
        # Calculate and log the date offset
        i_entry_relative_days = o_target_date - o_present_date
        logger.info(f"GFS LANCE target date: {colored_logging.time(o_target_date)} ({colored_logging.time(i_entry_relative_days)} days)")

        # Convert the date to UTC
        #o_time_solar = datetime(o_target_date.year, o_target_date.month, o_target_date.day, 13, 30)
        o_time_solar = datetime.now()
        logger.info(f"LANCE target time solar: {colored_logging.time(o_time_solar)}")
        o_time_utc = solar_to_UTC(o_time_solar, o_hls_geometry.centroid.latlon.x)

        # Check that the date is within the range of available LANCE dates
        if o_target_date < o_earliest_lance_date:
            # Before the earliest LANCE date
            logger.warning(f"target date {o_target_date} is before earliest available LANCE {o_earliest_lance_date}")
            continue

        elif o_target_date <= o_latest_lance_date:
            # Date is within the available lance information
            logger.warning(f"target date {colored_logging.time(o_target_date)} is within LANCE date range from {colored_logging.time(o_earliest_lance_date)} to "
                           f"{colored_logging.time(o_latest_lance_date)}")

            # Set the lance date to the target date
            o_lance_processing_date = o_target_date

        else:
            # Date is into the future. Use the most recent lance date available
            o_lance_processing_date = o_latest_lance_date

            # Log the date being used
            logger.info(f"processing LANCE on latest date available: {colored_logging.time(o_lance_processing_date)}")

        # Convert the lance dates to UTC
        o_lance_processing_datetime_solar = datetime(o_lance_processing_date.year, o_lance_processing_date.month, o_lance_processing_date.day, 13, 30)
        logger.info(f"LANCE processing date/time solar: {colored_logging.time(o_lance_processing_datetime_solar)}")
        o_lance_processing_datetime_UTC = solar_to_UTC(o_lance_processing_datetime_solar, o_hls_geometry.centroid.latlon.x)
        logger.info(f"LANCE processing date/time UTC: {colored_logging.time(o_lance_processing_datetime_UTC)}")

        # Load the target lance data
        o_most_recent = load_LANCE(LANCE_output_directory=s_lance_output_directory, target_date=o_lance_processing_date, target=tile)

        # Process if LANCE data is available #
        if len(o_most_recent) > 0:
            # Pull the most recent variables from the lance data
            o_landsat_scene_st_celsius = o_most_recent["ST"]
            o_emissivity = o_most_recent["emissivity"]
            o_ndvi = o_most_recent["NDVI"]
            o_albedo = o_most_recent["albedo"]
            o_soil_moisture = o_most_recent["SM"]

            # Log the tile that's being used
            logger.info(f"down-scaling GFS solar radiation to HLS composite for tile {colored_logging.place(tile)} on date {colored_logging.time(o_target_date)} from "
                        f"{colored_logging.val(GFS_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")

            # Not entirely sure what this is. Calls to the download and interpolate of the GFS forecast itself. Might be a swin transformer.
            o_swin_prior = o_most_recent["SWin"]
            o_swin_gfs = forecast_SWin(time_UTC=o_time_utc, geometry=o_gfs_geometry, directory=s_gfs_download_directory, listing=of_gfs_listing)

            # Apply bias correction if called for
            # if apply_GEOS5FP_GFS_bias_correction:
            #     # Request the forecast object
            #     o_matching_swin_gfs = forecast_SWin(time_UTC=o_lance_processing_datetime_UTC, geometry=o_gfs_geometry, directory=s_gfs_download_directory, resampling="cubic",
            #                                        listing=of_gfs_listing)
            #
            #     # Request the same product from geos
            #     o_matching_swin_geos5fp = o_geos5fp_connection.SWin(time_UTC=o_lance_processing_datetime_UTC, geometry=o_gfs_geometry, resampling="cubic")
            #
            #     # Calculate the bias
            #     o_swin_gfs_bias = o_matching_swin_gfs - o_matching_swin_geos5fp
            #     o_swin_gfs = o_swin_gfs - o_swin_gfs_bias
            #
            # # Perform the bias correction
            o_swin = o_swin_gfs.to_geometry(o_swin_prior.geometry, resampling='linear')
            #o_swin = bias_correct(coarse_image=o_swin_gfs, fine_image=o_swin_prior)


            # Bias correct the temperature
            # Log the attempt
            logger.info(f"down-scaling GFS air temperature to HLS composite for tile {colored_logging.place(tile)} on date {colored_logging.time(o_target_date)} from "
                        f"{colored_logging.val(GFS_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")

            # Extract the most recent air temperature and the forecast product
            o_air_temperature_celsius_prior = o_most_recent["Ta"]
            o_air_temperature_celsius_gfs = forecast_Ta_C(time_UTC=o_time_utc, geometry=o_gfs_geometry, directory=s_gfs_download_directory, listing=of_gfs_listing)

            # Apply bias correction if called for
            # if apply_GEOS5FP_GFS_bias_correction:
            #     # Get the air temperature from the gfs
            #     o_matching_air_temperature_celsius_gfs = forecast_Ta_C(time_UTC=o_lance_processing_datetime_UTC, geometry=o_gfs_geometry, directory=s_gfs_download_directory,
            #                                                            resampling="cubic", listing=of_gfs_listing)
            #
            #     # Get the air temperature at the gfs resolution from geos5fp
            #     o_matching_air_temperature_celsius_geos5fp = o_geos5fp_connection.Ta_C(time_UTC=o_lance_processing_datetime_UTC, geometry=o_gfs_geometry, resampling="cubic")
            #
            #     # Calculate the bias
            #     o_air_temperature_c_gfs_bias = o_matching_air_temperature_celsius_gfs - o_matching_air_temperature_celsius_geos5fp
            #     o_air_temperature_celsius_gfs = o_air_temperature_celsius_gfs - o_air_temperature_c_gfs_bias
            #
            # # Calculate the biase correction
            # o_air_temperature_celsius = bias_correct(coarse_image=o_air_temperature_celsius_gfs, fine_image=o_air_temperature_celsius_prior)
            o_air_temperature_celsius = o_air_temperature_celsius_gfs.to_geometry(o_air_temperature_celsius_prior.geometry, resampling='linear')


            # Bias correct the relative humidity
            logger.info(f"down-scaling GFS humidity to HLS composite for tile {colored_logging.place(tile)} on date {colored_logging.time(o_target_date)} from "
                        f"{colored_logging.val(GFS_cell_size)}m to {colored_logging.val(HLS_cell_size)}m resolution")

            # Extract the most recent relative humidity and the forecast product
            o_relative_humidity_prior = o_most_recent["RH"]
            o_relative_humidity_gfs = forecast_RH(time_UTC=o_time_utc, geometry=o_gfs_geometry, directory=s_gfs_download_directory, listing=of_gfs_listing)

            # Apply the bias correction if called for
            # if apply_GEOS5FP_GFS_bias_correction:
            #     # Request the forecast data
            #     o_matching_relative_humidity_gfs = forecast_RH(time_UTC=o_lance_processing_datetime_UTC, geometry=o_gfs_geometry, directory=s_gfs_download_directory,
            #                                                    resampling="cubic", listing=of_gfs_listing)
            #
            #     # Request the relative humidity at the GFS resolution
            #     o_matching_relative_humidity_geos5fp = o_geos5fp_connection.RH(time_UTC=o_lance_processing_datetime_UTC, geometry=o_gfs_geometry, resampling="cubic")
            #
            #     # Calculate the bias
            #     o_relative_humidity_gfs_bias = o_matching_relative_humidity_gfs - o_matching_relative_humidity_geos5fp
            #     o_relative_humidity_gfs = o_relative_humidity_gfs - o_relative_humidity_gfs_bias
            #
            # # Bias correct the
            # o_relative_humidity = bias_correct(coarse_image=o_relative_humidity_gfs, fine_image=o_relative_humidity_prior)
            o_relative_humidity = o_relative_humidity_gfs.to_geometry(o_relative_humidity_prior.geometry, resampling='linear')

            # Attempt to calculate the forecast
            try:
                LANCE_GFS_forecast(target_date=o_target_date,
                                   geometry=o_hls_geometry,
                                   coarse_geometry=o_gfs_geometry,
                                   coarse_cell_size=GFS_cell_size,
                                   target=tile,
                                   working_directory=s_working_directory,
                                   static_directory=s_static_directory,
                                   SRTM_connection=o_srtm_connection,
                                   SRTM_download=s_srtm_download_directory,
                                   GEDI_connection=GEDI_connection,
                                   GEDI_download=GEDI_download,
                                   ORNL_connection=ORNL_connection,
                                   CI_directory=CI_directory,
                                   soil_grids_connection=soil_grids_connection,
                                   soil_grids_download=soil_grids_download,
                                   LANCE_download_directory=s_lance_download_directory,
                                   intermediate_directory=intermediate_directory,
                                   preview_quality=preview_quality,
                                   ANN_model=ANN_model,
                                   ANN_model_filename=ANN_model_filename,
                                   model=model,
                                   model_name=ET_model_name,
                                   ST_C=o_landsat_scene_st_celsius,
                                   emissivity=o_emissivity,
                                   NDVI=o_ndvi,
                                   albedo=o_albedo,
                                   SM=o_soil_moisture,
                                   Ta_C=o_air_temperature_celsius,
                                   RH=o_relative_humidity,
                                   SWin=o_swin,
                                   water=bm_water_hls,
                                   GFS_listing=of_gfs_listing,
                                   target_variables=sl_target_variables,
                                   downscale_air=downscale_air,
                                   downscale_humidity=downscale_humidity,
                                   downscale_moisture=downscale_moisture,
                                   apply_GEOS5FP_GFS_bias_correction=apply_GEOS5FP_GFS_bias_correction,
                                   LANCE_processing_date=o_lance_processing_date,
                                   resampling=resampling,
                                   show_distribution=show_distribution,
                                   load_previous=load_previous,
                                   save_intermediate=save_intermediate)

            except Exception as e:
                # An error was encountered with the forecast and values could not be calculated.
                logger.exception(e)
                logger.warning(f"LANCE GFS cannot be processed for date: {o_target_date}")
                continue