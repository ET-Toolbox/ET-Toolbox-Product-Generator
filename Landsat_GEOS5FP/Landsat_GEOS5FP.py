from LandsatL2C2 import LandsatL2C2
from PTJPLSM import PTJPLSM
from datetime import datetime, date, timedelta
import posixpath
from typing import Union, List
from os.path import join, abspath, expanduser, splitext, exists, basename
from sentinel import sentinel_tile_grid
import logging
from dateutil import parser
from typing import Dict, Callable
from gedi_canopy_height import GEDICanopyHeight
from geos5fp import GEOS5FP
from modisci import MODISCI
from SRTM import SRTM
from PTJPL import PTJPL
from soil_grids import SoilGrids
from glob import glob
import numpy as np
import rasters as rt
import cl

DEFAULT_Landsat_DOWNLOAD_DIRECTORY = "Landsat_download"
DEFAULT_Landsat_OUTPUT_DIRECTORY = "Landsat_output"
DEFAULT_RESAMPLING = "cubic"
DEFAULT_PREVIEW_QUALITY = 20
DEFAULT_TARGET_VARIABLES = ["LE", "ET", "ESI"]

logger = logging.getLogger(__name__)

class LandsatNotAvailableError(Exception):
    pass

class GEOS5FPNotAvailableError(Exception):
    pass

def UTC_to_solar(time_UTC: datetime, lon: float) -> datetime:
    return time_UTC + timedelta(hours=(np.radians(lon) / np.pi * 12))

def solar_to_UTC(time_solar: datetime, lon: float) -> datetime:
    return time_solar - timedelta(hours=(np.radians(lon) / np.pi * 12))

def generate_Landsat_output_directory(
        Landsat_output_directory: str, 
        target_date: Union[date, str], 
        target: str):
    return join(
        Landsat_output_directory, 
        f"{target_date:%Y-%m-%d}", 
        f"Landsat_{target_date:%Y-%m-%d}_{target}", 
    )

def generate_Landsat_output_filename(
        Landsat_output_directory: str, 
        target_date: Union[date, str], 
        time_UTC: Union[datetime, str], 
        target: str,
        product: str):
    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()
    
    if isinstance(time_UTC, str):
        time_UTC = parser.parse(time_UTC)

    directory = generate_Landsat_output_directory(
        Landsat_output_directory=Landsat_output_directory,
        target_date=target_date,
        target=target
    )

    filename = join(directory, f"Landsat_{time_UTC:%Y.%m.%d.%H.%M.%S}_{target}_{product}.tif")

    return filename

def check_Landsat_already_processed(
        Landsat_output_directory: str, 
        target_date: Union[date, str], 
        time_UTC: Union[datetime, str], 
        target: str,
        products: List[str]):
    already_processed = True
    logger.info(f"checking if Landsat GEOS-5 FP has previously been processed at {cl.place(target)} on {cl.time(target_date)}")
    
    for product in products:
        filename = generate_Landsat_output_filename(
            Landsat_output_directory=Landsat_output_directory,
            target_date=target_date,
            time_UTC=time_UTC,
            target=target,
            product=product
        )

        if exists(filename):
            logger.info(f"found previous Landsat GEOS-5 FP {cl.name(product)} at {cl.place(target)} on {cl.time(target_date)}: {cl.file(filename)}")
        else:
            logger.info(f"did not find previous Landsat GEOS-5 FP {cl.name(product)} at {cl.place(target)} on {cl.time(target_date)}")
            already_processed = False
    
    return already_processed

def load_Landsat(Landsat_output_directory: str, target_date: Union[date, str], target: str):
    dataset = {}    

    directory = generate_Landsat_output_directory(
        Landsat_output_directory=Landsat_output_directory,
        target_date=target_date,
        target=target
    )

    filenames = glob(join(directory, "*.tif"))

    for filename in filenames:
        logger.info(f"loading Landsat GEOS-5 FP file: {cl.file(filename)}")
        product = splitext(basename(filename))[0].split("_")[-1]
        image = rt.Raster.open(filename)
        dataset[product] = image
    
    return dataset

def Landsat_GEOS5FP(
        target_date: Union[date, str],
        geometry: rt.RasterGeometry,
        target: str,
        ST_C: rt.Raster = None,
        emissivity: rt.Raster = None,
        NDVI: rt.Raster = None,
        albedo: rt.Raster = None,
        SWin: rt.Raster = None,
        Rn: rt.Raster = None, 
        SM: rt.Raster = None,
        wind_speed: rt.Raster = None,
        Ta_C: rt.Raster = None,
        RH: rt.Raster = None,
        water: rt.Raster = None,
        model: PTJPLSM = None,
        working_directory: str = None,
        static_directory: str = None,
        Landsat_download_directory: str = None,
        Landsat_output_directory: str = None,
        SRTM_connection: SRTM = None,
        SRTM_download: str = None,
        GEOS5FP_connection: GEOS5FP = None,
        GEOS5FP_download: str = None,
        GEOS5FP_products: str = None,
        GEDI_connection: GEDICanopyHeight = None,
        GEDI_download: str = None,
        ORNL_connection: MODISCI = None,
        CI_directory: str = None,
        soil_grids_connection: SoilGrids = None,
        soil_grids_download: str = None,
        intermediate_directory=None,
        preview_quality: int = DEFAULT_PREVIEW_QUALITY,
        ANN_model: Callable = None,
        ANN_model_filename: str = None,
        resampling: str = DEFAULT_RESAMPLING,
        downscale_air: bool = True,
        downscale_humidity: bool = True,
        save_intermediate: bool = False,
        include_preview: bool = True,
        show_distribution: bool = True,
        load_previous: bool = True,
        target_variables: List[str] = DEFAULT_TARGET_VARIABLES) -> Dict[str, rt.Raster]:
    results = {}
    
    logger.info(f"processing Landsat GOES-5 FP for target {cl.name(target)} ({cl.val(geometry.cell_width)}m, {cl.val(geometry.rows)} rows {cl.val(geometry.cols)} cols)")

    if isinstance(target_date, str):
        target_date = parser.parse(target_date).date()

    logger.info(f"Landsat target date: {cl.time(target_date)}")
    time_solar = datetime(target_date.year, target_date.month, target_date.day, 13, 30)
    logger.info(f"Landsat target time solar: {cl.time(time_solar)}")
    time_UTC = solar_to_UTC(time_solar, geometry.centroid.latlon.x)
    logger.info(f"Landsat target time UTC: {cl.time(time_UTC)}")

    if working_directory is None:
        working_directory = "."
    
    working_directory = abspath(expanduser(working_directory))
    
    logger.info(f"Landsat working directory: {cl.dir(working_directory)}")
    
    if Landsat_download_directory is None:
        Landsat_download_directory = join(working_directory, DEFAULT_Landsat_DOWNLOAD_DIRECTORY)
    
    logger.info(f"Landsat download directory: {cl.dir(Landsat_download_directory)}")

    if Landsat_output_directory is None:
        Landsat_output_directory = join(working_directory, DEFAULT_Landsat_OUTPUT_DIRECTORY)

    logger.info(f"Landsat output directory: {cl.dir(Landsat_output_directory)}")

    Landsat_already_processed = check_Landsat_already_processed(
        Landsat_output_directory=Landsat_output_directory, 
        target_date=target_date, 
        time_UTC=time_UTC, 
        target=target,
        products=target_variables
    )

    if Landsat_already_processed:
        if load_previous:
            logger.info("loading previously generated Landsat GEOS-5 FP output")
            return load_Landsat(
                Landsat_output_directory=Landsat_output_directory,
                target_date=target_date,
                target=target
            )
        else:
            return

    if GEOS5FP_connection is None:
        try:
            logger.info(f"connecting to GEOS-5 FP")
            GEOS5FP_connection = GEOS5FP(
                working_directory=working_directory,
                download_directory=GEOS5FP_download,
                products_directory=GEOS5FP_products,
            )
        except Exception as e:
            logger.exception(e)
            raise GEOS5FPNotAvailableError("unable to connect to GEOS-5 FP")

    GEOS5FP_connection = GEOS5FP_connection
    latest_GEOS5FP_time = GEOS5FP_connection.latest_time_available

    if time_UTC.strftime("%Y-%m-%d %H:%M:%S") > latest_GEOS5FP_time.strftime("%Y-%m-%d %H:%M:%S"):
        raise GEOS5FPNotAvailableError(f"Landsat target time {time_UTC} is past latest available GEOS-5 FP time {latest_GEOS5FP_time}")

    Landsat_processing_date = target_date
    Landsat_processing_time = time_UTC

    landsat = LandsatL2C2(
        working_directory=working_directory,
        download_directory=Landsat_download_directory
    )

    if ST_C is None:
        logger.info(f"retrieving {cl.name('ST_C')} from Landsat on {cl.time(Landsat_processing_date)}")
        
        ST_C = landsat.product(
            acquisition_date=target_date,
            product="ST_C",
            geometry=geometry,
            target_name=target
        )

    results["ST_C"] = ST_C
    
    if emissivity is None:
        logger.info(f"retrieving {cl.name('emissivity')} from Landsat on {cl.time(Landsat_processing_date)}")
        
        emissivity = landsat.product(
            acquisition_date=target_date,
            product="emissivity",
            geometry=geometry,
            target_name=target
        )



    results["emissivity"] = emissivity

    if NDVI is None:
        logger.info(f"retrieving {cl.name('NDVI')} from Landsat on {cl.time(Landsat_processing_date)}")

        NDVI = landsat.product(
            acquisition_date=target_date,
            product="NDVI",
            geometry=geometry,
            target_name=target
        )

    results["NDVI"] = NDVI
    
    if albedo is None:
        logger.info(f"retrieving {cl.name('VNP43MA4N')} {cl.name('albedo')} from Landsat on {cl.time(Landsat_processing_date)}")

        albedo = landsat.product(
            acquisition_date=target_date,
            product="albedo",
            geometry=geometry,
            target_name=target
        )

    results["albedo"] = albedo
    
    if model is None:
        model = PTJPLSM(
            working_directory=working_directory,
            static_directory=static_directory,
            SRTM_connection=SRTM_connection,
            SRTM_download=SRTM_download,
            GEOS5FP_connection=GEOS5FP_connection,
            GEOS5FP_download=GEOS5FP_download,
            GEOS5FP_products=GEOS5FP_products,
            GEDI_connection=GEDI_connection,
            GEDI_download=GEDI_download,
            ORNL_connection=ORNL_connection,
            CI_directory=CI_directory,
            soil_grids_connection=soil_grids_connection,
            soil_grids_download=soil_grids_connection,
            intermediate_directory=intermediate_directory,
            preview_quality=preview_quality,
            ANN_model=ANN_model,
            ANN_model_filename=ANN_model_filename,
            resampling=resampling,
            downscale_air=downscale_air,
            downscale_humidity=downscale_humidity,
            save_intermediate=save_intermediate,
            include_preview=include_preview,
            show_distribution=show_distribution
        )

    logger.info(f"running PT-JPL-SM ET model forecast at {cl.time(time_UTC)}")
        
    PTJPL_results = model.PTJPL(
        geometry=geometry,
        target=target,
        time_UTC=time_UTC,
        ST_C=ST_C,
        emissivity=emissivity,
        NDVI=NDVI,
        albedo=albedo,
        SWin=SWin,
        SM=SM,
        wind_speed=wind_speed,
        Ta_C=Ta_C,
        RH=RH,
        water=water
    )

    for k, v in PTJPL_results.items():
        results[k] = v

    for product, image in results.items():
        filename = generate_Landsat_output_filename(
            Landsat_output_directory=Landsat_output_directory,
            target_date=target_date,
            time_UTC=time_UTC,
            target=target,
            product=product
        )

        logger.info(f"writing Landsat GEOS-5 FP {cl.name(product)} at {cl.place(target)} at {cl.time(time_UTC)} to file: {cl.file(filename)}")
        image.to_geotiff(filename)

    return results
