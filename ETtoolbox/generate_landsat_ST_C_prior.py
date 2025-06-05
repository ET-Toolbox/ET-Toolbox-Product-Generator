from typing import Union
from datetime import date
from rasters import RasterGrid

def generate_landsat_ST_C_prior(
        date_UTC: Union[date, str],
        geometry: RasterGrid,
        target_name: str,
        landsat: LandsatL2C2 = None,
        working_directory: str = None,
        download_directory: str = None,
        landsat_initialization_days: int = LANDSAT_INITIALIZATION_DAYS) -> Raster:
    if isinstance(date_UTC, str):
        date_UTC = parser.parse(date_UTC).date()

    if landsat is None:
        landsat = LandsatL2C2(
            working_directory=working_directory,
            download_directory=download_directory
        )

    landsat_start = date_UTC - timedelta(days=landsat_initialization_days)
    landsat_end = date_UTC - timedelta(days=1)
    logger.info(f"generating Landsat temperature composite from {colored_logging.time(landsat_start)} to {colored_logging.time(landsat_end)}")
    landsat_listing = landsat.scene_search(start=landsat_start, end=landsat_end, target_geometry=geometry)
    landsat_composite_dates = sorted(set(landsat_listing.date_UTC))
    logger.info(f"found Landsat granules on dates: {', '.join([colored_logging.time(d) for d in landsat_composite_dates])}")

    ST_C_images = []

    for date_UTC in landsat_composite_dates:
        try:
            ST_C = landsat.product(acquisition_date=date_UTC, product="ST_C", geometry=geometry, target_name=target_name)
            ST_C_images.append(ST_C)
        except Exception as e:
            logger.warning(e)
            continue

    composite = Raster(np.nanmedian(np.stack(ST_C_images), axis=0), geometry=geometry)

    return composite