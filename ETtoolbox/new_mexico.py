import logging
import sys
from datetime import datetime
from os.path import join
from time import sleep

from ETtoolbox.ETtoolbox_hindcast_coarse import ET_toolbox_hindcast_coarse_tile

logger = logging.getLogger(__name__)


def new_mexico_VIIRS_server(
        working_directory: str = None,
        output_directory: str = None,
        output_bucket_name: str = None,
        static_directory: str = None,
        VIIRS_download_directory: str = None,
        NASADEM_download_directory: str = None,
        GEOS5FP_download_directory: str = None):
    if working_directory is None:
        working_directory = "/new_mexico_VIIRS"
    
    if output_directory is None:
        output_directory = "/VIIRS_output"

    if static_directory is None:
        static_directory = "/static"

    if VIIRS_download_directory is None:
        VIIRS_download_directory = "/VIIRS"

    if NASADEM_download_directory is None:
        NASADEM_download_directory = "/SRTM"

    if GEOS5FP_download_directory is None:
        GEOS5FP_download_directory = "/GEOS5FP"

    logger.info("starting New Mexico VIIRS data production")

    logger.info(f"working directory: {working_directory}")
    logger.info(f"static directory: {static_directory}")
    logger.info(f"VIIRS directory: {VIIRS_download_directory}")
    logger.info(f"SRTM directory: {NASADEM_download_directory}")
    logger.info(f"GEOS-5 FP directory: {GEOS5FP_download_directory}")

    # tiles = ["h08v05", "h09v05"]
    tiles = [
        '12RXV',
        '12RYV',
        '12SXA',
        '12SXB',
        '12SXC',
        '12SXD',
        '12SXE',
        '12SXF',
        '12SXG',
        '12SYA',
        '12SYB',
        '12SYC',
        '12SYD',
        '12SYE',
        '12SYF',
        '12SYG',
        '13SBA',
        '13SBB',
        '13SBR',
        '13SBS',
        '13SBT',
        '13SBU',
        '13SBV',
        '13SCA',
        '13SCB',
        '13SCR',
        '13SCS',
        '13SCT',
        '13SCU',
        '13SCV',
        '13SDA',
        '13SDB',
        '13SDR',
        '13SDS',
        '13SDT',
        '13SDU',
        '13SDV',
        '13SEA',
        '13SEB',
        '13SER',
        '13SES',
        '13SET',
        '13SEU',
        '13SEV',
        '13SFA',
        '13SFB',
        '13SFR',
        '13SFS',
        '13SFT',
        '13SFU',
        '13SFV'
    ]

    while (True):
        runtime = datetime.utcnow()
        logger.info(f"running New Nexico VIIRS data production at time {runtime} UTC")

        for tile in tiles:
            ET_toolbox_hindcast_coarse_tile(
                tile=tile,
                working_directory=working_directory,
                output_directory=output_directory,
                output_bucket_name=output_bucket_name,
                static_directory=static_directory,
                SRTM_download=NASADEM_download_directory,
                VIIRS_download_directory=VIIRS_download_directory,
                GEOS5FP_download=GEOS5FP_download_directory,
            )

        while (datetime.utcnow().hour % 3 != 0):
            sleep(60)


def main(argv=sys.argv):
    if "--working" in argv:
        working_directory = argv[argv.index("--working") + 1]
    else:
        working_directory = "."

    if "--output" in argv:
        output_directory = argv[argv.index("--output") + 1]
    else:
        output_directory = join(working_directory, "VIIRS_output")

    if "--bucket" in argv:
        output_bucket_name = argv[argv.index("--bucket") + 1]
    else:
        output_bucket_name = join(working_directory, "jpl-nmw-dev-viirs")

    if "--static" in argv:
        static_directory = argv[argv.index("--static") + 1]
    else:
        static_directory = join(working_directory, "PTJPL_static")

    if "--SRTM" in argv:
        SRTM_download = argv[argv.index("--SRTM") + 1]
    else:
        SRTM_download = join(working_directory, "SRTM_download_directory")

    if "--VIIRS" in argv:
        VIIRS_download = argv[argv.index("--VIIRS") + 1]
    else:
        VIIRS_download = join(working_directory, "VIIRS_download_directory")

    if "--GEOS5FP" in argv:
        GEOS5FP_download = argv[argv.index("--GEOS5FP") + 1]
    else:
        GEOS5FP_download = join(working_directory, "GEOS5FP_download_directory")

    return new_mexico_VIIRS_server(
        working_directory=working_directory,
        output_directory=output_directory,
        static_directory=static_directory,
        NASADEM_download_directory=SRTM_download,
        VIIRS_download_directory=VIIRS_download,
        GEOS5FP_download_directory=GEOS5FP_download
    )


if __name__ == "__main__":
    sys.exit(main(argv=sys.argv))
