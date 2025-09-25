import sys
from os.path import join

from ETtoolbox import ET_toolbox_hindcast_forecast_tile


def main(argv=sys.argv):
    RIO_GRANDE_TILES = [
        "13SBB",
        "13SCB",
        "13SDB",
        "13SCA",
        "13SDA",
        "13SCV",
        "13SDV",
        "13SCU",
        "13SBT",
        "13SCT",
        "13SBS",
        "13SCS",
        "13SCR"
    ]

    if "--working" in argv:
        working_directory = argv[argv.index("--working") + 1]
    else:
        working_directory = "."

    if "--static" in argv:
        static_directory = argv[argv.index("--static") + 1]
    else:
        static_directory = join(working_directory, "PTJPL_static")

    if "--SRTM" in argv:
        SRTM_download = argv[argv.index("--SRTM") + 1]
    else:
        SRTM_download = join(working_directory, "SRTM_download_directory")

    if "--LANCE" in argv:
        LANCE_download_directory = argv[argv.index("--LANCE") + 1]
    else:
        LANCE_download_directory = join(working_directory, "LANCE_download_directory")

    if "--GEOS5FP" in argv:
        GEOS5FP_download = argv[argv.index("--GEOS5FP") + 1]
    else:
        GEOS5FP_download = join(working_directory, "GEOS5FP_download_directory")

    for tile in RIO_GRANDE_TILES:
        ET_toolbox_hindcast_forecast_tile(
            tile=tile,
            working_directory=working_directory,
            static_directory=static_directory,
            SRTM_download_directory=SRTM_download,
            LANCE_download_directory=LANCE_download_directory,
            GEOS5FP_download_directory=GEOS5FP_download,
        )
