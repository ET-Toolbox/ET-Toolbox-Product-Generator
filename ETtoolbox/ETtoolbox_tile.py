import sys
from os.path import join

from ETtoolbox import ET_toolbox_hindcast_forecast_tile


def main(argv=sys.argv):
    tile = argv[1]

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

    if "--VIIRS" in argv:
        VIIRS_download_directory = argv[argv.index("--VIIRS") + 1]
    else:
        VIIRS_download_directory = join(working_directory, "VIIRS_download_directory")

    if "--GEOS5FP" in argv:
        GEOS5FP_download = argv[argv.index("--GEOS5FP") + 1]
    else:
        GEOS5FP_download = join(working_directory, "GEOS5FP_download_directory")

    ET_toolbox_hindcast_forecast_tile(
        tile=tile,
        working_directory=working_directory,
        static_directory=static_directory,
        SRTM_download_directory=SRTM_download,
        VIIRS_download_directory=VIIRS_download_directory,
        GEOS5FP_download_directory=GEOS5FP_download,
    )

if __name__ == "__main__":
    sys.exit(main(argv=sys.argv))
