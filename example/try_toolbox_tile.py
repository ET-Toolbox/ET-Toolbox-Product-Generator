from ETtoolbox import ET_toolbox_hindcast_forecast_tile

working_directory = "~/data/ET_toolbox_testing"
static_directory = "~/data/L3T_L4T_static"
SRTM_directory = "~/data/SRTM_download_directory"
tile = "11SPS"

ET_toolbox_hindcast_forecast_tile(
    tile=tile,
    working_directory=working_directory,
    static_directory=static_directory,
    SRTM_download_directory=SRTM_directory
)
