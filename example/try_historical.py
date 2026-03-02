from ETtoolbox.ETtoolbox_historical_fine import ET_toolbox_historical_fine_tile

working_directory = "~/data/ET_toolbox_historical_fine_testing"
static_directory = "~/data/PTJPL_static"
SRTM_directory = "~/data/SRTM_download_directory"
tile = "11SPS"

ET_toolbox_historical_fine_tile(
    tile=tile,
    start_date="2022-07-01",
    end_date="2022-07-05",
    working_directory=working_directory,
    static_directory=static_directory,
    SRTM_download=SRTM_directory
)
