from ETtoolbox.ETtoolbox_hindcast_coarse import ET_toolbox_hindcast_coarse_tile

working_directory = "~/data/coarse_NRT_testing"
static_directory = "~/data/L3T_L4T_static"
tile = "11SPS"

ET_toolbox_hindcast_coarse_tile(tile=tile, working_directory=working_directory, static_directory=static_directory)
