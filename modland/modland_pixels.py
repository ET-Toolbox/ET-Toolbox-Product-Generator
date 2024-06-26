from .MODIS_land_tile import MODISLandTile
from .latlon_to_MODLAND import latlon_to_MODLAND
from .sinusoidal_to_MODLAND import sinusoidal_to_MODLAND

# finds target containing a latitude and longitude coordinate
# then finds pixel within that target nearest to coordinate given size of matrix
# returns horizontal index, vertical index, row, and column as tuple
def latlon_to_MODLAND_pixel(latitude, longitude, rows_per_tile, columns_per_tile):
    horizontal_index, vertical_index = latlon_to_MODLAND(latitude, longitude)
    tile = MODISLandTile(horizontal_index, vertical_index, rows_per_tile, columns_per_tile)
    row, column = tile.row_column_from_latlon(latitude, longitude)

    return horizontal_index, vertical_index, row, column


# finds target containing a sinusoidal coordinate
# then finds pixel within that target nearest to coordinate given size of matrix
# returns horizontal index, vertical index, row, and column as tuple
def sinusoidal_to_MODLAND_pixel(x_sinusoidal, y_sinusoidal, rows_per_tile, columns_per_tile):
    horizontal_index, vertical_index = sinusoidal_to_MODLAND(x_sinusoidal, y_sinusoidal)
    tile = MODISLandTile(horizontal_index, vertical_index, rows_per_tile, columns_per_tile)
    row, column = tile.row_column_from_sinusoidal(x_sinusoidal, y_sinusoidal)

    return horizontal_index, vertical_index, row, column
