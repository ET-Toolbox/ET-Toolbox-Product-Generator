"""
This module contains the unit tests for netCDF4 compatibility.
"""

import os
import unittest

__author__ = "Gregory Halverson"


class TestNetCDF4(unittest.TestCase):
    FLOATING_POINT_ERROR_TOLERANCE = 0.0000001
    NETCDF_TEST_FILE = 'netcdf_test_file.nc'

    def test_import_netcdf4(self):
        print('testing netCDF4 import')
        import netCDF4
        print('netCDF4 version: {}'.format(netCDF4.__version__))

    def test_io_netcdf4(self):
        import numpy as np
        import netCDF4

        # generate coordinate vectors
        lat_vector = np.linspace(90, -90, num=180 + 1)[1:-1]
        lon_vector = np.linspace(-180, 180, num=360 + 1)[1:-1]

        # calculate matrix shape
        rows = len(lat_vector)
        cols = len(lon_vector)

        # generate matrix of normal floats
        data = np.random.randn(rows, cols)

        # remove test file if it already exists
        if os.path.exists(self.NETCDF_TEST_FILE):
            os.remove(self.NETCDF_TEST_FILE)

        # test writing file
        with netCDF4.Dataset(self.NETCDF_TEST_FILE, 'w') as netcdf_file:

            # create dimensions
            netcdf_file.createDimension('lat', rows)
            netcdf_file.createDimension('lon', cols)

            # create variables
            netcdf_lat = netcdf_file.createVariable('lat', 'f4', ('lat',))
            netcdf_lon = netcdf_file.createVariable('lon', 'f4', ('lon',))
            netcdf_data = netcdf_file.createVariable('data', 'f4', ('lat', 'lon'))

            # assign variables
            netcdf_lat[:] = lat_vector
            netcdf_lon[:] = lon_vector
            netcdf_data[:, :] = data

        # test reading file
        with netCDF4.Dataset(self.NETCDF_TEST_FILE, 'r') as netcdf_file:
            retrieved_data = netcdf_file.variables['data'][...]

        # check that the same data was both written and retrieved
        assert ((retrieved_data - data).mean() < self.FLOATING_POINT_ERROR_TOLERANCE)

        # delete test file
        if os.path.exists(self.NETCDF_TEST_FILE):
            os.remove(self.NETCDF_TEST_FILE)
