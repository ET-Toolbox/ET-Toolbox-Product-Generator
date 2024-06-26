"""
This module contains the unit tests for HDF5 compatibility.
"""

import os
import unittest

__author__ = "Gregory Halverson"


class TestHDF5(unittest.TestCase):
    FLOATING_POINT_ERROR_TOLERANCE = 0.0000001
    HDF5_TEST_FILE = 'hdf5_test_file.nc'

    def test_io_hdf5(self):
        import numpy as np
        import h5py

        # generate coordinate vectors
        lat_vector = np.linspace(90, -90, num=180 + 1)[1:-1]
        lon_vector = np.linspace(-180, 180, num=360 + 1)[1:-1]

        # calculate matrix shape
        rows = len(lat_vector)
        cols = len(lon_vector)

        # grid coordinates
        lon_matrix, lat_matrix = np.meshgrid(lon_vector, lat_vector)

        # generate matrix of normal floats
        data = np.random.randn(rows, cols)

        # remove test file if it already exists
        if os.path.exists(self.HDF5_TEST_FILE):
            os.remove(self.HDF5_TEST_FILE)

        # write test file
        with h5py.File(self.HDF5_TEST_FILE, 'w') as hdf5_file:
            hdf5_file.create_dataset('data', data=data)
            hdf5_file.create_dataset('lat', data=lat_matrix)
            hdf5_file.create_dataset('lon', data=lon_matrix)

        # read test file
        with h5py.File(self.HDF5_TEST_FILE, 'r') as hdf5_file:
            retrieved_data = np.array(hdf5_file.get('data'))

        # check that the same data was both written and retrieved
        assert ((retrieved_data - data).mean() < self.FLOATING_POINT_ERROR_TOLERANCE)

        # delete test file
        if os.path.exists(self.HDF5_TEST_FILE):
            os.remove(self.HDF5_TEST_FILE)
