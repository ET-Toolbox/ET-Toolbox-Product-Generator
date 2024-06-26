"""
This module contains the unit tests for HDF4 compatibility.
"""

import os
import unittest

__author__ = "Gregory Halverson"


class TestHDF4(unittest.TestCase):
    FLOATING_POINT_ERROR_TOLERANCE = 0.0000001
    HDF4_TEST_FILE = 'hdf4_test_file.hdf'

    def test_io_hdf4(self):
        import numpy as np
        from pyhdf.SD import SD, SDC

        # generate coordinate vectors
        lat_vector = np.linspace(90, -90, num=180 + 1)[1:-1]
        lon_vector = np.linspace(-180, 180, num=360 + 1)[1:-1]

        # calculate matrix shape
        rows = len(lat_vector)
        cols = len(lon_vector)

        # grid coordinates
        lon_matrix, lat_matrix = np.meshgrid(lon_vector, lat_vector)

        # generate matrix of normal floats
        data = np.random.randn(rows, cols).astype(np.float32)

        # remove test file if it already exists
        if os.path.exists(self.HDF4_TEST_FILE):
            os.remove(self.HDF4_TEST_FILE)

        # open file for writing
        hdf4_file = SD(self.HDF4_TEST_FILE, SDC.WRITE | SDC.CREATE)

        # create scientific dataset
        data_sds = hdf4_file.create('data', SDC.FLOAT32, data.shape)

        # set dimensions
        lat_dim = data_sds.dim(0)
        lat_dim.setname('lat')
        lon_dim = data_sds.dim(1)
        lon_dim.setname('lon')

        # assign data
        data_sds[:, :] = data

        # close file
        data_sds.endaccess()
        hdf4_file.end()

        # open file for reading
        hdf4_file = SD(self.HDF4_TEST_FILE, SDC.READ)

        # load dataset
        data_sds = hdf4_file.select('data')

        # load matrix from dataset
        retrieved_data = data_sds[:, :]

        # close file
        data_sds.endaccess()
        hdf4_file.end()

        # check that the same data was both written and retrieved
        assert ((retrieved_data - data).mean() < self.FLOATING_POINT_ERROR_TOLERANCE)

        # delete test file
        if os.path.exists(self.HDF4_TEST_FILE):
            os.remove(self.HDF4_TEST_FILE)
