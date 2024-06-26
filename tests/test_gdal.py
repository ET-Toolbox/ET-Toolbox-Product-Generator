"""
This module contains the unit tests for the gdal package.
"""

import os
import unittest

__author__ = "Gregory Halverson"

directory = os.path.abspath(os.path.dirname(__file__))


class TestGdal(unittest.TestCase):
    def test_import_gdal(self):
        print('testing gdal import')
        from osgeo import gdal
        print("gdal version: {}".format(gdal.__version__))
