"""
This module contains the unit tests for the geopandas package.
"""

import os
import unittest

__author__ = "Gregory Halverson"

directory = os.path.abspath(os.path.dirname(__file__))


class TestGeoPandas(unittest.TestCase):
    def test_import_geopandas(self):
        print('testing geopandas import')
        import geopandas as gp
        print("geopandas version: {}".format(gp.__version__))
