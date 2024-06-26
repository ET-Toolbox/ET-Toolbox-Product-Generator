"""
This module contains the unit tests for the shapely package.
"""

import os
import unittest

__author__ = "Gregory Halverson"

directory = os.path.abspath(os.path.dirname(__file__))


class TestShapely(unittest.TestCase):
    def test_import_shapely(self):
        print('testing shapely import')
        import shapely
        print("shapely version: {}".format(shapely.__version__))
