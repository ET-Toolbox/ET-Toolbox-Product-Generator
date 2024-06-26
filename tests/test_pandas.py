"""
This module contains the unit tests for the pandas package.
"""

import unittest
from os.path import join, abspath, dirname

__author__ = "Gregory Halverson"


class TestPandas(unittest.TestCase):
    def test_import_pandas(self):
        print('testing pandas import')
        import pandas as pd
        print("pandas version: {}".format(pd.__version__))

