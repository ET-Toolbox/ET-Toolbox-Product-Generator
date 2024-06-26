"""
This module contains the unit tests for ephem compatibility.
"""

import numpy as np
import os
import unittest

__author__ = "Gregory Halverson"


class TestEphem(unittest.TestCase):
    def test_import_ephem(self):
        import ephem
