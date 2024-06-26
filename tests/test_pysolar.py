"""
This module contains the unit tests for pysolar compatibility.
"""

import numpy as np
import os
import unittest

__author__ = "Gregory Halverson"


class TestPySolar(unittest.TestCase):
    def test_import_pysolar(self):
        from pysolar.solar import get_azimuth
        from pysolar.solar import get_altitude
