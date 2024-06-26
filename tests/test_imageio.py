"""
This module contains the unit tests for imageio compatibility.
"""

import numpy as np
import os
import unittest

__author__ = "Gregory Halverson"


class TestImageIO(unittest.TestCase):
    def test_import_imageio(self):
        import imageio
