"""
This module contains the unit tests for affine compatibility.
"""

import numpy as np
import os
import unittest

__author__ = "Gregory Halverson"


class TestAffine(unittest.TestCase):
    def test_import_affine(self):
        from affine import Affine
