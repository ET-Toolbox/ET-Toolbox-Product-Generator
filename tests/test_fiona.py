"""
This module contains the unit tests for the fiona package.
"""

import os
import unittest

__author__ = "Gregory Halverson"

directory = os.path.abspath(os.path.dirname(__file__))


class TestFiona(unittest.TestCase):
    def test_import_fiona(self):
        import fiona
