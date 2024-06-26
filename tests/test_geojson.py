"""
This module contains the unit tests for GeoJSON compatibility.
"""

import os

import unittest

directory = os.path.abspath(os.path.dirname(__file__))

__author__ = "Gregory Halverson"


class TestGeoJSON(unittest.TestCase):
    def test_geojson(self):
        import fiona
        from fiona.crs import from_string

        output_filename = os.path.join(
            directory,
            'test_output.geojson'
        )

        schema = {
            'geometry': 'Polygon',
            'properties': {}
        }

        geometry = {"type": "Polygon", "coordinates": [
            [[-112.80124510641818, 32.71172508145989], [-117.32868536317275, 31.97159632359229],
             [-118.48335056504993, 35.38814771486749], [-113.78231658739557, 36.16082226259867],
             [-112.80124510641818, 32.71172508145989]]]}

        records = [
            {
                'geometry': geometry,
                'id': 0,
                'properties': {}
            },
        ]

        crs = from_string('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')

        driver = 'GeoJSON'

        # when you write a geojson file
        # you MUST make sure the file doesn't already exist
        if os.path.exists(output_filename):
            os.remove(output_filename)

        with fiona.open(
                output_filename,
                mode='w',
                crs=crs,
                driver=driver,
                schema=schema) as geojson_file:
            geojson_file.writerecords(records)

        os.remove(output_filename)
