from .WRS2Descending import *

with open(join(abspath(dirname(__file__)), "version.txt")) as f:
    version = f.read()

__version__ = version
__author__ = "Gregory H. Halverson"
