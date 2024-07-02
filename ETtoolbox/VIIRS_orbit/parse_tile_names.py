import json
from os.path import exists
from typing import Iterable

from six import string_types


def parse_tile_names(tile_names_argument):
    if isinstance(tile_names_argument, string_types):
        if tile_names_argument.endswith(".json") and exists(tile_names_argument):
            with open(tile_names_argument, "r") as f:
                tile_list = [item.strip() for item in json.loads(f.read(1))]
        else:
            tile_list = [item.strip() for item in tile_names_argument.split(",")]
    elif isinstance(tile_names_argument, Iterable):
        tile_list = tile_names_argument
    else:
        raise ValueError("invalid tile names")

    for tile_name in tile_list:
        if not isinstance(tile_name, string_types):
            raise ValueError("invalid tile names")

        if not tile_name[0] == "h" or not tile_name[3] == "v":
            raise ValueError("invalid tile name: {}".format(tile_name))

        try:
            h = int(tile_name[1:3])
            v = int(tile_name[4:6])
        except:
            raise ValueError("invalid tile name: {}".format(tile_name))

    return tile_list