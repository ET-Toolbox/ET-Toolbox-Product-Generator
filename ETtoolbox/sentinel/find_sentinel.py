import logging
import sys
from datetime import date
from typing import Union
from dateutil import parser
import colored_logging
from ETtoolbox.sentinel import Sentinel


def find_sentinel(start: Union[date, str], end: Union[date, str], tile: str):
    logger = logging.getLogger(__name__)

    logger.info(f"searching Sentinel L2A at {colored_logging.place(tile)}")

    sentinel = Sentinel()
    listing = sentinel.search_L2A(start_date=start, end_date=end, tile=tile)

    listing = listing.sort_values(by="date")
    logger.info(f"found {colored_logging.val(len(listing))} Sentinel scenes")

    for i, row in listing.iterrows():
        date_UTC = row.date
        granule_ID = row.ID

        logger.info("* date: " + colored_logging.time(f"{date_UTC} UTC") + " granule: " + colored_logging.val(granule_ID))


def main(argv=sys.argv):
    if "--start-date-UTC" in argv:
        start_date_UTC = parser.parse(argv[argv.index("--start-date-UTC") + 1]).date()
    elif "--start" in argv:
        start_date_UTC = parser.parse(argv[argv.index("--start") + 1]).date()
    else:
        start_date_UTC = None

    if "--end-date-UTC" in argv:
        end_date_UTC = parser.parse(argv[argv.index("--end-date-UTC") + 1]).date()
    elif "--end" in argv:
        end_date_UTC = parser.parse(argv[argv.index("--end") + 1]).date()
    else:
        end_date_UTC = None

    if "--tile" in argv:
        tile = argv[argv.index("--tile") + 1]
    else:
        tile = None

    find_sentinel(start=start_date_UTC, end=end_date_UTC, tile=tile)


if __name__ == "__main__":
    sys.exit(main(argv=sys.argv))
