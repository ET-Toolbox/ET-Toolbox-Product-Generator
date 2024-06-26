import sys
from LANCE_GFS_forecast.LANCE_GFS_forecast import available_LANCE_dates

def main(argv=sys.argv):
    dates = available_LANCE_dates("VNP43MA4N")
    print("available LANCE VIIRS dates:")

    for d in dates:
        print(f"* {d:%Y-%m-%d}")

if __name__ == "__main__":
    sys.exit(main(argv=sys.argv))
