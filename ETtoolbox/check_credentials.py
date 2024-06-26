import sys

from ERS_credentials import get_ERS_credentials
from spacetrack_credentials import get_spacetrack_credentials

def main(argv=sys.argv):
    print("checking EROS Registration System credentials")
    ERS_credentials = get_ERS_credentials()

    for key, value in ERS_credentials.items():
        print(f"{key}: {value}")

    print("checking Spacetrack credentials")
    spacetrack_credentials = get_spacetrack_credentials()

    for key, value in spacetrack_credentials.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    sys.exit(main(argv=sys.argv))
