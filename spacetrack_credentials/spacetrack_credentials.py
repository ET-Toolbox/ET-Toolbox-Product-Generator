from typing import Dict
from os.path import join, abspath, dirname, exists, expanduser
from credentials import get_credentials

FILENAME = join(abspath(expanduser("~")), ".spacetrack_credentials")

class SpaceTrackCredentials:
    def __init__(self, username, password):
        self.username = str(username)
        self.password = str(password)

def get_spacetrack_credentials(filename: str = FILENAME) -> Dict[str, str]:
    if filename is None or not exists(filename):
        filename = FILENAME
    
    credentials = get_credentials(
        filename=filename,
        displayed=["username"],
        hidden=["password"],
        prompt="credentials for Spacetrack https://www.space-track.org/auth/createAccount"
    )

    username = credentials["username"]
    password = credentials["password"]
    spacetrack_credentials = SpaceTrackCredentials(username, password)

    return spacetrack_credentials
