import base64
import json
from os import remove
from os.path import expanduser, abspath, exists
from typing import Dict, List
import getpass

def input_credentials(displayed: List[str] = (), hidden: List[str] = ()) -> Dict[str, str]:
    credentials = {}

    for key in displayed:
        credentials[key] = input(f"{key}: ")

    for key in hidden:
        credentials[key] = getpass.getpass(f"{key}: ")

    return credentials

def encode_credentials(credentials: Dict[str, str]) -> str:
    return base64.b64encode(json.dumps(credentials).encode()).decode()

def write_credentials(credentials: Dict[str, str], filename: str):
    filename = abspath(expanduser(filename))
    encoded = encode_credentials(credentials)

    with open(filename, "w") as file:
        file.write(encoded)

def decode_credentials(credential_text: str) -> Dict[str, str]:
    return json.loads(base64.b64decode(credential_text).decode())

def read_credentials(filename: str) -> Dict[str, str]:
    filename = abspath(expanduser(filename))

    print(f"reading credential file: {filename}")

    with open(filename, "r") as file:
        encoded = file.read()

    credentials = decode_credentials(encoded)

    return credentials

def get_credentials(
        filename: str,
        displayed: List[str] = (),
        hidden: List[str] = (),
        prompt: str = None,
        replace: bool = False) -> Dict[str, str]:
    filename = abspath(expanduser(filename))

    if replace and exists(filename):
        remove(filename)

    if exists(filename):
        return read_credentials(filename)

    if prompt is not None:
        print(prompt)

    credentials = input_credentials(displayed=displayed, hidden=hidden)
    write_credentials(credentials, filename)

    return credentials