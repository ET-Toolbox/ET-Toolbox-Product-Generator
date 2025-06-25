import os, base64, json, requests, time
from typing import Dict


def get_ers_credentials() -> Dict[str, str]:
    """
    Converts the ERS environmental variables to the internal Python format

    Returns
    -------
    c_credentials: dict
        Authentication credentials for ERS

    """

    # Authenticate if the token isn't available
    if 'EROS_BEARER' not in os.environ:

        # Request API key
        s_service_url = "https://m2m.cr.usgs.gov/api/api/json/stable/login-token"
        c_payload = {"username": os.environ['EROS_USERNAME'], "token": os.environ["EROS_TOKEN"]}

        # Issue the command
        i_number_of_attempts = 3
        i_attempt_counter = 0

        # Attempt to reauthenticate if a response is not received
        while i_attempt_counter < i_number_of_attempts:
            try:
                # Construct the command
                json_data = json.dumps(c_payload)
                s_response = requests.post(s_service_url, json_data)

                # Parse the response
                c_response = json.loads(s_response.content)
                s_token_bearer = c_response['data']

                # Set global token value
                os.environ['EROS_BEARER'] = s_token_bearer

                # Break from the loop
                break

            except:
                # Authentication failed. Wait a few seconds and attempt again.
                time.sleep(10)
                i_attempt_counter += 1

        # Confirm that authentication is successful
        if i_attempt_counter == i_number_of_attempts:
            # Authentication has not been successful. Raise an error to stop program execution
            raise ConnectionError('Not able to authenticate to USGS EROS.')

    # Create the dictionary
    c_credentials = {'username': os.environ["EROS_USERNAME"],
                     'header': os.environ['EROS_BEARER'],
                     'password': os.environ["EROS_PASSWORD"]}

    # Return to the calling function
    return c_credentials


def get_earthdata_credentials() -> Dict[str, str]:
    """
    Authenticates and request M2M token and converts environmental variables to the internal python format

    Returns
    -------
    c_credentials: dict
        Contains M2M authentication information

    """


    # Set the user token
    s_token_earthdata = get_earthdata_user_tokens()

    # Create the authentication dictionary
    c_credentials = {"username": os.environ["EARTHDATA_USERNAME"],
                     "password": os.environ["EARTHDATA_PASSWORD"],
                     "earthdata_token": s_token_earthdata}

    # Return to the aclling function
    return c_credentials


# todo: remove this object
class SpaceTrackCredentials:
    def __init__(self):
        self.username = os.environ["SPACETRACK_USERNAME"]
        self.password = os.environ["SPACETRACK_PASSWORD"]


def get_spacetrack_credentials() -> object:
    """
    Creates and returns the spacetrack credential object

    Returns
    -------
    o_spacetrack_credentials: object
        Contains the spacetrack username and password
    """

    # Create the object
    o_spacetrack_credentials = SpaceTrackCredentials()

    # Return to the calling function
    return o_spacetrack_credentials


def encode_earthdata_credentials() -> str:
    """
    Converts the m2m credentials from plaintext to base64 bytes. The information must be in this format when passing information to the API.

    Returns
    -------
    s_combined_auth_base64_bytes: str
        Converted base64 byte string

    """

    # Convert auth information to base 64 bytes for transfer
    s_combined_auth = os.environ["EARTHDATA_USERNAME"] + ":" + os.environ["EARTHDATA_PASSWORD"]
    s_combined_auth_bytes = s_combined_auth.encode("ascii")
    s_combined_auth_base64_bytes = base64.b64encode(s_combined_auth_bytes)

    # Return to the calling function
    return s_combined_auth_base64_bytes


def get_earthdata_user_tokens() -> str:
    """
    Requests and sets the access token from the API

    Returns
    -------
    s_token: str
        Access token string from the earthdata API

    """

    # Authenticate only if token isn't available
    if "EARTHDATA_TOKEN" not in os.environ:

        # Encode the user credentials
        s_combined_auth_base64_bytes = encode_earthdata_credentials()

        # Formulate the request command
        s_command = 'curl --request POST --url https://urs.earthdata.nasa.gov/api/users/find_or_create_token -H "Authorization: Basic (' + \
                     str(s_combined_auth_base64_bytes)[2:-1] + ')"'

        # Attempt authentication
        i_number_of_attempts = 3
        i_attempt_counter = 0

        # Attempt to reauthenticate if a response is not received
        while i_attempt_counter < i_number_of_attempts:
            try:
                # Make the command
                s_response = os.popen(s_command).readlines()

                # Parse the response
                c_response = json.loads(s_response[0])
                s_token_earthdata = c_response['access_token']

                # Set global token value
                os.environ["EARTHDATA_TOKEN"] = s_token_earthdata

                # Attempt successful. Break from the loop
                break

            except:
                # Authentication failed. Wait a few seconds and attempt again.
                time.sleep(10)
                i_attempt_counter += 1

        # Confirm that authentication is successful
        if i_attempt_counter == i_number_of_attempts:
            # Authentication has not been successful. Raise an error to stop program execution
            raise ConnectionError('Not able to authenticate to NASA Earthdata.')

    else:
        # Token already exists. Reuse it.
        s_token_earthdata = os.environ["EARTHDATA_TOKEN"]

    # Return to the calling function
    return s_token_earthdata


def revoke_earthdata_user_token():
    """
    Revokes the m2m access token for general cleanup

    Returns
    -------
    None.

    """

    # Revoke the token only if it exists
    if os.environ["EARTHDATA_TOKEN"] is not None:
        # Encode the user credentials
        s_combined_auth_base64_bytes = encode_earthdata_credentials()

        # Revoke the token
        s_command = 'curl --request POST --url https://urs.earthdata.nasa.gov/api/users/revoke_token?token=' + os.environ["EARTHDATA_TOKEN"] + ' -H "Authorization: Basic (' + \
                    str(s_combined_auth_base64_bytes)[2:-1] + ')"'

        # Make the command
        s_response = os.system(s_command)