# ==========================================================================
# Author: Hoang Anh Quan
# Purpose: Refresh API KEY
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
from dotenv import find_dotenv, dotenv_values
import httpx

from utils.path import ACCESS_TOKEN_PATH
from utils.api_util import (
    base_url,
)

# ==========================================================================
# PARAMETERS
# ==========================================================================
env_path = find_dotenv()
config = dotenv_values(env_path)
REFRESH_TOKEN = config.get("REFRESH_TOKEN")
if REFRESH_TOKEN is None:
    REFRESH_TOKEN = ""

base_url = base_url


# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================


class RefreshToken:
    def __init__(self, refresh_token: str = REFRESH_TOKEN):
        header = {}
        self.token = refresh_token
        self.client = httpx.Client(
            base_url=base_url,
            headers=header,
        )

    def refresh(self):
        body = {"refreshToken": self.token}

        response = self.client.post(
            url="/v2/tokens/refresh",
            json=body,
        )


# ==========================================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================================
