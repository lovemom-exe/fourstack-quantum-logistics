# ==========================================================================
# Author: Hoang Anh Quan
# Purpose: Refresh API KEY
# ==========================================================================
# IMPORTS & MODULE LOADING
# ==========================================================================
import httpx

from utils.path import ACCESS_TOKEN_PATH, REFRESH_TOKEN_PATH
from utils.api_util import (
    base_url,
)

# ==========================================================================
# PARAMETERS
# ==========================================================================

with open(REFRESH_TOKEN_PATH, "r") as token:
    REFRESH_TOKEN = token.read().strip()

base_url = base_url


# ==========================================================================
# CORE LOGIC & FUNCTIONS
# ==========================================================================


class RefreshToken:
    def __init__(self, refresh_token: str = REFRESH_TOKEN):
        header = {"Content-Type": "application/json"}
        self.token = refresh_token
        self.client = httpx.Client(
            base_url=base_url,
            headers=header,
        )

    def refresh(self):
        body = {"refreshToken": self.token}

        response = self.client.post(
            url="/api/v2/tokens/refresh",
            json=body,
        )

        response.raise_for_status()
        # print(response.status_code)
        # print(response.headers)
        # print(response.text)
        new_access_token = response.json()["accessToken"]
        new_refresh_token = response.json()["refreshToken"]

        self._save(new_access_token, new_refresh_token)

    def _save(self, access_key: str, refresh_token: str):
        with open(ACCESS_TOKEN_PATH, "w") as file:
            file.write(access_key)

        with open(REFRESH_TOKEN_PATH, "w") as file:
            file.write(refresh_token)


# ==========================================================================
# MAIN EXECUTION ENTRYPOINT
# ==========================================================================


def main():
    re_token = RefreshToken()
    re_token.refresh()


if __name__ == "__main__":
    main()
