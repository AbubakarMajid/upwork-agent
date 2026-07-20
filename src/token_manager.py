"""Fetches and caches Upwork's visitor_gql_token cookie via TLS-fingerprint impersonation.

No Upwork account/login involved - this hits the public homepage only.
"""

import os
import time

from curl_cffi import requests

TOKEN_TTL_SECONDS = 20 * 60  # cookie is valid ~25 min; refresh a bit early


class TokenFetchFailed(Exception):
    pass


class TokenManager:
    def __init__(self, proxy_url: str | None = None):
        self._proxy_url = proxy_url or os.environ.get("WEBSHARE_PROXY_URL") or None
        self._session = requests.Session()
        self._token: str | None = None
        self._fetched_at: float = 0.0

    def _proxies(self) -> dict | None:
        if not self._proxy_url:
            return None
        return {"http": self._proxy_url, "https": self._proxy_url}

    def get_token(self, force_refresh: bool = False) -> str:
        if not force_refresh and self._token and (time.time() - self._fetched_at) < TOKEN_TTL_SECONDS:
            return self._token

        resp = self._session.get(
            "https://www.upwork.com/",
            impersonate="chrome",
            proxies=self._proxies(),
            timeout=30,
        )
        token = self._session.cookies.get("visitor_gql_token")
        if not token:
            raise TokenFetchFailed(
                f"visitor_gql_token not found (status={resp.status_code}, "
                f"cookies={list(self._session.cookies.keys())})"
            )

        self._token = token
        self._fetched_at = time.time()
        return token

    @property
    def session(self):
        return self._session


if __name__ == "__main__":
    tm = TokenManager()
    print("token:", tm.get_token())
