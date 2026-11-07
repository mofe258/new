"""Explicit, no-retry SportyBet share-order client.

The caller must supply SportyBet's actual event, market, and outcome identifiers.
This module never fabricates search results, selection IDs, or booking codes.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

import requests


SPORTYBET_SHARE_URL = "https://www.sportybet.com/api/ng/orders/share"
SPORTYBET_SEARCH_URL = "https://www.sportybet.com/api/ng/factsCenter/pcOrTabletsSearch"
MARKET_IDS = {
    "1X2": "1",
    "DOUBLE_CHANCE": "10",
    "DNB": "14",
    "OVER_UNDER": "18",
}
_CLUB_TOKENS = {
    "afc",
    "aif",
    "bc",
    "fc",
    "fk",
    "kaa",
    "sc",
    "vv",
}


def normalize_team_name(name: str) -> str:
    """Normalize common club prefixes/suffixes for exact-name comparisons."""
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_name = decomposed.encode("ascii", "ignore").decode("ascii").lower()
    tokens = re.findall(r"[a-z0-9]+", ascii_name)
    return " ".join(token for token in tokens if token not in _CLUB_TOKENS)


def market_id_for(market: str) -> str:
    key = re.sub(r"[\s/-]+", "_", market.strip().upper())
    try:
        return MARKET_IDS[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported SportyBet market: {market}") from exc


def _server_booking_code(payload: Any) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    for key in ("shareCode", "bookingCode"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    data = payload.get("data")
    if isinstance(data, Mapping):
        for key in ("shareCode", "bookingCode"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _server_booked_games(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    data = payload.get("data")
    for container in (payload, data):
        if isinstance(container, Mapping):
            games = container.get("bookedGames")
            if isinstance(games, list):
                return [game for game in games if isinstance(game, dict)]
    return []


class SportyBetClient:
    def __init__(
        self,
        *,
        timeout: float = 15.0,
        session: requests.Session | None = None,
        share_url: str = SPORTYBET_SHARE_URL,
    ):
        self.timeout = timeout
        self.session = session or requests.Session()
        self.share_url = share_url

    def search_facts(
        self, params: Mapping[str, Any], *, timeout: float | None = None
    ) -> Any:
        """Return the raw search response for caller-supplied official API parameters."""
        try:
            response = self.session.get(
                SPORTYBET_SEARCH_URL,
                params=dict(params),
                headers={"Accept": "application/json"},
                timeout=timeout or self.timeout,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise RuntimeError(f"SportyBet search request failed: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError("SportyBet search returned invalid JSON.") from exc

    def share_order(
        self,
        order_payload: Mapping[str, Any],
        booked_games: Sequence[Mapping[str, Any]] = (),
    ) -> tuple[str | None, str | None, list[dict[str, Any]]]:
        """Submit one caller-built order and return only a code from SportyBet."""
        try:
            response = self.session.post(
                self.share_url,
                json=dict(order_payload),
                headers={"Accept": "application/json"},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            return None, f"SportyBet request failed: {exc}", []

        try:
            payload = response.json()
        except ValueError:
            return None, f"SportyBet returned non-JSON response (HTTP {response.status_code}).", []

        if not response.ok:
            message = (
                payload.get("message") or payload.get("msg")
                if isinstance(payload, Mapping)
                else None
            )
            reason = str(message) if message else f"HTTP {response.status_code}"
            return None, f"SportyBet rejected the order: {reason}", []

        code = _server_booking_code(payload)
        if not code:
            message = (
                payload.get("message") or payload.get("msg")
                if isinstance(payload, Mapping)
                else None
            )
            reason = str(message) if message else (
                "successful HTTP response contained no shareCode or bookingCode"
            )
            return None, f"No booking code returned by SportyBet: {reason}.", []

        confirmed_games = _server_booked_games(payload)
        if not confirmed_games:
            confirmed_games = [dict(game) for game in booked_games]
        return code, None, confirmed_games
