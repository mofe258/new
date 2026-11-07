"""ESPN public scoreboard and optional football-data.org fixture ingestion."""

from __future__ import annotations

from datetime import date
from typing import Any

import requests


ESPN_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"
SUPPORTED_LEAGUES = {
    "premier-league": "eng.1",
    "la-liga": "esp.1",
    "serie-a": "ita.1",
    "bundesliga": "ger.1",
    "champions-league": "uefa.champions",
    "europa-league": "uefa.europa",
    "conference-league": "uefa.europa.conf",
}


class DataSourceError(RuntimeError):
    """Raised when a configured football data source cannot be read."""


def _score(competitor: dict[str, Any]) -> int | None:
    value = competitor.get("score")
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _espn_event_to_match(event: dict[str, Any], league: str) -> dict[str, Any]:
    competitions = event.get("competitions") or []
    competition = competitions[0] if competitions else {}
    competitors = competition.get("competitors") or []
    by_side = {item.get("homeAway"): item for item in competitors}
    home = by_side.get("home", {})
    away = by_side.get("away", {})
    status_type = (
        competition.get("status", {}).get("type")
        or event.get("status", {}).get("type")
        or {}
    )
    state = status_type.get("state")
    status = {"in": "LIVE", "post": "FINISHED"}.get(state, "SCHEDULED")
    event_id = str(event.get("id", "")).strip()
    if not event_id:
        raise DataSourceError("ESPN event did not include an event ID.")
    return {
        "match_id": f"espn:{league}:{event_id}",
        "source": "espn",
        "league": league,
        "kickoff_utc": event.get("date"),
        "status": status,
        "home_team": home.get("team", {}).get("displayName", "Unknown"),
        "away_team": away.get("team", {}).get("displayName", "Unknown"),
        "home_score": _score(home) if status == "FINISHED" else None,
        "away_score": _score(away) if status == "FINISHED" else None,
        "live_clock": (
            competition.get("status", {}).get("displayClock")
            or event.get("status", {}).get("displayClock")
        ),
    }


def fetch_espn_matches(
    leagues: list[str] | None = None,
    dates: str | None = None,
    *,
    timeout: float = 10.0,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Fetch current ESPN scoreboards; dates may be YYYYMMDD or YYYYMMDD-YYYYMMDD."""
    selected = leagues or list(SUPPORTED_LEAGUES)
    unknown = sorted(set(selected) - set(SUPPORTED_LEAGUES))
    if unknown:
        raise ValueError(f"Unsupported ESPN league key(s): {', '.join(unknown)}")
    if dates is not None and not all(char.isdigit() or char == "-" for char in dates):
        raise ValueError("dates must contain only YYYYMMDD digits and an optional dash.")

    client = session or requests
    matches = []
    for league in selected:
        params = {"limit": 500}
        if dates:
            params["dates"] = dates
        try:
            response = client.get(
                f"{ESPN_BASE_URL}/{SUPPORTED_LEAGUES[league]}/scoreboard",
                params=params,
                timeout=timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise DataSourceError(f"ESPN scoreboard request failed for {league}: {exc}") from exc
        except ValueError as exc:
            raise DataSourceError(f"ESPN returned invalid JSON for {league}.") from exc
        events = payload.get("events")
        if not isinstance(events, list):
            raise DataSourceError(f"ESPN response for {league} has no events list.")
        matches.extend(_espn_event_to_match(event, league) for event in events)
    return matches


def fetch_football_data_matches(
    competition_code: str,
    api_key: str,
    *,
    date_from: date | None = None,
    date_to: date | None = None,
    timeout: float = 10.0,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Fetch schedule/results from football-data.org v4; an API token is required."""
    if not api_key.strip():
        raise ValueError("football-data.org API key is required.")
    if not competition_code.strip() or "/" in competition_code:
        raise ValueError("A valid competition code is required.")
    params: dict[str, str] = {}
    if date_from:
        params["dateFrom"] = date_from.isoformat()
    if date_to:
        params["dateTo"] = date_to.isoformat()

    client = session or requests
    try:
        response = client.get(
            f"{FOOTBALL_DATA_BASE_URL}/competitions/{competition_code}/matches",
            headers={"X-Auth-Token": api_key},
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        raise DataSourceError(f"football-data.org request failed: {exc}") from exc
    except ValueError as exc:
        raise DataSourceError("football-data.org returned invalid JSON.") from exc
    fixtures = payload.get("matches")
    if not isinstance(fixtures, list):
        raise DataSourceError("football-data.org response has no matches list.")

    normalized = []
    for fixture in fixtures:
        status = {
            "IN_PLAY": "LIVE",
            "PAUSED": "LIVE",
            "FINISHED": "FINISHED",
            "AWARDED": "FINISHED",
        }.get(fixture.get("status"), "SCHEDULED")
        score = fixture.get("score", {}).get("fullTime", {})
        normalized.append(
            {
                "match_id": f"football-data:{competition_code}:{fixture.get('id')}",
                "source": "football-data",
                "league": competition_code,
                "kickoff_utc": fixture.get("utcDate"),
                "status": status,
                "home_team": fixture.get("homeTeam", {}).get("name", "Unknown"),
                "away_team": fixture.get("awayTeam", {}).get("name", "Unknown"),
                "home_score": score.get("home") if status == "FINISHED" else None,
                "away_score": score.get("away") if status == "FINISHED" else None,
                "live_clock": None,
            }
        )
    return normalized


def sync_matches_to_db(matches: list[dict[str, Any]], database: Any) -> int:
    """Persist fixtures and settle predictions when official full-time scores arrive."""
    synced = 0
    for match in matches:
        database.upsert_match(match)
        if (
            match.get("status") == "FINISHED"
            and match.get("home_score") is not None
            and match.get("away_score") is not None
        ):
            database.settle_match(
                match["match_id"], int(match["home_score"]), int(match["away_score"])
            )
        synced += 1
    return synced


def refresh_espn_scoreboards(
    database: Any,
    leagues: list[str] | None = None,
    dates: str | None = None,
    *,
    timeout: float = 10.0,
    session: requests.Session | None = None,
) -> int:
    """Fetch ESPN fixtures and immediately persist statuses and final scores."""
    matches = fetch_espn_matches(
        leagues, dates, timeout=timeout, session=session
    )
    return sync_matches_to_db(matches, database)
