"""SQLite persistence and settlement ledger for match predictions."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

from engine import calculate_brier_score, compute_calibration_curve


DEFAULT_DB_PATH = Path(__file__).resolve().parent / "predictions.db"


class PredictionDB:
    def __init__(self, path: str | Path | None = None, timeout: float = 30.0):
        selected_path = path or os.environ.get("PREDICTIONS_DB_PATH", DEFAULT_DB_PATH)
        self._in_memory = str(selected_path) == ":memory:"
        self.path = (
            f"file:predictions-{id(self)}?mode=memory&cache=shared"
            if self._in_memory
            else Path(selected_path)
        )
        self.timeout = timeout
        self._memory_keeper = (
            sqlite3.connect(self.path, uri=True, timeout=self.timeout)
            if self._in_memory
            else None
        )
        if not self._in_memory:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.path), timeout=self.timeout, uri=self._in_memory
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def close(self) -> None:
        if self._memory_keeper is not None:
            self._memory_keeper.close()
            self._memory_keeper = None

    def initialize(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS matches (
                    match_id TEXT PRIMARY KEY,
                    league TEXT NOT NULL,
                    kickoff_utc TEXT,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'SCHEDULED',
                    live_clock TEXT,
                    home_score INTEGER,
                    away_score INTEGER,
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
                    model_version TEXT NOT NULL,
                    calculated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                    home_xg REAL NOT NULL,
                    away_xg REAL NOT NULL,
                    probabilities_json TEXT NOT NULL,
                    recommended_pick_json TEXT,
                    entropy_index REAL NOT NULL,
                    is_accumulator_eligible INTEGER NOT NULL,
                    brier_score REAL,
                    actual_outcome TEXT,
                    settlement_status TEXT NOT NULL DEFAULT 'PENDING'
                        CHECK (settlement_status IN ('BOOM','LOST','VOID','PENDING')),
                    UNIQUE(match_id, model_version)
                );
                CREATE INDEX IF NOT EXISTS idx_matches_status_kickoff
                    ON matches(status, kickoff_utc);
                CREATE INDEX IF NOT EXISTS idx_predictions_settlement
                    ON predictions(settlement_status, calculated_at);
                """
            )

    @staticmethod
    def _prediction_dict(prediction: Any) -> dict[str, Any]:
        if hasattr(prediction, "to_dict"):
            return prediction.to_dict()
        if isinstance(prediction, Mapping):
            return dict(prediction)
        raise TypeError("prediction must be a mapping or provide to_dict().")

    def upsert_match(self, match: Mapping[str, Any]) -> None:
        required = ("match_id", "home_team", "away_team")
        if any(not match.get(key) for key in required):
            raise ValueError("match_id, home_team, and away_team are required.")
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO matches (
                    match_id, league, kickoff_utc, home_team, away_team,
                    status, live_clock, home_score, away_score, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                ON CONFLICT(match_id) DO UPDATE SET
                    league=excluded.league,
                    kickoff_utc=excluded.kickoff_utc,
                    home_team=excluded.home_team,
                    away_team=excluded.away_team,
                    status=excluded.status,
                    live_clock=excluded.live_clock,
                    home_score=COALESCE(excluded.home_score, matches.home_score),
                    away_score=COALESCE(excluded.away_score, matches.away_score),
                    updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
                """,
                (
                    str(match["match_id"]),
                    str(match.get("league") or "unknown"),
                    match.get("kickoff_utc"),
                    str(match["home_team"]),
                    str(match["away_team"]),
                    str(match.get("status") or "SCHEDULED"),
                    match.get("live_clock"),
                    match.get("home_score"),
                    match.get("away_score"),
                ),
            )

    def save_prediction(self, match: Mapping[str, Any], prediction: Any) -> int:
        self.upsert_match(match)
        data = self._prediction_dict(prediction)
        probabilities = data.get("probabilities")
        if not isinstance(probabilities, Mapping) or set(probabilities) != {
            "home",
            "draw",
            "away",
        }:
            raise ValueError("prediction probabilities must contain home, draw, and away.")
        model_version = str(data.get("model_version") or "unknown")
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO predictions (
                    match_id, model_version, home_xg, away_xg, probabilities_json,
                    recommended_pick_json, entropy_index, is_accumulator_eligible
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(match_id, model_version) DO UPDATE SET
                    home_xg=excluded.home_xg,
                    away_xg=excluded.away_xg,
                    probabilities_json=excluded.probabilities_json,
                    recommended_pick_json=excluded.recommended_pick_json,
                    entropy_index=excluded.entropy_index,
                    is_accumulator_eligible=excluded.is_accumulator_eligible,
                    calculated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
                """,
                (
                    str(match["match_id"]),
                    model_version,
                    float(data["home_xg"]),
                    float(data["away_xg"]),
                    json.dumps(probabilities, separators=(",", ":"), allow_nan=False),
                    json.dumps(data["recommended_pick"], separators=(",", ":"), allow_nan=False)
                    if data.get("recommended_pick") is not None
                    else None,
                    float(data["entropy_index"]),
                    int(bool(data["is_accumulator_eligible"])),
                ),
            )
            row = connection.execute(
                "SELECT id FROM predictions WHERE match_id=? AND model_version=?",
                (str(match["match_id"]), model_version),
            ).fetchone()
            if row is None:
                raise RuntimeError("Prediction was not persisted.")
            return int(row["id"])

    def update_match_status(
        self,
        match_id: str,
        status: str,
        *,
        home_score: int | None = None,
        away_score: int | None = None,
        live_clock: str | None = None,
    ) -> None:
        if status not in {"SCHEDULED", "LIVE", "FINISHED", "POSTPONED", "CANCELLED"}:
            raise ValueError("Unsupported match status.")
        if (home_score is None) != (away_score is None):
            raise ValueError("Provide both scores or neither.")
        with self._connection() as connection:
            result = connection.execute(
                """
                UPDATE matches SET status=?, live_clock=?,
                    home_score=COALESCE(?, home_score),
                    away_score=COALESCE(?, away_score),
                    updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
                WHERE match_id=?
                """,
                (status, live_clock, home_score, away_score, match_id),
            )
            if result.rowcount == 0:
                raise KeyError(f"Unknown match_id: {match_id}")

    @staticmethod
    def _grade_pick(pick: Mapping[str, Any] | None, home: int, away: int) -> str:
        if not pick:
            return "VOID"
        selection = str(pick.get("selection", "")).upper().replace(" ", "")
        market = str(pick.get("market", "")).lower()
        result = "1" if home > away else "X" if home == away else "2"
        if market == "dnb" and result == "X":
            return "VOID"
        if selection in {"1", "HOME"}:
            won = result == "1"
        elif selection in {"2", "AWAY"}:
            won = result == "2"
        elif selection in {"X", "DRAW"}:
            won = result == "X"
        elif selection == "1X":
            won = result in {"1", "X"}
        elif selection == "X2":
            won = result in {"X", "2"}
        elif selection == "12":
            won = result in {"1", "2"}
        elif selection in {"OVER2.5", "OVER2_5"}:
            won = home + away >= 3
        elif selection in {"UNDER3.5", "UNDER3_5"}:
            won = home + away <= 3
        else:
            return "VOID"
        return "BOOM" if won else "LOST"

    def settle_match(self, match_id: str, home_score: int, away_score: int) -> int:
        if min(home_score, away_score) < 0:
            raise ValueError("Final scores cannot be negative.")
        actual = "home" if home_score > away_score else "draw" if home_score == away_score else "away"
        with self._connection() as connection:
            match_row = connection.execute(
                "SELECT match_id FROM matches WHERE match_id=?", (match_id,)
            ).fetchone()
            if match_row is None:
                raise KeyError(f"Unknown match_id: {match_id}")
            connection.execute(
                """
                UPDATE matches SET status='FINISHED', home_score=?, away_score=?,
                    live_clock=NULL, updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now')
                WHERE match_id=?
                """,
                (home_score, away_score, match_id),
            )
            predictions = connection.execute(
                """
                SELECT id, probabilities_json, recommended_pick_json
                FROM predictions WHERE match_id=?
                """,
                (match_id,),
            ).fetchall()
            for prediction in predictions:
                probabilities = json.loads(prediction["probabilities_json"])
                pick = (
                    json.loads(prediction["recommended_pick_json"])
                    if prediction["recommended_pick_json"]
                    else None
                )
                score = calculate_brier_score(
                    [{"probabilities": probabilities, "actual_outcome": actual}]
                )
                settlement = self._grade_pick(pick, home_score, away_score)
                connection.execute(
                    """
                    UPDATE predictions SET brier_score=?, actual_outcome=?,
                        settlement_status=? WHERE id=?
                    """,
                    (score, actual, settlement, prediction["id"]),
                )
            return len(predictions)

    def list_matches(self, status: str | None = None) -> list[dict[str, Any]]:
        query = """
            SELECT m.*, p.id AS prediction_id, p.model_version, p.home_xg, p.away_xg,
                   p.probabilities_json, p.recommended_pick_json, p.entropy_index,
                   p.is_accumulator_eligible, p.brier_score, p.actual_outcome,
                   p.settlement_status
            FROM matches m LEFT JOIN predictions p ON p.match_id=m.match_id
        """
        parameters: tuple[Any, ...] = ()
        if status:
            query += " WHERE m.status=?"
            parameters = (status,)
        query += " ORDER BY m.kickoff_utc DESC, m.match_id"
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            for column, key in (
                ("probabilities_json", "probabilities"),
                ("recommended_pick_json", "recommended_pick"),
            ):
                value = item.pop(column)
                item[key] = json.loads(value) if value else None
            item["is_accumulator_eligible"] = (
                bool(item["is_accumulator_eligible"])
                if item["is_accumulator_eligible"] is not None
                else None
            )
            results.append(item)
        return results

    def performance(self) -> dict[str, Any]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT probabilities_json, actual_outcome, settlement_status, brier_score
                FROM predictions WHERE actual_outcome IS NOT NULL
                ORDER BY calculated_at
                """
            ).fetchall()
        predictions = [
            {
                "probabilities": json.loads(row["probabilities_json"]),
                "actual_outcome": row["actual_outcome"],
            }
            for row in rows
        ]
        settled = [row for row in rows if row["settlement_status"] != "VOID"]
        won = sum(row["settlement_status"] == "BOOM" for row in settled)
        return {
            "settled_predictions": len(rows),
            "graded_picks": len(settled),
            "won": won,
            "lost": sum(row["settlement_status"] == "LOST" for row in settled),
            "void": sum(row["settlement_status"] == "VOID" for row in rows),
            "hit_rate": won / len(settled) if settled else None,
            "brier_score": calculate_brier_score(predictions),
            "calibration_curve": compute_calibration_curve(predictions),
        }

    calculate_brier_score = staticmethod(calculate_brier_score)
    compute_calibration_curve = staticmethod(compute_calibration_curve)
