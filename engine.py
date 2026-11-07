"""Dixon-Coles match model and probability calibration helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import exp, factorial, log
from typing import Any, Iterable, Mapping


MODEL_VERSION = "dixon-coles-1"
SHRINKAGE_MATCHES = 6.0
DC_RHO = -0.11
FORM_WEIGHT = 0.12
H2H_WEIGHT = 0.08
H2H_PRIOR_MATCHES = 6.0


@dataclass(frozen=True)
class LeagueAverages:
    home_goals: float = 1.45
    away_goals: float = 1.15

    def __post_init__(self) -> None:
        if self.home_goals <= 0 or self.away_goals <= 0:
            raise ValueError("League goal averages must be positive.")

    @property
    def team_goals(self) -> float:
        return (self.home_goals + self.away_goals) / 2


@dataclass(frozen=True)
class TeamStats:
    name: str
    matches_played: int
    goals_for: int
    goals_against: int
    recent_form: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Team name cannot be empty.")
        if min(self.matches_played, self.goals_for, self.goals_against) < 0:
            raise ValueError("Match and goal totals cannot be negative.")
        if len(self.recent_form) > 10 or any(
            result not in {"W", "D", "L"} for result in self.recent_form
        ):
            raise ValueError("Recent form must contain up to 10 W, D, or L results.")
        if self.matches_played == 0 and (self.goals_for or self.goals_against):
            raise ValueError("Goal totals require at least one played match.")


@dataclass(frozen=True)
class HeadToHead:
    home_wins: int = 0
    draws: int = 0
    away_wins: int = 0

    def __post_init__(self) -> None:
        if min(self.home_wins, self.draws, self.away_wins) < 0:
            raise ValueError("Head-to-head counts cannot be negative.")


@dataclass(frozen=True)
class MatchPrediction:
    home_team: str
    away_team: str
    home_xg: float
    away_xg: float
    probabilities: dict[str, float]
    over_2_5_probability: float
    under_3_5_probability: float
    entropy_index: float
    is_accumulator_eligible: bool
    trap_reason: str | None
    recommended_pick: dict[str, Any] | None
    model_version: str = MODEL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _shrunken_rating(
    goals: int, matches: int, league_rate: float, prior_matches: float
) -> float:
    if matches == 0:
        return 1.0
    raw_rating = (goals / matches) / league_rate
    return (raw_rating * matches + prior_matches) / (matches + prior_matches)


def _form_adjustment(team: TeamStats) -> float:
    if not team.recent_form:
        return 1.0
    points = sum({"W": 3, "D": 1, "L": 0}[result] for result in team.recent_form)
    points_rate = points / (3 * len(team.recent_form))
    sample_weight = len(team.recent_form) / (len(team.recent_form) + SHRINKAGE_MATCHES)
    return 1 + FORM_WEIGHT * (points_rate - 0.5) * sample_weight


def _poisson_probability(goals: int, expected: float) -> float:
    return exp(-expected) * expected**goals / factorial(goals)


def _goal_matrix(home_xg: float, away_xg: float, max_goals: int) -> list[list[float]]:
    matrix: list[list[float]] = []
    for home_goals in range(max_goals + 1):
        row = []
        for away_goals in range(max_goals + 1):
            probability = _poisson_probability(home_goals, home_xg) * _poisson_probability(
                away_goals, away_xg
            )
            if (home_goals, away_goals) == (0, 0):
                probability *= 1 - home_xg * away_xg * DC_RHO
            elif (home_goals, away_goals) == (0, 1):
                probability *= 1 + home_xg * DC_RHO
            elif (home_goals, away_goals) == (1, 0):
                probability *= 1 + away_xg * DC_RHO
            elif (home_goals, away_goals) == (1, 1):
                probability *= 1 - DC_RHO
            row.append(max(0.0, probability))
        matrix.append(row)

    total = sum(map(sum, matrix))
    if total <= 0:
        raise ValueError("Could not calculate a valid score probability distribution.")
    return [[probability / total for probability in row] for row in matrix]


def _probability_over(matrix: list[list[float]], goal_line: int) -> float:
    return sum(
        probability
        for home_goals, row in enumerate(matrix)
        for away_goals, probability in enumerate(row)
        if home_goals + away_goals > goal_line
    )


def _select_market(
    probabilities: Mapping[str, float],
    over_2_5: float,
    under_3_5: float,
    xg_difference: float,
) -> dict[str, Any] | None:
    home = probabilities["home"]
    draw = probabilities["draw"]
    away = probabilities["away"]
    candidates: list[dict[str, Any]] = []

    if xg_difference >= 0.60:
        if home >= 0.60:
            candidates.append(
                {"market": "1X2", "selection": "1", "probability": home, "market_id": "1"}
            )
        if away >= 0.60:
            candidates.append(
                {"market": "1X2", "selection": "2", "probability": away, "market_id": "1"}
            )

    if 1 - draw > 0:
        if home >= away and home / (1 - draw) >= 0.64:
            candidates.append(
                {"market": "DNB", "selection": "1", "probability": home, "market_id": "14"}
            )
        elif away / (1 - draw) >= 0.64:
            candidates.append(
                {"market": "DNB", "selection": "2", "probability": away, "market_id": "14"}
            )

    if home + draw >= 0.72:
        candidates.append(
            {"market": "Double Chance", "selection": "1X", "probability": home + draw, "market_id": "10"}
        )
    if away + draw >= 0.72:
        candidates.append(
            {"market": "Double Chance", "selection": "X2", "probability": away + draw, "market_id": "10"}
        )
    if over_2_5 >= 0.62:
        candidates.append(
            {"market": "Over/Under", "selection": "Over 2.5", "probability": over_2_5, "market_id": "18"}
        )
    if under_3_5 >= 0.78:
        candidates.append(
            {"market": "Over/Under", "selection": "Under 3.5", "probability": under_3_5, "market_id": "18"}
        )

    return max(candidates, key=lambda pick: pick["probability"]) if candidates else None


def predict_match(
    home: TeamStats,
    away: TeamStats,
    league: LeagueAverages | None = None,
    *,
    head_to_head: HeadToHead | None = None,
    max_goals: int = 10,
) -> MatchPrediction:
    """Estimate a match from smoothed goal rates using a Dixon-Coles score grid."""
    league = league or LeagueAverages()
    if max_goals < 5 or max_goals > 20:
        raise ValueError("max_goals must be between 5 and 20.")

    home_attack = _shrunken_rating(
        home.goals_for, home.matches_played, league.team_goals, SHRINKAGE_MATCHES
    )
    home_defense = _shrunken_rating(
        home.goals_against, home.matches_played, league.team_goals, SHRINKAGE_MATCHES
    )
    away_attack = _shrunken_rating(
        away.goals_for, away.matches_played, league.team_goals, SHRINKAGE_MATCHES
    )
    away_defense = _shrunken_rating(
        away.goals_against, away.matches_played, league.team_goals, SHRINKAGE_MATCHES
    )

    home_xg = league.home_goals * home_attack * away_defense * _form_adjustment(home)
    away_xg = league.away_goals * away_attack * home_defense * _form_adjustment(away)
    if head_to_head:
        h2h_matches = head_to_head.home_wins + head_to_head.draws + head_to_head.away_wins
        if h2h_matches:
            h2h_balance = (head_to_head.home_wins - head_to_head.away_wins) / h2h_matches
            h2h_balance *= h2h_matches / (h2h_matches + H2H_PRIOR_MATCHES)
            home_xg *= exp(H2H_WEIGHT * h2h_balance)
            away_xg *= exp(-H2H_WEIGHT * h2h_balance)
    home_xg = min(max(home_xg, 0.05), 5.0)
    away_xg = min(max(away_xg, 0.05), 5.0)

    matrix = _goal_matrix(home_xg, away_xg, max_goals)
    probabilities = {
        "home": sum(
            probability
            for home_goals, row in enumerate(matrix)
            for away_goals, probability in enumerate(row)
            if home_goals > away_goals
        ),
        "draw": sum(
            probability
            for home_goals, row in enumerate(matrix)
            for away_goals, probability in enumerate(row)
            if home_goals == away_goals
        ),
        "away": sum(
            probability
            for home_goals, row in enumerate(matrix)
            for away_goals, probability in enumerate(row)
            if home_goals < away_goals
        ),
    }
    over_2_5 = _probability_over(matrix, 2)
    under_3_5 = 1 - _probability_over(matrix, 3)
    entropy = -sum(p * log(p) for p in probabilities.values() if p > 0) / log(3)
    xg_difference = abs(home_xg - away_xg)
    is_trap = xg_difference < 0.28 and probabilities["draw"] >= 0.26

    return MatchPrediction(
        home_team=home.name,
        away_team=away.name,
        home_xg=round(home_xg, 3),
        away_xg=round(away_xg, 3),
        probabilities={outcome: round(value, 6) for outcome, value in probabilities.items()},
        over_2_5_probability=round(over_2_5, 6),
        under_3_5_probability=round(under_3_5, 6),
        entropy_index=round(entropy, 6),
        is_accumulator_eligible=not is_trap,
        trap_reason=(
            "Low xG separation and elevated draw probability."
            if is_trap
            else None
        ),
        recommended_pick=(
            None
            if is_trap
            else _select_market(probabilities, over_2_5, under_3_5, xg_difference)
        ),
    )


def team_stats_from_matches(
    team_name: str, matches: Iterable[Mapping[str, Any]], recent_limit: int = 5
) -> TeamStats:
    """Build team goals and recent W/D/L form from normalized finished fixtures."""
    if recent_limit < 0 or recent_limit > 10:
        raise ValueError("recent_limit must be between 0 and 10.")
    finished = []
    for match in matches:
        if (
            match.get("status") != "FINISHED"
            or match.get("home_score") is None
            or match.get("away_score") is None
        ):
            continue
        if team_name in (match.get("home_team"), match.get("away_team")):
            finished.append(match)
    finished.sort(key=lambda match: str(match.get("kickoff_utc", "")))

    goals_for = 0
    goals_against = 0
    form: list[str] = []
    for match in finished:
        is_home = match["home_team"] == team_name
        scored = int(match["home_score"] if is_home else match["away_score"])
        conceded = int(match["away_score"] if is_home else match["home_score"])
        goals_for += scored
        goals_against += conceded
        if scored > conceded:
            form.append("W")
        elif scored == conceded:
            form.append("D")
        else:
            form.append("L")

    return TeamStats(
        name=team_name,
        matches_played=len(finished),
        goals_for=goals_for,
        goals_against=goals_against,
        recent_form=tuple(form[-recent_limit:]) if recent_limit else (),
    )


def league_averages_from_matches(
    matches: Iterable[Mapping[str, Any]],
    fallback: LeagueAverages | None = None,
) -> LeagueAverages:
    """Estimate home/away goal baselines from finished fixtures in a league."""
    home_goals = 0
    away_goals = 0
    count = 0
    for match in matches:
        if (
            match.get("status") != "FINISHED"
            or match.get("home_score") is None
            or match.get("away_score") is None
        ):
            continue
        home_score = int(match["home_score"])
        away_score = int(match["away_score"])
        if min(home_score, away_score) < 0:
            raise ValueError("Finished match scores cannot be negative.")
        home_goals += home_score
        away_goals += away_score
        count += 1
    if not count:
        return fallback or LeagueAverages()
    return LeagueAverages(home_goals=home_goals / count, away_goals=away_goals / count)


def head_to_head_from_matches(
    home_team: str,
    away_team: str,
    matches: Iterable[Mapping[str, Any]],
    recent_limit: int = 10,
) -> HeadToHead:
    """Count recent finished meetings from the perspective of the requested home team."""
    if recent_limit < 0 or recent_limit > 50:
        raise ValueError("recent_limit must be between 0 and 50.")
    meetings = [
        match
        for match in matches
        if match.get("status") == "FINISHED"
        and match.get("home_score") is not None
        and match.get("away_score") is not None
        and {match.get("home_team"), match.get("away_team")} == {home_team, away_team}
    ]
    meetings.sort(key=lambda match: str(match.get("kickoff_utc", "")))
    if recent_limit:
        meetings = meetings[-recent_limit:]
    else:
        meetings = []
    home_wins = draws = away_wins = 0
    for match in meetings:
        home_score = int(match["home_score"])
        away_score = int(match["away_score"])
        home_score, away_score = (
            (home_score, away_score)
            if match["home_team"] == home_team
            else (away_score, home_score)
        )
        if home_score > away_score:
            home_wins += 1
        elif home_score == away_score:
            draws += 1
        else:
            away_wins += 1
    return HeadToHead(home_wins, draws, away_wins)


def calculate_brier_score(
    predictions: Iterable[Mapping[str, Any]],
) -> float | None:
    """Return multiclass 1X2 Brier score; lower is better."""
    outcomes = ("home", "draw", "away")
    scores = []
    for prediction in predictions:
        actual = prediction.get("actual_outcome")
        probabilities = prediction.get("probabilities")
        if actual not in outcomes or not isinstance(probabilities, Mapping):
            continue
        scores.append(
            sum(
                (float(probabilities.get(outcome, 0.0)) - (outcome == actual)) ** 2
                for outcome in outcomes
            )
        )
    return sum(scores) / len(scores) if scores else None


def compute_calibration_curve(
    predictions: Iterable[Mapping[str, Any]], bins: int = 10
) -> dict[str, list[dict[str, float | int]]]:
    """Return one-vs-rest reliability bins for each 1X2 outcome."""
    if bins < 1 or bins > 100:
        raise ValueError("bins must be between 1 and 100.")
    outcomes = ("home", "draw", "away")
    grouped: dict[str, list[list[tuple[float, int]]]] = {
        outcome: [[] for _ in range(bins)] for outcome in outcomes
    }
    for prediction in predictions:
        actual = prediction.get("actual_outcome")
        probabilities = prediction.get("probabilities")
        if actual not in outcomes or not isinstance(probabilities, Mapping):
            continue
        for outcome in outcomes:
            confidence = max(0.0, min(1.0, float(probabilities.get(outcome, 0.0))))
            index = min(int(confidence * bins), bins - 1)
            grouped[outcome][index].append((confidence, int(actual == outcome)))
    return {
        outcome: [
            {
                "mean_predicted": sum(p for p, _ in bucket) / len(bucket),
                "observed_frequency": sum(result for _, result in bucket) / len(bucket),
                "count": len(bucket),
            }
            for bucket in outcome_bins
            if bucket
        ]
        for outcome, outcome_bins in grouped.items()
    }


class PredictionEngine:
    """Class facade for callers that prefer an engine instance."""

    predict_match = staticmethod(predict_match)
    team_stats_from_matches = staticmethod(team_stats_from_matches)
    league_averages_from_matches = staticmethod(league_averages_from_matches)
    head_to_head_from_matches = staticmethod(head_to_head_from_matches)
    calculate_brier_score = staticmethod(calculate_brier_score)
    compute_calibration_curve = staticmethod(compute_calibration_curve)
