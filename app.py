from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import streamlit as st

from db import PredictionDB
from engine import (
    HeadToHead,
    LeagueAverages,
    MatchPrediction,
    TeamStats,
    head_to_head_from_matches,
    league_averages_from_matches,
    predict_match,
    team_stats_from_matches,
)
from fetcher import DataSourceError, SUPPORTED_LEAGUES, fetch_espn_matches


st.set_page_config(
    page_title="Queen Rins | Football Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, rgba(16, 185, 129, .13), transparent 34%),
            radial-gradient(ellipse at 95% 5%, rgba(56, 189, 248, .10), transparent 30%),
            #070a10;
    }
    [data-testid="stHeader"] { background: rgba(7, 10, 16, .78); }
    [data-testid="stSidebar"] {
        background: rgba(12, 17, 26, .96);
        border-right: 1px solid rgba(255,255,255,.08);
    }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,.045);
        border: 1px solid rgba(255,255,255,.09);
        padding: 18px;
        border-radius: 16px;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(255,255,255,.10);
        border-radius: 18px;
    }
    .hero {
        padding: 1.3rem 1.5rem;
        margin: .2rem 0 1.4rem;
        border-radius: 22px;
        border: 1px solid rgba(255,255,255,.10);
        background: linear-gradient(115deg, rgba(16,185,129,.12), rgba(56,189,248,.06));
    }
    .eyebrow {
        color: #6ee7b7;
        font-size: .75rem;
        font-weight: 800;
        letter-spacing: .16em;
        text-transform: uppercase;
    }
    .muted { color: #9aa8bb; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _probability_percent(probability: float | None) -> str:
    return f"{probability:.1%}" if probability is not None else "—"


def _fixture_prediction(
    fixture: dict[str, Any], league_matches: list[dict[str, Any]]
) -> tuple[MatchPrediction, int, int]:
    home_name = fixture["home_team"]
    away_name = fixture["away_team"]
    home_stats = team_stats_from_matches(home_name, league_matches)
    away_stats = team_stats_from_matches(away_name, league_matches)
    league = league_averages_from_matches(league_matches)
    head_to_head = head_to_head_from_matches(home_name, away_name, league_matches)
    prediction = predict_match(
        home_stats, away_stats, league, head_to_head=head_to_head
    )
    return prediction, home_stats.matches_played, away_stats.matches_played


def _show_prediction(
    prediction: MatchPrediction,
    *,
    home_history: int | None = None,
    away_history: int | None = None,
) -> None:
    columns = st.columns(3)
    labels = (prediction.home_team, "Draw", prediction.away_team)
    keys = ("home", "draw", "away")
    for column, label, key in zip(columns, labels, keys):
        probability = prediction.probabilities[key]
        with column:
            st.metric(label, _probability_percent(probability))
            st.progress(max(0.0, min(1.0, probability)))
            st.caption(f"Estimated probability · {key.title()}")

    xg_home, xg_away = st.columns(2)
    xg_home.metric(f"{prediction.home_team} expected goals", f"{prediction.home_xg:.2f}")
    xg_away.metric(f"{prediction.away_team} expected goals", f"{prediction.away_xg:.2f}")

    if prediction.is_accumulator_eligible:
        st.success(
            "Eligible under the model's entropy filter. This is not a guarantee of safety."
        )
    else:
        st.warning(f"Excluded from accumulators: {prediction.trap_reason}")

    pick = prediction.recommended_pick
    if pick:
        st.info(
            f"Model market suggestion: **{pick['market']} — {pick['selection']}** "
            f"({pick['probability']:.1%} estimated probability). "
            "Odds and expected value have not been verified."
        )
    else:
        st.caption("No market passed the model's current suggestion thresholds.")

    if home_history is not None and away_history is not None:
        st.caption(
            f"Finished ESPN matches used for form: {home_history} for "
            f"{prediction.home_team}, {away_history} for {prediction.away_team}. "
            "Small samples are shrunk toward the league baseline."
        )
    st.caption(
        f"Normalized outcome entropy: {prediction.entropy_index:.3f} · "
        f"Model: {prediction.model_version}"
    )


def _load_scoreboards(leagues: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    fixtures: list[dict[str, Any]] = []
    errors: list[str] = []
    for league in leagues:
        try:
            fixtures.extend(fetch_espn_matches([league]))
        except DataSourceError as exc:
            errors.append(f"{league}: {exc}")
    return fixtures, errors


def _prepare_ledger(
    database: PredictionDB, fixtures: list[dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    for fixture in fixtures:
        try:
            related_matches = [
                match
                for match in fixtures
                if match.get("league") == fixture.get("league")
            ]
            prediction, _, _ = _fixture_prediction(fixture, related_matches)
            database.save_prediction(fixture, prediction)
            if (
                fixture.get("status") == "FINISHED"
                and fixture.get("home_score") is not None
                and fixture.get("away_score") is not None
            ):
                database.settle_match(
                    fixture["match_id"],
                    int(fixture["home_score"]),
                    int(fixture["away_score"]),
                )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            errors.append(
                f"{fixture.get('home_team', 'Home')} vs "
                f"{fixture.get('away_team', 'Away')}: {exc}"
            )
    return errors


def main() -> None:
    st.sidebar.markdown("## ⚽ Queen Rins")
    st.sidebar.caption("Football intelligence · model estimates, not guarantees")
    page = st.sidebar.radio(
        "Workspace",
        (
            "⚡ Prime Banker Intelligence",
            "🎯 Anti-Cut Accumulator",
            "📊 Model Performance",
        ),
    )
    selected_leagues = st.sidebar.multiselect(
        "ESPN competitions",
        options=list(SUPPORTED_LEAGUES),
        default=["premier-league"],
        format_func=lambda key: key.replace("-", " ").title(),
    )
    if st.sidebar.button("↻ Refresh ESPN scoreboards", use_container_width=True):
        st.session_state.pop("scoreboards", None)
        st.session_state.pop("scoreboard_errors", None)
        st.session_state.pop("loaded_leagues", None)
        st.rerun()

    database = PredictionDB()
    if (
        "scoreboards" not in st.session_state
        or st.session_state.get("loaded_leagues") != selected_leagues
    ):
        with st.spinner("Loading current ESPN scoreboards…"):
            fixtures, errors = _load_scoreboards(selected_leagues)
        st.session_state["scoreboards"] = fixtures
        st.session_state["scoreboard_errors"] = errors
        st.session_state["loaded_leagues"] = list(selected_leagues)

    fixtures = st.session_state.get("scoreboards", [])
    errors = st.session_state.get("scoreboard_errors", [])
    selected_set = set(selected_leagues)
    fixtures = [item for item in fixtures if item.get("league") in selected_set]
    league_names = {
        key: key.replace("-", " ").title() for key in SUPPORTED_LEAGUES
    }

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Queen Rins · Football intelligence</div>
          <h1 style="margin:.35rem 0">Read the match. Respect the variance.</h1>
          <div class="muted">Dixon–Coles score probabilities, trap filtering, live score ingestion, and a transparent performance ledger.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    status_columns = st.columns(3)
    status_columns[0].metric("Competitions selected", len(selected_leagues))
    status_columns[1].metric("Fixtures loaded", len(fixtures))
    status_columns[2].metric("As of (UTC)", datetime.now(timezone.utc).strftime("%H:%M"))

    for error in errors:
        st.warning(f"ESPN feed unavailable — {error}")
    if not selected_leagues:
        st.info("Select at least one competition in the sidebar.")

    ledger_errors = _prepare_ledger(database, fixtures)
    for error in ledger_errors:
        st.warning(f"Could not save this fixture: {error}")

    if page == "⚡ Prime Banker Intelligence":
        st.subheader("Prime Banker Intelligence")
        st.caption(
            "Only accumulator-eligible fixtures are listed here. Probabilities are model "
            "estimates; no market odds or expected value are supplied."
        )
        eligible_count = 0
        if not fixtures:
            st.info(
                "No fixtures were returned for the selected competitions right now. "
                "Try another competition or refresh. You can still use Manual prediction below."
            )
        else:
            for fixture in fixtures:
                try:
                    prediction, home_count, away_count = _fixture_prediction(
                        fixture, fixtures
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    st.warning(
                        f"Could not model {fixture.get('home_team', 'home')} vs "
                        f"{fixture.get('away_team', 'away')}: {exc}"
                    )
                    continue
                if not prediction.is_accumulator_eligible:
                    continue
                eligible_count += 1
                kickoff = fixture.get("kickoff_utc") or "Kickoff time unavailable"
                score = ""
                if fixture.get("status") in {"LIVE", "FINISHED"}:
                    score = (
                        f" · {fixture.get('home_score') if fixture.get('home_score') is not None else '—'}"
                        f"–{fixture.get('away_score') if fixture.get('away_score') is not None else '—'}"
                    )
                with st.container(border=True):
                    left, right = st.columns([3, 1])
                    left.markdown(
                        f"### {prediction.home_team} **vs** {prediction.away_team}"
                    )
                    left.caption(
                        f"{league_names.get(fixture.get('league'), fixture.get('league', ''))} "
                        f"· {kickoff} · {fixture.get('status', 'SCHEDULED')}{score}"
                    )
                    right.markdown("**MODEL APPROVED**")
                    _show_prediction(
                        prediction, home_history=home_count, away_history=away_count
                    )
            if eligible_count == 0:
                st.info(
                    "Fixtures were received, but all were excluded by the entropy filter. "
                    "No match is being presented as an accumulator pick."
                )

        with st.expander("Manual prediction · enter team ratings"):
            st.caption(
                "Use this when a live scoreboard is empty or the feed has no recent results. "
                "These inputs are user-provided, not verified match statistics."
            )
            with st.form("manual_prediction"):
                home_col, away_col = st.columns(2)
                with home_col:
                    manual_home = st.text_input("Home team", "Home team")
                    home_played = st.number_input("Home matches played", 0, 100, 0)
                    home_gf = st.number_input("Home goals for", 0, 500, 0)
                    home_ga = st.number_input("Home goals against", 0, 500, 0)
                    home_form_text = st.text_input("Home recent form (e.g. W,D,L)", "")
                with away_col:
                    manual_away = st.text_input("Away team", "Away team")
                    away_played = st.number_input("Away matches played", 0, 100, 0)
                    away_gf = st.number_input("Away goals for", 0, 500, 0)
                    away_ga = st.number_input("Away goals against", 0, 500, 0)
                    away_form_text = st.text_input("Away recent form (e.g. W,D,L)", "")
                h2h_col_a, h2h_col_d, h2h_col_b = st.columns(3)
                with h2h_col_a:
                    h2h_home = st.number_input("H2H home wins", 0, 100, 0)
                with h2h_col_d:
                    h2h_draws = st.number_input("H2H draws", 0, 100, 0)
                with h2h_col_b:
                    h2h_away = st.number_input("H2H away wins", 0, 100, 0)
                submit_manual = st.form_submit_button("Calculate manual prediction")
            if submit_manual:
                try:
                    home_form = tuple(
                        item.strip().upper()
                        for item in home_form_text.split(",")
                        if item.strip()
                    )
                    away_form = tuple(
                        item.strip().upper()
                        for item in away_form_text.split(",")
                        if item.strip()
                    )
                    manual_prediction = predict_match(
                        TeamStats(
                            manual_home.strip(),
                            home_played,
                            home_gf,
                            home_ga,
                            home_form,
                        ),
                        TeamStats(
                            manual_away.strip(),
                            away_played,
                            away_gf,
                            away_ga,
                            away_form,
                        ),
                        head_to_head=HeadToHead(h2h_home, h2h_draws, h2h_away),
                    )
                    st.session_state["manual_prediction"] = manual_prediction.to_dict()
                except ValueError as exc:
                    st.error(str(exc))
            saved_manual = st.session_state.get("manual_prediction")
            if saved_manual:
                _show_prediction(MatchPrediction(**saved_manual))

    elif page == "🎯 Anti-Cut Accumulator":
        st.subheader("Anti-Cut Accumulator")
        st.warning(
            "An eligible label is a model filter, not a guarantee. Combined probabilities "
            "assume independence and can materially overstate a real multi-match ticket."
        )
        if not fixtures:
            st.info("No current fixtures to build a slip from. Refresh the ESPN scoreboards.")
        else:
            eligible: list[tuple[dict[str, Any], MatchPrediction]] = []
            for fixture in fixtures:
                try:
                    prediction, _, _ = _fixture_prediction(fixture, fixtures)
                except (KeyError, TypeError, ValueError):
                    continue
                if prediction.is_accumulator_eligible and prediction.recommended_pick:
                    eligible.append((fixture, prediction))
            if not eligible:
                st.info("No fixture passed both the entropy filter and market thresholds.")
            else:
                labels = {
                    fixture["match_id"]: (
                        f"{prediction.home_team} vs {prediction.away_team} · "
                        f"{prediction.recommended_pick['market']} "
                        f"{prediction.recommended_pick['selection']}"
                    )
                    for fixture, prediction in eligible
                }
                chosen_ids = st.multiselect(
                    "Choose eligible selections",
                    options=list(labels),
                    format_func=lambda match_id: labels[match_id],
                    max_selections=15,
                )
                chosen = [
                    prediction
                    for fixture, prediction in eligible
                    if fixture["match_id"] in chosen_ids
                ]
                if chosen:
                    combined = 1.0
                    for prediction in chosen:
                        combined *= float(prediction.recommended_pick["probability"])
                    st.metric(
                        "Naive combined model probability",
                        _probability_percent(combined),
                        help="Product of individual estimates; assumes independent matches.",
                    )
                    st.caption(
                        "No odds, payout, or expected value is calculated. No SportyBet "
                        "code is generated because official event/selection identifiers and "
                        "the current order payload have not been verified."
                    )

    else:
        st.subheader("Model Performance")
        performance = database.performance()
        metric_columns = st.columns(4)
        metric_columns[0].metric(
            "Settled predictions", performance["settled_predictions"]
        )
        metric_columns[1].metric(
            "Graded pick hit rate", _probability_percent(performance["hit_rate"])
        )
        metric_columns[2].metric(
            "Multiclass Brier score",
            f"{performance['brier_score']:.3f}"
            if performance["brier_score"] is not None
            else "—",
            help="Lower is better; 0 is perfect.",
        )
        metric_columns[3].metric(
            "Won / Lost / Void",
            f"{performance['won']} / {performance['lost']} / {performance['void']}",
        )
        st.caption(
            "This ledger is stored in the app's local SQLite file. Streamlit hosting may "
            "discard local-file changes on restart; use persistent external storage for a "
            "durable production ledger."
        )
        curve = performance["calibration_curve"]
        rows = [
            {"Outcome": outcome.title(), **bucket}
            for outcome, buckets in curve.items()
            for bucket in buckets
        ]
        if rows:
            st.markdown("#### Reliability bins")
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info(
                "No settled model predictions yet. Predictions are settled when ESPN "
                "returns their final scores on a later refresh."
            )

    st.divider()
    st.caption(
        "Queen Rins predictions are statistical estimates, not betting advice. "
        "No outcome, accumulator, or profit is guaranteed."
    )


if __name__ == "__main__":
    main()
