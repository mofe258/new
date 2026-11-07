# Queen Rins model modules

The Streamlit `app.py` is intentionally left for you to build. These modules
provide the prediction engine, score ingestion, SportyBet client, and SQLite
ledger used by that app.

## Setup

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The SQLite database is created as `predictions.db` on first use. Set
`PREDICTIONS_DB_PATH` to choose another path.

## Module entry points

- `fetcher.fetch_espn_matches(["premier-league"])` returns normalized fixtures.
  `refresh_espn_scoreboards(db)` also stores scores and settles existing
  predictions when final scores are available. ESPN supplies scoreboards, not a
  historical team-statistics feed; the app must choose a date window and build
  team statistics from the finished fixtures it fetches.
- `engine.team_stats_from_matches(team_name, matches)` builds goals and recent
  form; `engine.league_averages_from_matches(matches)` estimates league goal
  baselines; and `engine.head_to_head_from_matches(home, away, matches)` counts
  recent meetings. Pass these to `engine.predict_match(home_stats, away_stats,
  league_averages, head_to_head=h2h)` for Dixon-Coles probabilities, xG,
  entropy, eligibility, and a threshold-based market suggestion. Form and
  head-to-head signals are shrunk toward neutral to limit small-sample effects.
  Filter the UI on
  `is_accumulator_eligible` so trap matches are not shown in accumulator
  builders; trap matches also have no suggested pick.
- `db.PredictionDB().save_prediction(match, prediction)` stores a fixture and
  its model output. Call `settle_match(match_id, home_score, away_score)` when
  an official final score is available. `performance()` reports hit rate,
  Brier score, and calibration bins.
- `sportybet.SportyBetClient.search_facts(params)` exposes the search response.
  `share_order(order_payload, booked_games)` submits one caller-built request.
  SportyBet's current request/response contract and selection identifiers must
  be verified against its live API; only a server-returned `shareCode` or
  `bookingCode` is accepted. A failed or unrecognized response returns an
  explicit error and no generated code.

The predictions are estimates from limited inputs, not guarantees, and the
market suggestions do not include odds or expected value. Do not treat
accumulator eligibility as a guarantee of safety or profitability.
