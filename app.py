import streamlit as st
from db import PredictionDB
from engine import (
    TeamStats,
    predict_match,
    team_stats_from_matches,
    league_averages_from_matches,
    head_to_head_from_matches,
)
from fetcher import fetch_espn_matches, sync_matches_to_db