#!/usr/bin/env python3
"""
bayesian_nfl_predictor.py

Standalone Kaggle-compatible prediction script for NFL Big Data Bowl style data.

What this file does
-------------------
1. Embeds the Bayesian condition/key design used in bayesian_conditional_control_distribution.py.
2. Builds a posterior-weighted historical future-delta model from train/input_*.csv and train/output_*.csv.
3. Exposes a Kaggle-style predict(test, test_input) function.
4. By default, does NOT start the Kaggle inference server. It runs local prediction on ./test_input.csv.
5. If a truth/test file with x,y is available, it computes x/y errors.

Typical local run
-----------------
python bayesian_nfl_predictor.py \
  --train-dir ./train \
  --test-input ./test_input.csv \
  --test ./test.csv \
  --test-submit ./test_submit.csv \
  --output predictions.csv

Kaggle/server mode, only when explicitly requested
--------------------------------------------------
python bayesian_nfl_predictor.py --serve

Optional local gateway mode
---------------------------
python bayesian_nfl_predictor.py --run-local-gateway \
  --competition-data-dir /kaggle/input/nfl-big-data-bowl-2026-prediction/


  python3 bayesian_nfl_predictor.py \
  --train-dir ./train \
  --test-input ./test_input.csv \
  --test ./test.csv \
  --test-submit ./test_submit.csv \
  --output predictions.csv  
"""

from __future__ import annotations

SCRIPT_VERSION = "2026-05-11-test-submit-fixed"

import argparse
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import polars as pl  # type: ignore
except Exception:  # pragma: no cover - polars exists on Kaggle, but keep local fallback.
    pl = None

try:
    import kaggle_evaluation.nfl_inference_server as nfl_inference_server  # type: ignore
except Exception:  # pragma: no cover - not available outside Kaggle.
    nfl_inference_server = None


# =============================================================================
# Columns and configuration
# =============================================================================

ID_COLS = ["game_id", "play_id", "nfl_id"]
TRAJ_COLS = ["source_year", "source_week", "game_id", "play_id", "nfl_id"]

CONTROL_COLS = ["s", "a", "o", "dir"]
ANGLE_CONTROL_COLS = ["o", "dir"]

STATE_CATEGORICAL_COLS = ["player_position", "player_side", "player_role"]
EXTERNAL_CATEGORICAL_COLS = ["play_direction"]
STATE_NUMERIC_RAW_COLS = ["player_weight", "player_birth_date", "x", "y"]
EXTERNAL_NUMERIC_RAW_COLS = ["ball_land_x", "ball_land_y", "absolute_yardline_number"]

STATE_KEY_COLS_DEFAULT = [
    "player_position",
    "player_side",
    "player_role",
    "player_weight_bin",
    "player_age_bin",
    "x_bin",
    "y_bin",
]

EXTERNAL_KEY_COLS_DEFAULT = [
    "play_direction",
    "ball_land_x_bin",
    "ball_land_y_bin",
    "absolute_yardline_bin",
]

DEFAULT_COMPETITION_DIR = Path("/kaggle/input/nfl-big-data-bowl-2026-prediction")
DEFAULT_TRAIN_DIR = DEFAULT_COMPETITION_DIR / "train"


@dataclass
class PredictorConfig:
    # Data paths
    train_dir: str = str(DEFAULT_TRAIN_DIR if DEFAULT_TRAIN_DIR.exists() else Path("./train"))

    # Binning parameters retained from the shared scripts
    x_bin: float = 5.0
    y_bin: float = 2.0
    ball_bin: float = 5.0
    yardline_bin: float = 5.0
    age_bin: float = 3.0
    weight_bin: float = 20.0

    # Bayesian smoothing parameters
    alpha_state_prior: float = 1.0
    beta_external_global: float = 1.0
    lambda_external: float = 20.0
    tau_control_shrinkage: float = 30.0

    # Candidate filtering radii, in the same units as bin lower bounds
    candidate_x_radius: float = 10.0
    candidate_y_radius: float = 4.0
    candidate_age_radius: float = 6.0
    candidate_weight_radius: float = 40.0
    min_candidate_states: int = 20
    max_candidate_states: int = 50

    # Kernel bandwidths
    h_x: float = 10.0
    h_y: float = 4.0
    h_age: float = 6.0
    h_weight: float = 40.0
    cat_mismatch_penalty: float = 0.05

    # Prediction / fallback
    default_num_future_frames: int = 30
    dt: float = 0.1
    clip_field: bool = True
    field_x_min: float = 0.0
    field_x_max: float = 120.0
    field_y_min: float = 0.0
    field_y_max: float = 53.3

    # Debug / performance
    max_train_files: Optional[int] = None
    require_output: bool = True
    verbose: bool = True


# =============================================================================
# Basic I/O helpers, matching the user's file-reading structure
# =============================================================================

def discover_input_output_pairs(train_dir: Path) -> List[Dict[str, Any]]:
    """Match input_YYYY_wXX.csv to output_YYYY_wXX.csv."""
    pairs: List[Dict[str, Any]] = []
    pattern = re.compile(r"^input_(\d{4})_w(\d+)\.csv$")

    for input_path in sorted(train_dir.glob("input_*_w*.csv")):
        match = pattern.match(input_path.name)
        if not match:
            continue
        year = int(match.group(1))
        week_str = match.group(2)
        week = int(week_str)
        output_path = train_dir / f"output_{year}_w{week_str}.csv"
        if not output_path.exists():
            output_path = None
        pairs.append({"year": year, "week": week, "input_path": input_path, "output_path": output_path})
    return pairs


def read_csv_selected(path: Path, wanted_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Read only selected columns when requested; read all columns if wanted_cols is None."""
    if wanted_cols is None:
        return pd.read_csv(path, low_memory=False)

    header = pd.read_csv(path, nrows=0)
    available = list(header.columns)
    usecols = [c for c in wanted_cols if c in available]
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def _to_numeric_if_exists(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def add_age_and_bins(
    df: pd.DataFrame,
    source_year: Optional[int] = None,
    x_bin: float = 5.0,
    y_bin: float = 2.0,
    ball_bin: float = 5.0,
    yardline_bin: float = 5.0,
    age_bin: float = 3.0,
    weight_bin: float = 20.0,
) -> pd.DataFrame:
    """
    Convert raw numeric variables and create bin columns.

    If source_year is None, the function uses df['source_year'] if available.
    Otherwise, it falls back to 2023 for age calculation.
    """
    df = df.copy()
    numeric_cols = [
        "player_weight",
        "x",
        "y",
        "ball_land_x",
        "ball_land_y",
        "absolute_yardline_number",
        "s",
        "a",
        "o",
        "dir",
    ]
    _to_numeric_if_exists(df, numeric_cols)

    if "player_birth_date" in df.columns:
        birth = pd.to_datetime(df["player_birth_date"], errors="coerce")
        if source_year is not None:
            ref_date = pd.Timestamp(f"{int(source_year)}-09-01")
            df["player_age"] = (ref_date - birth).dt.days / 365.25
        elif "source_year" in df.columns:
            year_series = pd.to_numeric(df["source_year"], errors="coerce").fillna(2023).astype(int)
            ref_dates = pd.to_datetime(year_series.astype(str) + "-09-01", errors="coerce")
            df["player_age"] = (ref_dates - birth).dt.days / 365.25
        else:
            ref_date = pd.Timestamp("2023-09-01")
            df["player_age"] = (ref_date - birth).dt.days / 365.25
    else:
        df["player_age"] = np.nan

    def make_bin(src_col: str, dst_col: str, width: float) -> None:
        if src_col not in df.columns:
            df[dst_col] = np.nan
            return
        values = pd.to_numeric(df[src_col], errors="coerce")
        if width is None or width <= 0:
            df[dst_col] = values
        else:
            df[dst_col] = np.floor(values / width) * width

    make_bin("player_weight", "player_weight_bin", weight_bin)
    make_bin("player_age", "player_age_bin", age_bin)
    make_bin("x", "x_bin", x_bin)
    make_bin("y", "y_bin", y_bin)
    make_bin("ball_land_x", "ball_land_x_bin", ball_bin)
    make_bin("ball_land_y", "ball_land_y_bin", ball_bin)
    make_bin("absolute_yardline_number", "absolute_yardline_bin", yardline_bin)

    for col in STATE_CATEGORICAL_COLS + EXTERNAL_CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("MISSING").astype(str)

    return df


# =============================================================================
# Stable key design: no pandas ngroup()
# =============================================================================

def _is_missing_value(value: Any) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def format_key_value(value: Any) -> str:
    if _is_missing_value(value):
        return "NA"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if not np.isfinite(value):
            return "NA"
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.6g}"
    text = str(value)
    return text.replace("|", "/").replace("=", ":").replace("\n", " ").strip()


def make_key_series(df: pd.DataFrame, cols: Sequence[str], key_name: str) -> pd.Series:
    existing_cols = [c for c in cols if c in df.columns]
    if len(existing_cols) == 0:
        return pd.Series([f"{key_name}=EMPTY"] * len(df), index=df.index)

    def row_to_key(row: pd.Series) -> str:
        return "|".join(f"{c}={format_key_value(row[c])}" for c in existing_cols)

    return df[existing_cols].apply(row_to_key, axis=1)


def stable_hash_id(text: str, prefix: str = "cid") -> str:
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=8).hexdigest()
    return f"{prefix}_{digest}"


def add_stable_condition_keys(
    df: pd.DataFrame,
    state_key_cols: Sequence[str],
    external_key_cols: Sequence[str],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = df.copy()
    state_cols = [c for c in state_key_cols if c in df.columns]
    external_cols = [c for c in external_key_cols if c in df.columns]
    df["state_key"] = make_key_series(df, state_cols, "state")
    df["external_key"] = make_key_series(df, external_cols, "external")
    df["condition_key"] = df["state_key"] + " || " + df["external_key"]
    df["state_id"] = df["state_key"].map(lambda x: stable_hash_id(x, "state"))
    df["external_id"] = df["external_key"].map(lambda x: stable_hash_id(x, "ext"))
    df["condition_id"] = df["condition_key"].map(lambda x: stable_hash_id(x, "cond"))
    return df, state_cols, external_cols


def make_trajectory_id(df: pd.DataFrame) -> pd.Series:
    required = ["source_year", "source_week", "game_id", "play_id", "nfl_id"]
    for col in required:
        if col not in df.columns:
            df[col] = "NA"
    return (
        df["source_year"].astype(str)
        + "_w"
        + df["source_week"].astype(str)
        + "_g"
        + df["game_id"].astype(str)
        + "_p"
        + df["play_id"].astype(str)
        + "_n"
        + df["nfl_id"].astype(str)
    )


# =============================================================================
# Bayesian trajectory model
# =============================================================================

def build_state_prior(df_unit: pd.DataFrame, state_key_cols: Sequence[str], alpha: float) -> pd.DataFrame:
    state_values = (
        df_unit.groupby("state_key", dropna=False, sort=False)[list(state_key_cols) + ["state_id"]]
        .first()
        .reset_index()
    )
    state_counts = (
        df_unit.groupby("state_key", dropna=False, sort=False)
        .agg(
            n_state=("state_key", "size"),
            n_state_trajectories=("_trajectory_id", "nunique"),
            n_state_games=("game_id", "nunique"),
            n_state_plays=("play_id", "nunique"),
            n_state_players=("nfl_id", "nunique"),
        )
        .reset_index()
    )
    prior = state_values.merge(state_counts, on="state_key", how="left")
    n_total = len(df_unit)
    n_states = max(1, len(prior))
    prior["prior_prob"] = (prior["n_state"] + alpha) / (n_total + alpha * n_states)
    return prior


def build_external_registry(df_unit: pd.DataFrame, external_key_cols: Sequence[str], beta: float) -> pd.DataFrame:
    external_values = (
        df_unit.groupby("external_key", dropna=False, sort=False)[list(external_key_cols) + ["external_id"]]
        .first()
        .reset_index()
    )
    external_counts = df_unit.groupby("external_key", dropna=False, sort=False).agg(
        n_external=("external_key", "size")
    ).reset_index()
    registry = external_values.merge(external_counts, on="external_key", how="left")
    n_total = len(df_unit)
    n_external = max(1, len(registry))
    registry["global_external_prob"] = (registry["n_external"] + beta) / (n_total + beta * n_external)
    return registry


def build_condition_registry(
    df_unit: pd.DataFrame,
    state_key_cols: Sequence[str],
    external_key_cols: Sequence[str],
) -> pd.DataFrame:
    value_cols = [
        "condition_id",
        "state_id",
        "external_id",
        "state_key",
        "external_key",
        *state_key_cols,
        *external_key_cols,
    ]
    value_cols = [c for c in value_cols if c in df_unit.columns]
    values = df_unit.groupby("condition_key", dropna=False, sort=False)[value_cols].first().reset_index()
    counts = (
        df_unit.groupby("condition_key", dropna=False, sort=False)
        .agg(
            n_condition=("condition_key", "size"),
            n_condition_trajectories=("_trajectory_id", "nunique"),
        )
        .reset_index()
    )
    return values.merge(counts, on="condition_key", how="left")


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def _weighted_mean(values: Sequence[float], weights: Sequence[float], default: float = np.nan) -> float:
    vals = np.asarray(values, dtype=float)
    ws = np.asarray(weights, dtype=float)
    mask = np.isfinite(vals) & np.isfinite(ws) & (ws > 0)
    if not mask.any():
        return default
    return float(np.average(vals[mask], weights=ws[mask]))


def _field_clip(x: float, y: float, cfg: PredictorConfig) -> Tuple[float, float]:
    if not cfg.clip_field:
        return x, y
    if np.isfinite(x):
        x = float(np.clip(x, cfg.field_x_min, cfg.field_x_max))
    if np.isfinite(y):
        y = float(np.clip(y, cfg.field_y_min, cfg.field_y_max))
    return x, y


def kinematic_fallback(last_row: pd.Series, future_step: int, cfg: PredictorConfig) -> Tuple[float, float]:
    """
    Kinematic fallback using current x,y,s,a,dir.

    NFL tracking convention is commonly 0 degrees = positive y-axis and 90 degrees = positive x-axis.
    Therefore dx = distance * sin(dir), dy = distance * cos(dir).
    """
    x0 = _safe_float(last_row.get("x"), 0.0)
    y0 = _safe_float(last_row.get("y"), 0.0)
    speed = max(0.0, _safe_float(last_row.get("s"), 0.0))
    accel = _safe_float(last_row.get("a"), 0.0)
    direction = _safe_float(last_row.get("dir"), np.nan)
    if not np.isfinite(direction):
        direction = _safe_float(last_row.get("o"), 0.0)

    t = max(1, int(future_step)) * cfg.dt
    distance = speed * t + 0.5 * accel * t * t
    if not np.isfinite(distance):
        distance = 0.0
    distance = max(0.0, distance)

    rad = math.radians(direction % 360)
    pred_x = x0 + distance * math.sin(rad)
    pred_y = y0 + distance * math.cos(rad)
    return _field_clip(pred_x, pred_y, cfg)


class BayesianTrajectoryPredictor:
    """Posterior-weighted trajectory predictor using embedded Bayesian condition distribution logic."""

    def __init__(self, config: PredictorConfig):
        self.cfg = config
        self.is_fitted = False
        self.state_key_cols: List[str] = []
        self.external_key_cols: List[str] = []
        self.state_prior: pd.DataFrame = pd.DataFrame()
        self.external_registry: pd.DataFrame = pd.DataFrame()
        self.condition_registry: pd.DataFrame = pd.DataFrame()
        self.default_external_prob: float = 1.0
        self.state_count: Dict[str, int] = {}
        self.state_prior_prob: Dict[str, float] = {}
        self.external_p0: Dict[str, float] = {}
        self.joint_count: Dict[Tuple[str, str], int] = {}
        self.condition_key_lookup: Dict[Tuple[str, str], str] = {}
        self.condition_id_lookup: Dict[Tuple[str, str], str] = {}
        self.future_by_condition_step: Dict[Tuple[str, int], Tuple[float, float, int]] = {}
        self.future_by_state_step: Dict[Tuple[str, int], Tuple[float, float, int]] = {}
        self.global_future_by_step: Dict[int, Tuple[float, float, int]] = {}
        self.max_future_step: int = 0

    def log(self, message: str) -> None:
        if self.cfg.verbose:
            print(message, file=sys.stderr)

    def fit(self) -> "BayesianTrajectoryPredictor":
        train_dir = Path(self.cfg.train_dir)
        pairs = discover_input_output_pairs(train_dir)
        if self.cfg.max_train_files is not None:
            pairs = pairs[: self.cfg.max_train_files]
        if not pairs:
            raise RuntimeError(f"No input_YYYY_wXX.csv files found under {train_dir}")

        self.log(f"[fit] matched train files: {len(pairs)}")

        input_wanted_cols = list(
            set(
                ID_COLS
                + ["frame_id"]
                + STATE_CATEGORICAL_COLS
                + EXTERNAL_CATEGORICAL_COLS
                + STATE_NUMERIC_RAW_COLS
                + EXTERNAL_NUMERIC_RAW_COLS
                + CONTROL_COLS
            )
        )
        output_wanted_cols = ID_COLS + ["frame_id", "x", "y"]

        all_last_rows: List[pd.DataFrame] = []
        all_future_rows: List[pd.DataFrame] = []

        for idx, pair in enumerate(pairs, start=1):
            input_path: Path = pair["input_path"]
            output_path: Optional[Path] = pair["output_path"]
            year = int(pair["year"])
            week = int(pair["week"])
            if output_path is None and self.cfg.require_output:
                continue

            self.log(f"[fit] reading {idx}/{len(pairs)}: {input_path.name}")
            inp = read_csv_selected(input_path, input_wanted_cols)
            inp["source_year"] = year
            inp["source_week"] = week
            _to_numeric_if_exists(inp, ID_COLS + ["frame_id"])
            inp = inp.dropna(subset=ID_COLS + ["frame_id"])
            if inp.empty:
                continue

            inp = add_age_and_bins(
                inp,
                source_year=year,
                x_bin=self.cfg.x_bin,
                y_bin=self.cfg.y_bin,
                ball_bin=self.cfg.ball_bin,
                yardline_bin=self.cfg.yardline_bin,
                age_bin=self.cfg.age_bin,
                weight_bin=self.cfg.weight_bin,
            )
            inp["_trajectory_id"] = make_trajectory_id(inp)
            state_cols = [c for c in STATE_KEY_COLS_DEFAULT if c in inp.columns]
            external_cols = [c for c in EXTERNAL_KEY_COLS_DEFAULT if c in inp.columns]
            inp, state_cols, external_cols = add_stable_condition_keys(inp, state_cols, external_cols)
            if not self.state_key_cols:
                self.state_key_cols = state_cols
            if not self.external_key_cols:
                self.external_key_cols = external_cols

            last = inp.sort_values(TRAJ_COLS + ["frame_id"]).groupby(TRAJ_COLS, dropna=False, sort=False).tail(1).copy()
            all_last_rows.append(last)

            if output_path is None or not output_path.exists():
                continue
            out = read_csv_selected(output_path, output_wanted_cols)
            _to_numeric_if_exists(out, ID_COLS + ["frame_id", "x", "y"])
            out = out.dropna(subset=ID_COLS)
            if out.empty:
                continue
            out["source_year"] = year
            out["source_week"] = week
            out = out.sort_values(TRAJ_COLS + ["frame_id"])
            out["future_step"] = out.groupby(TRAJ_COLS, dropna=False, sort=False).cumcount() + 1

            last_merge_cols = TRAJ_COLS + [
                "x",
                "y",
                "state_key",
                "external_key",
                "condition_key",
                "condition_id",
                "state_id",
                "external_id",
            ]
            last_small = last[last_merge_cols].rename(columns={"x": "last_x", "y": "last_y"})
            fut = out.merge(last_small, on=TRAJ_COLS, how="inner")
            if fut.empty:
                continue
            fut["delta_x"] = fut["x"] - fut["last_x"]
            fut["delta_y"] = fut["y"] - fut["last_y"]
            keep_cols = [
                "state_key",
                "external_key",
                "condition_key",
                "condition_id",
                "future_step",
                "delta_x",
                "delta_y",
            ]
            all_future_rows.append(fut[keep_cols])

        if not all_last_rows:
            raise RuntimeError("No usable input rows were found for fitting.")
        df_unit = pd.concat(all_last_rows, ignore_index=True)
        if self.cfg.require_output and not all_future_rows:
            raise RuntimeError("No usable output rows were found for fitting future deltas.")

        # Build Bayesian count tables.
        self.state_prior = build_state_prior(df_unit, self.state_key_cols, self.cfg.alpha_state_prior)
        self.external_registry = build_external_registry(df_unit, self.external_key_cols, self.cfg.beta_external_global)
        self.condition_registry = build_condition_registry(df_unit, self.state_key_cols, self.external_key_cols)
        self.condition_registry = self.condition_registry.merge(
            self.state_prior[["state_key", "n_state", "prior_prob"]], on="state_key", how="left"
        )
        self.default_external_prob = 1.0 / max(1, len(self.external_registry))

        self.state_count = dict(zip(self.state_prior["state_key"], self.state_prior["n_state"].astype(int)))
        self.state_prior_prob = dict(zip(self.state_prior["state_key"], self.state_prior["prior_prob"].astype(float)))
        self.external_p0 = dict(zip(self.external_registry["external_key"], self.external_registry["global_external_prob"].astype(float)))
        for _, row in self.condition_registry.iterrows():
            key = (row["state_key"], row["external_key"])
            self.joint_count[key] = int(row["n_condition"])
            self.condition_key_lookup[key] = row["condition_key"]
            self.condition_id_lookup[key] = row["condition_id"]

        if all_future_rows:
            future_df = pd.concat(all_future_rows, ignore_index=True)
            self.max_future_step = int(future_df["future_step"].max()) if len(future_df) else 0
            self.future_by_condition_step = self._build_future_lookup(future_df, ["condition_key", "future_step"])
            self.future_by_state_step = self._build_future_lookup(future_df, ["state_key", "future_step"])
            self.global_future_by_step = self._build_global_future_lookup(future_df)
        else:
            self.max_future_step = 0
            self.future_by_condition_step = {}
            self.future_by_state_step = {}
            self.global_future_by_step = {}

        self.is_fitted = True
        self.log(
            f"[fit] done: states={len(self.state_prior):,}, "
            f"conditions={len(self.condition_registry):,}, max_future_step={self.max_future_step}"
        )
        return self

    @staticmethod
    def _build_future_lookup(future_df: pd.DataFrame, group_cols: Sequence[str]) -> Dict[Tuple[str, int], Tuple[float, float, int]]:
        grouped = (
            future_df.groupby(list(group_cols), dropna=False, sort=False)
            .agg(delta_x_mean=("delta_x", "mean"), delta_y_mean=("delta_y", "mean"), n=("delta_x", "size"))
            .reset_index()
        )
        lookup: Dict[Tuple[str, int], Tuple[float, float, int]] = {}
        key_col = group_cols[0]
        for _, row in grouped.iterrows():
            lookup[(row[key_col], int(row["future_step"]))] = (
                float(row["delta_x_mean"]),
                float(row["delta_y_mean"]),
                int(row["n"]),
            )
        return lookup

    @staticmethod
    def _build_global_future_lookup(future_df: pd.DataFrame) -> Dict[int, Tuple[float, float, int]]:
        grouped = (
            future_df.groupby("future_step", dropna=False, sort=False)
            .agg(delta_x_mean=("delta_x", "mean"), delta_y_mean=("delta_y", "mean"), n=("delta_x", "size"))
            .reset_index()
        )
        return {
            int(row["future_step"]): (float(row["delta_x_mean"]), float(row["delta_y_mean"]), int(row["n"]))
            for _, row in grouped.iterrows()
        }

    def _external_likelihood(self, state_key: str, external_key: str) -> float:
        n_s = float(self.state_count.get(state_key, 0))
        n_se = float(self.joint_count.get((state_key, external_key), 0))
        p0 = float(self.external_p0.get(external_key, self.default_external_prob))
        if n_s <= 0:
            return p0
        return (n_se + self.cfg.lambda_external * p0) / (n_s + self.cfg.lambda_external)

    def _filter_candidate_states(self, obs_row: pd.Series) -> pd.DataFrame:
        cfg = self.cfg
        levels = [
            ("position_side_role_xy_age_weight", ["player_position", "player_side", "player_role"], True, True),
            ("position_side_role_xy", ["player_position", "player_side", "player_role"], True, False),
            ("side_role_xy", ["player_side", "player_role"], True, False),
            ("side_role", ["player_side", "player_role"], False, False),
            ("global", [], False, False),
        ]

        for level_name, cat_cols, use_xy, use_age_weight in levels:
            cand = self.state_prior.copy()
            for col in cat_cols:
                if col in cand.columns and col in obs_row.index and not pd.isna(obs_row[col]):
                    cand = cand[cand[col].astype(str) == str(obs_row[col])]
            if use_xy:
                if "x_bin" in cand.columns and "x_bin" in obs_row.index and not pd.isna(obs_row["x_bin"]):
                    cand = cand[(cand["x_bin"] - obs_row["x_bin"]).abs() <= cfg.candidate_x_radius]
                if "y_bin" in cand.columns and "y_bin" in obs_row.index and not pd.isna(obs_row["y_bin"]):
                    cand = cand[(cand["y_bin"] - obs_row["y_bin"]).abs() <= cfg.candidate_y_radius]
            if use_age_weight:
                if "player_age_bin" in cand.columns and "player_age_bin" in obs_row.index and not pd.isna(obs_row["player_age_bin"]):
                    cand = cand[(cand["player_age_bin"] - obs_row["player_age_bin"]).abs() <= cfg.candidate_age_radius]
                if "player_weight_bin" in cand.columns and "player_weight_bin" in obs_row.index and not pd.isna(obs_row["player_weight_bin"]):
                    cand = cand[(cand["player_weight_bin"] - obs_row["player_weight_bin"]).abs() <= cfg.candidate_weight_radius]
            if len(cand) >= cfg.min_candidate_states or level_name == "global":
                cand = cand.copy()
                cand["candidate_level"] = level_name
                return cand
        cand = self.state_prior.copy()
        cand["candidate_level"] = "global"
        return cand

    def _state_similarity(self, obs_row: pd.Series, cand: pd.DataFrame) -> np.ndarray:
        cfg = self.cfg
        sim = np.ones(len(cand), dtype=float)
        bandwidths = {
            "x_bin": cfg.h_x,
            "y_bin": cfg.h_y,
            "player_age_bin": cfg.h_age,
            "player_weight_bin": cfg.h_weight,
        }
        for col, h in bandwidths.items():
            if h <= 0 or col not in cand.columns or col not in obs_row.index:
                continue
            obs_val = _safe_float(obs_row[col])
            if not np.isfinite(obs_val):
                continue
            vals = pd.to_numeric(cand[col], errors="coerce").astype(float).values
            diff = vals - obs_val
            valid = np.isfinite(diff)
            factor = np.ones(len(cand), dtype=float)
            factor[valid] = np.exp(-0.5 * (diff[valid] / h) ** 2)
            sim *= factor

        for col in ["player_position", "player_side", "player_role"]:
            if col not in cand.columns or col not in obs_row.index or pd.isna(obs_row[col]):
                continue
            matches = cand[col].astype(str).values == str(obs_row[col])
            sim *= np.where(matches, 1.0, cfg.cat_mismatch_penalty)
        return sim

    def posterior_states(self, obs_row: pd.Series) -> pd.DataFrame:
        external_key = str(obs_row["external_key"])
        cand = self._filter_candidate_states(obs_row).copy()
        sim = self._state_similarity(obs_row, cand)
        prior = cand["prior_prob"].astype(float).values
        likelihood = np.array([self._external_likelihood(str(s), external_key) for s in cand["state_key"].values], dtype=float)
        raw = sim * prior * likelihood
        total = float(np.nansum(raw))
        if not np.isfinite(total) or total <= 0:
            raw = prior.copy()
            total = float(np.nansum(raw))
        if total <= 0:
            cand["posterior_weight"] = 1.0 / max(1, len(cand))
        else:
            cand["posterior_weight"] = raw / total
        cand["state_similarity"] = sim
        cand["likelihood_external_given_state"] = likelihood
        cand = cand.sort_values("posterior_weight", ascending=False)
        if self.cfg.max_candidate_states and self.cfg.max_candidate_states > 0:
            cand = cand.head(self.cfg.max_candidate_states).copy()
            s = cand["posterior_weight"].sum()
            if s > 0:
                cand["posterior_weight"] = cand["posterior_weight"] / s
        return cand

    def _mixture_delta_for_step(self, obs_row: pd.Series, future_step: int) -> Tuple[float, float, int]:
        """Return posterior-weighted mean delta_x, delta_y for one future step."""
        posterior = self.posterior_states(obs_row)
        external_key = str(obs_row["external_key"])

        dx_values: List[float] = []
        dy_values: List[float] = []
        weights: List[float] = []

        for _, prow in posterior.iterrows():
            state_key = str(prow["state_key"])
            pw = float(prow["posterior_weight"])
            n_se = int(self.joint_count.get((state_key, external_key), 0))
            rho = n_se / (n_se + self.cfg.tau_control_shrinkage) if n_se > 0 else 0.0
            condition_key = self.condition_key_lookup.get((state_key, external_key), state_key + " || " + external_key)

            exact = self.future_by_condition_step.get((condition_key, int(future_step)))
            if exact is not None and rho > 0:
                dx, dy, n = exact
                # n gives a mild reliability scaling but does not overwhelm posterior weights.
                reliability = math.sqrt(max(1, n))
                dx_values.append(dx)
                dy_values.append(dy)
                weights.append(pw * rho * reliability)

            state_only = self.future_by_state_step.get((state_key, int(future_step)))
            if state_only is not None:
                dx, dy, n = state_only
                reliability = math.sqrt(max(1, n))
                dx_values.append(dx)
                dy_values.append(dy)
                weights.append(pw * (1.0 - rho) * reliability)

        if len(weights) == 0:
            # Last-resort global historical future-delta before kinematic fallback.
            global_delta = self.global_future_by_step.get(int(future_step))
            if global_delta is not None:
                dx, dy, n = global_delta
                return dx, dy, n
            return np.nan, np.nan, 0

        dx_mean = _weighted_mean(dx_values, weights)
        dy_mean = _weighted_mean(dy_values, weights)
        effective_n = int(np.sum(np.asarray(weights) > 0))
        return dx_mean, dy_mean, effective_n

    def preprocess_test_input(self, test_input: pd.DataFrame) -> pd.DataFrame:
        df = test_input.copy()
        _to_numeric_if_exists(df, ID_COLS + ["frame_id"])
        if "source_year" not in df.columns:
            if "game_id" in df.columns:
                df["source_year"] = df["game_id"].astype(str).str[:4].replace("nan", "2023")
                df["source_year"] = pd.to_numeric(df["source_year"], errors="coerce").fillna(2023).astype(int)
            else:
                df["source_year"] = 2023
        if "source_week" not in df.columns:
            df["source_week"] = 0
        df = add_age_and_bins(
            df,
            source_year=None,
            x_bin=self.cfg.x_bin,
            y_bin=self.cfg.y_bin,
            ball_bin=self.cfg.ball_bin,
            yardline_bin=self.cfg.yardline_bin,
            age_bin=self.cfg.age_bin,
            weight_bin=self.cfg.weight_bin,
        )
        df["_trajectory_id"] = make_trajectory_id(df)
        df, _, _ = add_stable_condition_keys(df, self.state_key_cols, self.external_key_cols)
        return df

    def predict_pandas(self, test: pd.DataFrame, test_input: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            self.fit()
        if test is None or len(test) == 0:
            return pd.DataFrame({"x": [], "y": []})

        test_df = test.copy().reset_index(drop=True)
        input_df = self.preprocess_test_input(test_input)
        last_rows = (
            input_df.sort_values(ID_COLS + ["frame_id"])
            .groupby(ID_COLS, dropna=False, sort=False)
            .tail(1)
            .copy()
        )
        last_lookup = {tuple(row[c] for c in ID_COLS): row for _, row in last_rows.iterrows()}

        # Predict in original test row order. Future step is the within-trajectory row rank in test.
        tmp = test_df.copy()
        _to_numeric_if_exists(tmp, ID_COLS + ["frame_id"])
        tmp["_original_order"] = np.arange(len(tmp))
        if "frame_id" in tmp.columns:
            tmp = tmp.sort_values(ID_COLS + ["frame_id", "_original_order"])
        else:
            tmp = tmp.sort_values(ID_COLS + ["_original_order"])
        tmp["_future_step"] = tmp.groupby(ID_COLS, dropna=False, sort=False).cumcount() + 1

        pred_x = np.zeros(len(tmp), dtype=float)
        pred_y = np.zeros(len(tmp), dtype=float)

        for j, (_, row) in enumerate(tmp.iterrows()):
            key = tuple(row[c] for c in ID_COLS)
            last = last_lookup.get(key)
            if last is None:
                # If the player is missing from test_input, use zeros rather than failing the server.
                pred_x[j] = 0.0
                pred_y[j] = 0.0
                continue
            future_step = int(row["_future_step"])
            dx, dy, n_components = self._mixture_delta_for_step(last, future_step)
            if np.isfinite(dx) and np.isfinite(dy) and n_components > 0:
                x0 = _safe_float(last.get("x"), 0.0)
                y0 = _safe_float(last.get("y"), 0.0)
                x_hat, y_hat = _field_clip(x0 + dx, y0 + dy, self.cfg)
            else:
                x_hat, y_hat = kinematic_fallback(last, future_step, self.cfg)
            pred_x[j] = x_hat
            pred_y[j] = y_hat

        tmp["x"] = pred_x
        tmp["y"] = pred_y
        out = tmp.sort_values("_original_order")[["x", "y"]].reset_index(drop=True)
        return out


# =============================================================================
# Global Kaggle predict function
# =============================================================================

_GLOBAL_PREDICTOR: Optional[BayesianTrajectoryPredictor] = None
_GLOBAL_CONFIG = PredictorConfig()


def get_global_predictor() -> BayesianTrajectoryPredictor:
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None:
        _GLOBAL_PREDICTOR = BayesianTrajectoryPredictor(_GLOBAL_CONFIG).fit()
    return _GLOBAL_PREDICTOR


def _to_pandas(df: Any) -> pd.DataFrame:
    if pl is not None and isinstance(df, pl.DataFrame):
        return df.to_pandas()
    if isinstance(df, pd.DataFrame):
        return df.copy()
    return pd.DataFrame(df)


def predict(test: Any, test_input: Any) -> Any:
    """
    Kaggle-compatible predict function.

    Parameters
    ----------
    test:
        Rows that require x,y predictions. The returned dataframe must have the same length.
    test_input:
        Observed historical tracking data for the current batch.

    Returns
    -------
    Polars DataFrame if polars is available, otherwise pandas DataFrame, with columns ['x', 'y'].
    """
    predictor = get_global_predictor()
    test_pd = _to_pandas(test)
    input_pd = _to_pandas(test_input)
    pred_pd = predictor.predict_pandas(test_pd, input_pd)
    assert len(pred_pd) == len(test_pd)
    if pl is not None:
        return pl.from_pandas(pred_pd[["x", "y"]])
    return pred_pd[["x", "y"]]


# =============================================================================
# Local evaluation helpers
# =============================================================================

def make_prediction_template_from_test_input(test_input: pd.DataFrame, default_num_future_frames: int) -> pd.DataFrame:
    """Create a local prediction template when no Kaggle 'test' file is available."""
    df = test_input.copy()
    _to_numeric_if_exists(df, ID_COLS + ["frame_id"])
    if "player_to_predict" in df.columns:
        mask = df["player_to_predict"].astype(str).str.lower().isin(["true", "1", "yes"])
        targets = df[mask].copy()
        if targets.empty:
            targets = df.copy()
    else:
        targets = df.copy()

    last = targets.sort_values(ID_COLS + ["frame_id"]).groupby(ID_COLS, dropna=False, sort=False).tail(1)
    records: List[Dict[str, Any]] = []
    for _, row in last.iterrows():
        n_future = default_num_future_frames
        for candidate_col in ["num_frames_output", "output_num_frames", "n_future", "num_future_frames"]:
            if candidate_col in row.index and pd.notna(row[candidate_col]):
                n_future = max(1, _safe_int(row[candidate_col], default_num_future_frames))
                break
        for step in range(1, n_future + 1):
            rec = {c: row[c] for c in ID_COLS}
            rec["frame_id"] = step
            records.append(rec)
    return pd.DataFrame(records)


def _extract_truth_from_xy_file(xy_df: pd.DataFrame, template: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Build a truth dataframe with ID_COLS + frame_id + x,y from a file that contains x,y.

    This supports the common local-evaluation case where:
      - test.csv has the prediction template but no x,y, and
      - test_submit.csv has the true x,y.

    Matching priority:
      1. Direct keys: game_id, play_id, nfl_id, frame_id
      2. Shared row identifier: id / row_id / sample_id
      3. Row order, only when lengths match
    """
    if not {"x", "y"}.issubset(xy_df.columns):
        return None

    key_cols = ID_COLS + ["frame_id"]

    # Case 1: truth file already has the full football keys.
    if set(key_cols).issubset(xy_df.columns):
        truth = xy_df[key_cols + ["x", "y"]].copy()
        _to_numeric_if_exists(truth, key_cols + ["x", "y"])
        return truth

    # The template must provide the football keys for cases 2 and 3.
    if not set(key_cols).issubset(template.columns):
        return None

    # Case 2: match by a shared row id if both files have one.
    for row_id_col in ["id", "row_id", "sample_id"]:
        if row_id_col in template.columns and row_id_col in xy_df.columns:
            left = template[key_cols + [row_id_col]].copy()
            right = xy_df[[row_id_col, "x", "y"]].copy()
            merged = left.merge(right, on=row_id_col, how="left")
            if merged[["x", "y"]].notna().any().any():
                truth = merged[key_cols + ["x", "y"]].copy()
                _to_numeric_if_exists(truth, key_cols + ["x", "y"])
                return truth

    # Case 3: match by row order when both files have exactly the same length.
    if len(xy_df) == len(template):
        truth = template[key_cols].copy().reset_index(drop=True)
        truth["x"] = pd.to_numeric(xy_df["x"].reset_index(drop=True), errors="coerce")
        truth["y"] = pd.to_numeric(xy_df["y"].reset_index(drop=True), errors="coerce")
        _to_numeric_if_exists(truth, key_cols + ["x", "y"])
        return truth

    return None


def read_truth_if_available(
    path: Optional[str],
    template: pd.DataFrame,
    test_submit_path: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    candidates: List[Path] = []

    # Highest priority: the explicitly supplied test_submit.csv because test.csv may not contain x,y.
    if test_submit_path:
        candidates.append(Path(test_submit_path))
    if path:
        candidates.append(Path(path))

    for default_name in ["test_submit.csv", "solution.csv", "ground_truth.csv", "truth.csv", "test.csv"]:
        candidates.append(Path(default_name))

    seen: set[str] = set()
    for p in candidates:
        p_key = str(p.resolve()) if p.exists() else str(p)
        if p_key in seen:
            continue
        seen.add(p_key)
        if not p.exists():
            continue
        df = pd.read_csv(p, low_memory=False)
        truth = _extract_truth_from_xy_file(df, template)
        if truth is not None:
            return truth

    # If the template itself has x,y, treat those as truth before dropping for prediction.
    truth = _extract_truth_from_xy_file(template, template)
    if truth is not None:
        return truth
    return None


def evaluate_predictions(
    prediction_with_ids: pd.DataFrame,
    truth: pd.DataFrame,
    detail_path: Path,
    summary_path: Path,
) -> pd.DataFrame:
    pred = prediction_with_ids.rename(columns={"x": "x_pred", "y": "y_pred"})
    true = truth.rename(columns={"x": "x_true", "y": "y_true"})
    for df in [pred, true]:
        _to_numeric_if_exists(df, ID_COLS + ["frame_id"])
    merged = pred.merge(true, on=ID_COLS + ["frame_id"], how="inner")
    if merged.empty:
        summary = pd.DataFrame([
            {
                "n": 0,
                "rmse_x": np.nan,
                "rmse_y": np.nan,
                "rmse_xy": np.nan,
                "mae_x": np.nan,
                "mae_y": np.nan,
                "mae_xy": np.nan,
            }
        ])
    else:
        merged["err_x"] = merged["x_pred"] - merged["x_true"]
        merged["err_y"] = merged["y_pred"] - merged["y_true"]
        merged["err_xy"] = np.sqrt(merged["err_x"] ** 2 + merged["err_y"] ** 2)
        summary = pd.DataFrame([
            {
                "n": len(merged),
                "rmse_x": float(np.sqrt(np.mean(merged["err_x"] ** 2))),
                "rmse_y": float(np.sqrt(np.mean(merged["err_y"] ** 2))),
                "rmse_xy": float(np.sqrt(np.mean(merged["err_x"] ** 2 + merged["err_y"] ** 2))),
                "mae_x": float(np.mean(np.abs(merged["err_x"]))),
                "mae_y": float(np.mean(np.abs(merged["err_y"]))),
                "mae_xy": float(np.mean(merged["err_xy"])),
            }
        ])
        merged.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    return summary


def local_predict_and_evaluate(args: argparse.Namespace) -> None:
    cfg = config_from_args(args)
    cfg.verbose = True
    predictor = BayesianTrajectoryPredictor(cfg).fit()

    test_input_path = Path(args.test_input)
    if not test_input_path.exists():
        raise FileNotFoundError(f"test_input.csv not found: {test_input_path}")
    test_input = pd.read_csv(test_input_path, low_memory=False)

    test_path = Path(args.test) if args.test else None
    if test_path and test_path.exists():
        template = pd.read_csv(test_path, low_memory=False)
    else:
        template = make_prediction_template_from_test_input(test_input, cfg.default_num_future_frames)

    truth = read_truth_if_available(args.truth, template, args.test_submit)
    test_for_prediction = template.drop(columns=[c for c in ["x", "y"] if c in template.columns])
    pred_xy = predictor.predict_pandas(test_for_prediction, test_input)
    prediction_with_ids = pd.concat([test_for_prediction[ID_COLS + ["frame_id"]].reset_index(drop=True), pred_xy], axis=1)

    output_path = Path(args.output)
    prediction_with_ids.to_csv(output_path, index=False)
    print(f"[saved] predictions: {output_path}")

    if truth is not None:
        detail_path = Path(args.error_detail_output)
        summary_path = Path(args.error_summary_output)
        summary = evaluate_predictions(prediction_with_ids, truth, detail_path, summary_path)
        print(f"[saved] error detail: {detail_path}")
        print(f"[saved] error summary: {summary_path}")
        print(summary.to_string(index=False))
    else:
        print("[info] No truth file with columns game_id, play_id, nfl_id, frame_id, x, y was found; error was not computed.")

    config_path = Path(args.config_output)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
    print(f"[saved] config: {config_path}")


def config_from_args(args: argparse.Namespace) -> PredictorConfig:
    cfg = PredictorConfig()
    cfg.train_dir = args.train_dir
    cfg.x_bin = args.x_bin
    cfg.y_bin = args.y_bin
    cfg.ball_bin = args.ball_bin
    cfg.yardline_bin = args.yardline_bin
    cfg.age_bin = args.age_bin
    cfg.weight_bin = args.weight_bin
    cfg.alpha_state_prior = args.alpha_state_prior
    cfg.beta_external_global = args.beta_external_global
    cfg.lambda_external = args.lambda_external
    cfg.tau_control_shrinkage = args.tau_control_shrinkage
    cfg.candidate_x_radius = args.candidate_x_radius
    cfg.candidate_y_radius = args.candidate_y_radius
    cfg.candidate_age_radius = args.candidate_age_radius
    cfg.candidate_weight_radius = args.candidate_weight_radius
    cfg.min_candidate_states = args.min_candidate_states
    cfg.max_candidate_states = args.max_candidate_states
    cfg.h_x = args.h_x
    cfg.h_y = args.h_y
    cfg.h_age = args.h_age
    cfg.h_weight = args.h_weight
    cfg.cat_mismatch_penalty = args.cat_mismatch_penalty
    cfg.default_num_future_frames = args.num_future_frames
    cfg.dt = args.dt
    cfg.clip_field = not args.no_clip_field
    cfg.max_train_files = args.max_train_files
    cfg.require_output = not args.allow_no_output
    return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bayesian conditional trajectory predictor")
    parser.add_argument("--version", action="version", version=f"%(prog)s {SCRIPT_VERSION}")

    default_train = str(DEFAULT_TRAIN_DIR if DEFAULT_TRAIN_DIR.exists() else Path("./train"))
    parser.add_argument("--train-dir", type=str, default=default_train)
    parser.add_argument("--test-input", type=str, default="./test_input.csv")
    parser.add_argument("--test", type=str, default=None, help="Optional local Kaggle test template CSV. This may contain no x,y.")
    parser.add_argument("--test-submit", "--test_submit", dest="test_submit", type=str, default=None, help="Optional CSV containing true x,y for rows in --test, for example ./test_submit.csv.")
    parser.add_argument("--truth", type=str, default=None, help="Optional truth CSV with x,y for error calculation. Kept for backward compatibility.")
    parser.add_argument("--output", type=str, default="predictions.csv")
    parser.add_argument("--error-detail-output", type=str, default="prediction_error_detail.csv")
    parser.add_argument("--error-summary-output", type=str, default="prediction_error_summary.csv")
    parser.add_argument("--config-output", type=str, default="predictor_run_config.json")

    # Server behavior is opt-in.
    parser.add_argument("--serve", action="store_true", help="Start Kaggle inference_server.serve(). Default is off.")
    parser.add_argument("--run-local-gateway", action="store_true", help="Run Kaggle local gateway. Default is off.")
    parser.add_argument(
        "--competition-data-dir",
        type=str,
        default=str(DEFAULT_COMPETITION_DIR),
        help="Root directory passed to run_local_gateway.",
    )

    # Bins
    parser.add_argument("--x-bin", type=float, default=5.0)
    parser.add_argument("--y-bin", type=float, default=2.0)
    parser.add_argument("--ball-bin", type=float, default=5.0)
    parser.add_argument("--yardline-bin", type=float, default=5.0)
    parser.add_argument("--age-bin", type=float, default=3.0)
    parser.add_argument("--weight-bin", type=float, default=20.0)

    # Bayesian smoothing
    parser.add_argument("--alpha-state-prior", type=float, default=1.0)
    parser.add_argument("--beta-external-global", type=float, default=1.0)
    parser.add_argument("--lambda-external", type=float, default=20.0)
    parser.add_argument("--tau-control-shrinkage", type=float, default=30.0)

    # Candidate search
    parser.add_argument("--candidate-x-radius", type=float, default=10.0)
    parser.add_argument("--candidate-y-radius", type=float, default=4.0)
    parser.add_argument("--candidate-age-radius", type=float, default=6.0)
    parser.add_argument("--candidate-weight-radius", type=float, default=40.0)
    parser.add_argument("--min-candidate-states", type=int, default=20)
    parser.add_argument("--max-candidate-states", type=int, default=50)
    parser.add_argument("--h-x", type=float, default=10.0)
    parser.add_argument("--h-y", type=float, default=4.0)
    parser.add_argument("--h-age", type=float, default=6.0)
    parser.add_argument("--h-weight", type=float, default=40.0)
    parser.add_argument("--cat-mismatch-penalty", type=float, default=0.05)

    # Local prediction
    parser.add_argument("--num-future-frames", type=int, default=30)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--no-clip-field", action="store_true")
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--allow-no-output", action="store_true")

    return parser


def configure_global_from_args(args: argparse.Namespace) -> None:
    global _GLOBAL_CONFIG, _GLOBAL_PREDICTOR
    _GLOBAL_CONFIG = config_from_args(args)
    _GLOBAL_CONFIG.verbose = True
    _GLOBAL_PREDICTOR = None


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    configure_global_from_args(args)

    if args.serve or args.run_local_gateway:
        if nfl_inference_server is None:
            raise RuntimeError("kaggle_evaluation.nfl_inference_server is not available in this environment.")
        server = nfl_inference_server.NFLInferenceServer(predict)
        if args.serve:
            server.serve()
        else:
            server.run_local_gateway((args.competition_data_dir,))
        return

    # Default behavior: no server. Predict current ./test_input.csv and compute error if truth exists.
    local_predict_and_evaluate(args)


if __name__ == "__main__":
    main()
