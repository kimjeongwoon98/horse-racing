#!/usr/bin/env python3
# bayesian_conditional_control_distribution.py
#
# NFL Big Data Bowl style tracking data
# Bayesian posterior-weighted conditional control distribution builder.
#
# This script keeps the file-reading / writing pattern from the shared
# conditional_control_distribution*.py scripts, but replaces local ngroup()
# condition ids with stable key/hash ids and builds:
#   1) state-only prior p(S)
#   2) external likelihood p(E | S)
#   3) posterior state weights p(S~ | S_obs, E_obs)
#   4) posterior-weighted control distribution p(U | S_obs, E_obs)
#
# Main output files:
#   - state_prior.csv
#   - external_likelihood.csv
#   - condition_registry.csv
#   - exact_condition_control_distribution_summary.csv
#   - posterior_state_weights.csv
#   - bayesian_conditional_control_distribution_summary.csv
#   - conditional_control_distribution_summary.csv  # backward-compatible alias
#   - conditional_trajectory_membership.csv
#   - trajectory_control_distribution_summary.csv
#   - run_config.txt

"""
python3 bayesian_conditional_control_distribution.py \
  --train-dir ./train \
  --out-dir ./bayesian_conditional_control_distribution \
  --unit last_frame

  python3 bayesian_conditional_control_distribution.py \
  --train-dir ./train \
  --out-dir ./bayesian_conditional_control_distribution_debug \
  --unit last_frame \
  --max-files 1 \
  --max-query-conditions 100
"""

import os
import re
import json
import math
import hashlib
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ID_COLS = ["game_id", "play_id", "nfl_id"]
TRAJ_COLS = ["source_year", "source_week", "game_id", "play_id", "nfl_id"]

CONTROL_COLS = ["s", "a", "o", "dir"]
ANGLE_CONTROL_COLS = ["o", "dir"]

STATE_CATEGORICAL_COLS = [
    "player_position",
    "player_side",
    "player_role",
]

EXTERNAL_CATEGORICAL_COLS = [
    "play_direction",
]

STATE_NUMERIC_RAW_COLS = [
    "player_weight",
    "player_birth_date",
    "x",
    "y",
]

EXTERNAL_NUMERIC_RAW_COLS = [
    "ball_land_x",
    "ball_land_y",
    "absolute_yardline_number",
]

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


# =============================================================================
# I/O helpers retained from the user's shared code structure
# =============================================================================

def discover_input_output_pairs(train_dir: Path):
    """
    input_2023_w01.csv -> output_2023_w01.csv 자동 매칭.
    """
    pairs = []
    pattern = re.compile(r"^input_(\d{4})_w(\d+)\.csv$")

    for input_path in sorted(train_dir.glob("input_*_w*.csv")):
        m = pattern.match(input_path.name)
        if not m:
            continue

        year = int(m.group(1))
        week_str = m.group(2)
        week = int(week_str)

        output_name = f"output_{year}_w{week_str}.csv"
        output_path = train_dir / output_name

        if not output_path.exists():
            print(f"[warning] output file not found for {input_path.name}: {output_name}")
            output_path = None

        pairs.append(
            {
                "year": year,
                "week": week,
                "input_path": input_path,
                "output_path": output_path,
            }
        )

    return pairs


def read_csv_selected(path: Path, wanted_cols):
    """
    필요한 column만 읽어서 메모리 사용을 줄임.
    """
    header = pd.read_csv(path, nrows=0)
    available = list(header.columns)

    usecols = [c for c in wanted_cols if c in available]

    missing = [c for c in wanted_cols if c not in available]
    if missing:
        print(f"[warning] {path.name}: missing columns ignored: {missing}")

    return pd.read_csv(path, usecols=usecols, low_memory=False)


def add_age_and_bins(
    df,
    source_year,
    x_bin=5.0,
    y_bin=2.0,
    ball_bin=5.0,
    yardline_bin=5.0,
    age_bin=3.0,
    weight_bin=20.0,
):
    """
    연속형 condition 변수는 exact value로 묶으면 거의 전부 singleton이 되므로 binning 사용.
    *_bin column은 bin의 lower bound를 의미.
    예: x_bin = 35이면 [35, 40) 구간.
    """
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

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "player_birth_date" in df.columns:
        birth = pd.to_datetime(df["player_birth_date"], errors="coerce")
        ref_date = pd.Timestamp(f"{source_year}-09-01")
        df["player_age"] = (ref_date - birth).dt.days / 365.25
    else:
        df["player_age"] = np.nan

    def make_bin(src_col, dst_col, width):
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

    categorical_cols = STATE_CATEGORICAL_COLS + EXTERNAL_CATEGORICAL_COLS
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("MISSING").astype(str)

    return df


def summarize_output_file(output_path: Path, source_year: int, source_week: int):
    """
    output csv는 미래 위치 x, y만 있으므로 control variable 계산에는 직접 쓰지 않음.
    대신 input trajectory와 매칭해서 future target 존재 여부 및 endpoint 정보를 저장.
    """
    if output_path is None:
        return None

    wanted = ID_COLS + ["frame_id", "x", "y"]
    out = read_csv_selected(output_path, wanted)

    for col in ID_COLS + ["frame_id", "x", "y"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=ID_COLS)

    if "frame_id" in out.columns:
        out = out.sort_values(ID_COLS + ["frame_id"])
    else:
        out = out.sort_values(ID_COLS)

    grouped = out.groupby(ID_COLS, dropna=False, sort=False)

    if "frame_id" in out.columns:
        stats = grouped.agg(
            output_num_frames=("frame_id", "nunique"),
            output_first_frame=("frame_id", "min"),
            output_last_frame=("frame_id", "max"),
            output_first_x=("x", "first"),
            output_first_y=("y", "first"),
            output_end_x=("x", "last"),
            output_end_y=("y", "last"),
        ).reset_index()
    else:
        stats = grouped.agg(
            output_num_frames=("x", "size"),
            output_first_x=("x", "first"),
            output_first_y=("y", "first"),
            output_end_x=("x", "last"),
            output_end_y=("y", "last"),
        ).reset_index()

    stats["source_year"] = source_year
    stats["source_week"] = source_week

    return stats


# =============================================================================
# Stable key design: no pandas ngroup() condition ids
# =============================================================================

def _is_missing_value(v):
    try:
        return pd.isna(v)
    except Exception:
        return False


def format_key_value(v):
    """Stable string representation for a key value."""
    if _is_missing_value(v):
        return "NA"
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    if isinstance(v, (np.floating, float)):
        if not np.isfinite(v):
            return "NA"
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.6g}"
    text = str(v)
    return text.replace("|", "/").replace("=", ":").replace("\n", " ").strip()


def make_key_series(df: pd.DataFrame, cols, key_name: str):
    """
    Create a deterministic text key from selected columns.
    Example: player_position=WR|x_bin=45|y_bin=22
    """
    existing_cols = [c for c in cols if c in df.columns]
    if len(existing_cols) == 0:
        return pd.Series([f"{key_name}=EMPTY"] * len(df), index=df.index)

    def row_to_key(row):
        parts = [f"{c}={format_key_value(row[c])}" for c in existing_cols]
        return "|".join(parts)

    return df[existing_cols].apply(row_to_key, axis=1)


def stable_hash_id(text: str, prefix="cid"):
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=8).hexdigest()
    return f"{prefix}_{digest}"


def add_stable_condition_keys(df, state_key_cols, external_key_cols):
    df = df.copy()
    state_key_cols = [c for c in state_key_cols if c in df.columns]
    external_key_cols = [c for c in external_key_cols if c in df.columns]

    df["state_key"] = make_key_series(df, state_key_cols, "state")
    df["external_key"] = make_key_series(df, external_key_cols, "external")
    df["condition_key"] = df["state_key"] + " || " + df["external_key"]
    df["condition_id"] = df["condition_key"].map(lambda x: stable_hash_id(x, "cond"))
    df["state_id"] = df["state_key"].map(lambda x: stable_hash_id(x, "state"))
    df["external_id"] = df["external_key"].map(lambda x: stable_hash_id(x, "ext"))

    return df, state_key_cols, external_key_cols


# =============================================================================
# Summary statistics for control distributions
# =============================================================================

def circular_mean_deg(x):
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if len(vals) == 0:
        return np.nan

    rad = np.deg2rad(vals % 360)
    mean_sin = np.sin(rad).mean()
    mean_cos = np.cos(rad).mean()

    return (np.rad2deg(np.arctan2(mean_sin, mean_cos)) + 360) % 360


def circular_std_deg(x):
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if len(vals) == 0:
        return np.nan

    rad = np.deg2rad(vals % 360)
    mean_sin = np.sin(rad).mean()
    mean_cos = np.cos(rad).mean()

    R = np.sqrt(mean_sin ** 2 + mean_cos ** 2)
    R = np.clip(R, 1e-12, 1.0)

    return np.rad2deg(np.sqrt(-2 * np.log(R)))


def _nanmean_square(z):
    vals = pd.to_numeric(z, errors="coerce").dropna().astype(float)
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals * vals))


def _sin_mean_deg(z):
    vals = pd.to_numeric(z, errors="coerce").dropna()
    if len(vals) == 0:
        return np.nan
    return float(np.sin(np.deg2rad(vals % 360)).mean())


def _cos_mean_deg(z):
    vals = pd.to_numeric(z, errors="coerce").dropna()
    if len(vals) == 0:
        return np.nan
    return float(np.cos(np.deg2rad(vals % 360)).mean())


def summarize_control_by_group(df, group_cols, extra_count_cols=True):
    """
    Summarize control variables by group.
    Stores both ordinary moments and sin/cos moments for angle controls.
    """
    work = df.copy()
    for col in CONTROL_COLS:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    agg = {}
    if extra_count_cols:
        agg.update(
            {
                "n_rows": (group_cols[0], "size"),
                "n_trajectories": ("_trajectory_id", "nunique"),
                "n_games": ("game_id", "nunique"),
                "n_plays": ("play_id", "nunique"),
                "n_players": ("nfl_id", "nunique"),
            }
        )
    else:
        agg["n_rows"] = (group_cols[0], "size")

    for c in CONTROL_COLS:
        if c not in work.columns:
            continue
        agg[f"{c}_count"] = (c, lambda z: pd.to_numeric(z, errors="coerce").notna().sum())
        agg[f"{c}_mean"] = (c, "mean")
        agg[f"{c}_second_moment"] = (c, _nanmean_square)
        agg[f"{c}_min"] = (c, "min")
        agg[f"{c}_q05"] = (c, lambda z: pd.to_numeric(z, errors="coerce").quantile(0.05))
        agg[f"{c}_q25"] = (c, lambda z: pd.to_numeric(z, errors="coerce").quantile(0.25))
        agg[f"{c}_q50"] = (c, lambda z: pd.to_numeric(z, errors="coerce").quantile(0.50))
        agg[f"{c}_q75"] = (c, lambda z: pd.to_numeric(z, errors="coerce").quantile(0.75))
        agg[f"{c}_q95"] = (c, lambda z: pd.to_numeric(z, errors="coerce").quantile(0.95))
        agg[f"{c}_max"] = (c, "max")

        if c in ANGLE_CONTROL_COLS:
            agg[f"{c}_sin_mean"] = (c, _sin_mean_deg)
            agg[f"{c}_cos_mean"] = (c, _cos_mean_deg)
            agg[f"{c}_circular_mean"] = (c, circular_mean_deg)
            agg[f"{c}_circular_std"] = (c, circular_std_deg)

    summary = work.groupby(group_cols, dropna=False, sort=False).agg(**agg).reset_index()

    # Convert second moment to population variance/std.
    for c in CONTROL_COLS:
        mean_col = f"{c}_mean"
        second_col = f"{c}_second_moment"
        if mean_col in summary.columns and second_col in summary.columns:
            var_col = f"{c}_var"
            std_col = f"{c}_std"
            summary[var_col] = summary[second_col] - summary[mean_col] ** 2
            summary[var_col] = summary[var_col].clip(lower=0)
            summary[std_col] = np.sqrt(summary[var_col])

    return summary


def summarize_trajectory_distribution(all_rows, state_key_cols, external_key_cols):
    """
    각 trajectory, 즉 source_year/source_week/game_id/play_id/nfl_id별
    control variable의 시간축 분포를 저장.
    """
    all_rows = all_rows.sort_values(TRAJ_COLS + ["frame_id"])

    agg = {
        "n_input_frames": ("frame_id", "nunique"),
        "input_first_frame": ("frame_id", "min"),
        "input_last_frame": ("frame_id", "max"),
    }

    for c in CONTROL_COLS:
        if c not in all_rows.columns:
            continue
        agg[f"{c}_mean_over_input"] = (c, "mean")
        agg[f"{c}_std_over_input"] = (c, "std")
        agg[f"{c}_min_over_input"] = (c, "min")
        agg[f"{c}_q25_over_input"] = (c, lambda z: z.quantile(0.25))
        agg[f"{c}_q50_over_input"] = (c, lambda z: z.quantile(0.50))
        agg[f"{c}_q75_over_input"] = (c, lambda z: z.quantile(0.75))
        agg[f"{c}_max_over_input"] = (c, "max")
        agg[f"{c}_first_input"] = (c, "first")
        agg[f"{c}_last_input"] = (c, "last")

    traj_summary = (
        all_rows.groupby(TRAJ_COLS, dropna=False, sort=False)
        .agg(**agg)
        .reset_index()
    )

    last_rows = (
        all_rows.sort_values(TRAJ_COLS + ["frame_id"])
        .groupby(TRAJ_COLS, dropna=False, sort=False)
        .tail(1)
    )

    extra_cols = [
        "state_id",
        "external_id",
        "condition_id",
        "state_key",
        "external_key",
        "condition_key",
        *state_key_cols,
        *external_key_cols,
        "output_num_frames",
        "output_first_frame",
        "output_last_frame",
        "output_first_x",
        "output_first_y",
        "output_end_x",
        "output_end_y",
    ]

    existing_extra_cols = [c for c in extra_cols if c in last_rows.columns]
    last_info = last_rows[TRAJ_COLS + existing_extra_cols].drop_duplicates(TRAJ_COLS)
    traj_summary = traj_summary.merge(last_info, on=TRAJ_COLS, how="left")

    return traj_summary


# =============================================================================
# Bayesian model tables
# =============================================================================

def build_state_prior(df_unit, state_key_cols, alpha=1.0):
    state_values = (
        df_unit.groupby("state_key", dropna=False, sort=False)[state_key_cols + ["state_id"]]
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

    state_prior = state_values.merge(state_counts, on="state_key", how="left")
    n_total = len(df_unit)
    n_states = max(1, len(state_prior))
    state_prior["prior_prob"] = (state_prior["n_state"] + alpha) / (n_total + alpha * n_states)

    return state_prior


def build_external_registry(df_unit, external_key_cols, beta=1.0):
    external_values = (
        df_unit.groupby("external_key", dropna=False, sort=False)[external_key_cols + ["external_id"]]
        .first()
        .reset_index()
    )

    external_counts = (
        df_unit.groupby("external_key", dropna=False, sort=False)
        .agg(n_external=("external_key", "size"))
        .reset_index()
    )

    external_registry = external_values.merge(external_counts, on="external_key", how="left")
    n_total = len(df_unit)
    n_external = max(1, len(external_registry))
    external_registry["global_external_prob"] = (external_registry["n_external"] + beta) / (
        n_total + beta * n_external
    )

    return external_registry


def build_condition_registry(df_unit, state_key_cols, external_key_cols):
    condition_values_cols = [
        "condition_id",
        "state_id",
        "external_id",
        "state_key",
        "external_key",
        *state_key_cols,
        *external_key_cols,
    ]
    condition_values_cols = [c for c in condition_values_cols if c in df_unit.columns]

    condition_values = (
        df_unit.groupby("condition_key", dropna=False, sort=False)[condition_values_cols]
        .first()
        .reset_index()
    )

    condition_counts = (
        df_unit.groupby("condition_key", dropna=False, sort=False)
        .agg(
            n_condition=("condition_key", "size"),
            n_condition_trajectories=("_trajectory_id", "nunique"),
            n_condition_games=("game_id", "nunique"),
            n_condition_plays=("play_id", "nunique"),
            n_condition_players=("nfl_id", "nunique"),
        )
        .reset_index()
    )

    registry = condition_values.merge(condition_counts, on="condition_key", how="left")
    return registry


def build_external_likelihood(
    condition_registry,
    state_prior,
    external_registry,
    lambda_external=20.0,
):
    """
    Observed-pair likelihood table.
    For unobserved (state, external) pairs, use lookup_external_likelihood().
    """
    state_n = state_prior[["state_key", "n_state"]]
    external_p0 = external_registry[["external_key", "global_external_prob"]]

    table = condition_registry[["state_key", "external_key", "n_condition"]].copy()
    table = table.rename(columns={"n_condition": "n_state_external"})
    table = table.merge(state_n, on="state_key", how="left")
    table = table.merge(external_p0, on="external_key", how="left")
    table["lambda_external"] = float(lambda_external)
    table["likelihood_external_given_state"] = (
        table["n_state_external"] + lambda_external * table["global_external_prob"]
    ) / (table["n_state"] + lambda_external)

    return table


# =============================================================================
# Candidate state posterior and posterior-weighted control distribution
# =============================================================================

def _safe_float(v, default=np.nan):
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def build_lookup_dicts(state_prior, external_registry, condition_registry):
    state_count = dict(zip(state_prior["state_key"], state_prior["n_state"]))
    state_prior_prob = dict(zip(state_prior["state_key"], state_prior["prior_prob"]))
    external_p0 = dict(zip(external_registry["external_key"], external_registry["global_external_prob"]))

    joint_count = {}
    cond_id_lookup = {}
    condition_key_lookup = {}
    for _, row in condition_registry.iterrows():
        key = (row["state_key"], row["external_key"])
        joint_count[key] = int(row["n_condition"])
        cond_id_lookup[key] = row["condition_id"]
        condition_key_lookup[key] = row["condition_key"]

    return {
        "state_count": state_count,
        "state_prior_prob": state_prior_prob,
        "external_p0": external_p0,
        "joint_count": joint_count,
        "cond_id_lookup": cond_id_lookup,
        "condition_key_lookup": condition_key_lookup,
    }


def lookup_external_likelihood(state_key, external_key, lookups, lambda_external, default_external_prob):
    n_s = float(lookups["state_count"].get(state_key, 0))
    n_se = float(lookups["joint_count"].get((state_key, external_key), 0))
    p0 = float(lookups["external_p0"].get(external_key, default_external_prob))
    if n_s <= 0:
        return p0
    return (n_se + lambda_external * p0) / (n_s + lambda_external)


def filter_candidate_states(obs_row, state_prior, args):
    """
    Build candidate states with fallback levels.
    Level 0: position + side + role + nearby x/y + nearby age/weight
    Level 1: position + side + role + nearby x/y
    Level 2: side + role + nearby x/y
    Level 3: side + role only
    Level 4: global
    """
    levels = [
        {
            "name": "position_side_role_xy_age_weight",
            "cat_cols": ["player_position", "player_side", "player_role"],
            "use_xy": True,
            "use_age_weight": True,
        },
        {
            "name": "position_side_role_xy",
            "cat_cols": ["player_position", "player_side", "player_role"],
            "use_xy": True,
            "use_age_weight": False,
        },
        {
            "name": "side_role_xy",
            "cat_cols": ["player_side", "player_role"],
            "use_xy": True,
            "use_age_weight": False,
        },
        {
            "name": "side_role",
            "cat_cols": ["player_side", "player_role"],
            "use_xy": False,
            "use_age_weight": False,
        },
        {
            "name": "global",
            "cat_cols": [],
            "use_xy": False,
            "use_age_weight": False,
        },
    ]

    for level in levels:
        cand = state_prior.copy()

        for col in level["cat_cols"]:
            if col in cand.columns and col in obs_row.index:
                obs_val = obs_row[col]
                if not pd.isna(obs_val):
                    cand = cand[cand[col].astype(str) == str(obs_val)]

        if level["use_xy"]:
            if "x_bin" in cand.columns and "x_bin" in obs_row.index and not pd.isna(obs_row["x_bin"]):
                cand = cand[(cand["x_bin"] - obs_row["x_bin"]).abs() <= args.candidate_x_radius]
            if "y_bin" in cand.columns and "y_bin" in obs_row.index and not pd.isna(obs_row["y_bin"]):
                cand = cand[(cand["y_bin"] - obs_row["y_bin"]).abs() <= args.candidate_y_radius]

        if level["use_age_weight"]:
            if "player_age_bin" in cand.columns and "player_age_bin" in obs_row.index and not pd.isna(obs_row["player_age_bin"]):
                cand = cand[(cand["player_age_bin"] - obs_row["player_age_bin"]).abs() <= args.candidate_age_radius]
            if "player_weight_bin" in cand.columns and "player_weight_bin" in obs_row.index and not pd.isna(obs_row["player_weight_bin"]):
                cand = cand[(cand["player_weight_bin"] - obs_row["player_weight_bin"]).abs() <= args.candidate_weight_radius]

        if len(cand) >= args.min_candidate_states or level["name"] == "global":
            cand = cand.copy()
            cand["candidate_level"] = level["name"]
            return cand

    # Should not happen because global returns.
    cand = state_prior.copy()
    cand["candidate_level"] = "global"
    return cand


def compute_state_similarity(obs_row, cand, args):
    """
    Gaussian numeric kernel times categorical mismatch penalty.
    """
    sim = np.ones(len(cand), dtype=float)

    numeric_bandwidths = {
        "x_bin": args.h_x,
        "y_bin": args.h_y,
        "player_age_bin": args.h_age,
        "player_weight_bin": args.h_weight,
    }

    for col, h in numeric_bandwidths.items():
        if col not in cand.columns or col not in obs_row.index:
            continue
        obs_val = _safe_float(obs_row[col])
        if not np.isfinite(obs_val) or h <= 0:
            continue
        cand_vals = pd.to_numeric(cand[col], errors="coerce").astype(float).values
        diff = cand_vals - obs_val
        valid = np.isfinite(diff)
        factor = np.ones(len(cand), dtype=float)
        factor[valid] = np.exp(-0.5 * (diff[valid] / h) ** 2)
        sim *= factor

    # When fallback levels allow category mismatch, keep nonmatching states possible but penalized.
    for col in ["player_position", "player_side", "player_role"]:
        if col not in cand.columns or col not in obs_row.index:
            continue
        obs_val = obs_row[col]
        if pd.isna(obs_val):
            continue
        matches = cand[col].astype(str).values == str(obs_val)
        sim *= np.where(matches, 1.0, args.cat_mismatch_penalty)

    return sim


def compute_posterior_for_query(obs_row, state_prior, lookups, args, default_external_prob):
    external_key = obs_row["external_key"]
    cand = filter_candidate_states(obs_row, state_prior, args)
    cand = cand.copy()

    sim = compute_state_similarity(obs_row, cand, args)
    prior = cand["prior_prob"].astype(float).values

    likelihoods = np.array(
        [
            lookup_external_likelihood(
                state_key=s,
                external_key=external_key,
                lookups=lookups,
                lambda_external=args.lambda_external,
                default_external_prob=default_external_prob,
            )
            for s in cand["state_key"].values
        ],
        dtype=float,
    )

    raw_score = sim * prior * likelihoods
    total = raw_score.sum()
    if not np.isfinite(total) or total <= 0:
        # Fallback to prior among candidates.
        raw_score = prior.copy()
        total = raw_score.sum()

    cand["state_similarity"] = sim
    cand["likelihood_external_given_state"] = likelihoods
    cand["posterior_raw_score"] = raw_score
    cand["posterior_weight"] = raw_score / total if total > 0 else np.ones(len(cand)) / max(1, len(cand))

    cand = cand.sort_values("posterior_weight", ascending=False)
    if args.max_candidate_states is not None and args.max_candidate_states > 0:
        cand = cand.head(args.max_candidate_states).copy()
        # Re-normalize after truncation.
        s = cand["posterior_weight"].sum()
        if s > 0:
            cand["posterior_weight"] = cand["posterior_weight"] / s

    return cand


def component_from_summary(summary_row, component_weight, component_type):
    comp = {
        "component_weight": float(component_weight),
        "component_type": component_type,
        "n_rows": int(summary_row.get("n_rows", 0)),
    }
    for c in CONTROL_COLS:
        mean_col = f"{c}_mean"
        var_col = f"{c}_var"
        second_col = f"{c}_second_moment"
        if mean_col in summary_row.index:
            comp[f"{c}_mean"] = _safe_float(summary_row[mean_col])
        if var_col in summary_row.index:
            comp[f"{c}_var"] = max(0.0, _safe_float(summary_row[var_col], 0.0))
        elif second_col in summary_row.index and mean_col in summary_row.index:
            mean = _safe_float(summary_row[mean_col], 0.0)
            second = _safe_float(summary_row[second_col], mean * mean)
            comp[f"{c}_var"] = max(0.0, second - mean * mean)

        if c in ANGLE_CONTROL_COLS:
            sin_col = f"{c}_sin_mean"
            cos_col = f"{c}_cos_mean"
            if sin_col in summary_row.index:
                comp[f"{c}_sin_mean"] = _safe_float(summary_row[sin_col])
            if cos_col in summary_row.index:
                comp[f"{c}_cos_mean"] = _safe_float(summary_row[cos_col])
    return comp


def mixture_summary_from_components(components):
    """
    Combine component summaries into one approximate mixture summary.
    For numeric controls: uses mixture mean/variance formula.
    For angle controls: mixes sin/cos moments, then converts to circular mean/std.
    """
    result = {}
    if len(components) == 0:
        for c in CONTROL_COLS:
            result[f"{c}_mean"] = np.nan
            result[f"{c}_std"] = np.nan
        return result

    total_w = sum(c["component_weight"] for c in components)
    if total_w <= 0:
        total_w = 1.0
    for comp in components:
        comp["component_weight_norm"] = comp["component_weight"] / total_w

    result["n_components"] = len(components)
    result["component_total_weight_before_norm"] = total_w
    result["component_effective_n_rows"] = sum(
        comp["component_weight_norm"] * comp.get("n_rows", 0) for comp in components
    )

    for c in CONTROL_COLS:
        means = []
        vars_ = []
        weights = []
        for comp in components:
            mean = comp.get(f"{c}_mean", np.nan)
            var = comp.get(f"{c}_var", np.nan)
            w = comp.get("component_weight_norm", 0.0)
            if np.isfinite(mean) and np.isfinite(var) and w > 0:
                means.append(mean)
                vars_.append(var)
                weights.append(w)

        if len(weights) == 0:
            result[f"{c}_mean"] = np.nan
            result[f"{c}_std"] = np.nan
            result[f"{c}_var"] = np.nan
        else:
            weights = np.array(weights, dtype=float)
            means = np.array(means, dtype=float)
            vars_ = np.array(vars_, dtype=float)
            weights = weights / weights.sum()
            mix_mean = float(np.sum(weights * means))
            mix_second = float(np.sum(weights * (vars_ + means ** 2)))
            mix_var = max(0.0, mix_second - mix_mean ** 2)
            result[f"{c}_mean"] = mix_mean
            result[f"{c}_var"] = mix_var
            result[f"{c}_std"] = math.sqrt(mix_var)

        if c in ANGLE_CONTROL_COLS:
            sin_vals = []
            cos_vals = []
            angle_weights = []
            for comp in components:
                sin_v = comp.get(f"{c}_sin_mean", np.nan)
                cos_v = comp.get(f"{c}_cos_mean", np.nan)
                w = comp.get("component_weight_norm", 0.0)
                if np.isfinite(sin_v) and np.isfinite(cos_v) and w > 0:
                    sin_vals.append(sin_v)
                    cos_vals.append(cos_v)
                    angle_weights.append(w)
            if len(angle_weights) > 0:
                angle_weights = np.array(angle_weights, dtype=float)
                angle_weights = angle_weights / angle_weights.sum()
                mean_sin = float(np.sum(angle_weights * np.array(sin_vals)))
                mean_cos = float(np.sum(angle_weights * np.array(cos_vals)))
                circ_mean = (np.rad2deg(np.arctan2(mean_sin, mean_cos)) + 360) % 360
                R = np.sqrt(mean_sin ** 2 + mean_cos ** 2)
                R = np.clip(R, 1e-12, 1.0)
                circ_std = np.rad2deg(np.sqrt(-2 * np.log(R)))
                result[f"{c}_sin_mean"] = mean_sin
                result[f"{c}_cos_mean"] = mean_cos
                result[f"{c}_circular_mean"] = circ_mean
                result[f"{c}_circular_std"] = circ_std

    return result


def build_bayesian_conditional_summaries(
    condition_registry,
    state_prior,
    state_control_summary,
    exact_condition_summary,
    lookups,
    args,
    default_external_prob,
):
    """
    For each observed condition, compute:
      posterior p(S~ | S_obs, E_obs)
      posterior-weighted mixture p(U | S_obs, E_obs)
    """
    state_summary_by_key = {
        row["state_key"]: row for _, row in state_control_summary.iterrows()
    }
    condition_summary_by_key = {
        row["condition_key"]: row for _, row in exact_condition_summary.iterrows()
    }

    query_df = condition_registry.copy()
    if args.max_query_conditions is not None and args.max_query_conditions > 0:
        query_df = query_df.head(args.max_query_conditions).copy()

    bayes_summary_records = []
    posterior_records = []

    total_queries = len(query_df)
    print(f"[info] building Bayesian summaries for {total_queries:,} query conditions")

    for idx, (_, qrow) in enumerate(query_df.iterrows(), start=1):
        if idx == 1 or idx % 500 == 0 or idx == total_queries:
            print(f"[info] posterior mixture progress: {idx:,}/{total_queries:,}")

        posterior = compute_posterior_for_query(
            obs_row=qrow,
            state_prior=state_prior,
            lookups=lookups,
            args=args,
            default_external_prob=default_external_prob,
        )

        components = []
        e_key = qrow["external_key"]
        exact_weight_total = 0.0
        state_shrink_weight_total = 0.0

        for rank, (_, prow) in enumerate(posterior.iterrows(), start=1):
            cand_state_key = prow["state_key"]
            posterior_w = float(prow["posterior_weight"])
            n_s = int(lookups["state_count"].get(cand_state_key, 0))
            n_se = int(lookups["joint_count"].get((cand_state_key, e_key), 0))
            rho = n_se / (n_se + args.tau_control_shrinkage) if n_se > 0 else 0.0

            candidate_condition_key = lookups["condition_key_lookup"].get(
                (cand_state_key, e_key), cand_state_key + " || " + e_key
            )
            candidate_condition_id = lookups["cond_id_lookup"].get(
                (cand_state_key, e_key), stable_hash_id(candidate_condition_key, "cond")
            )

            posterior_records.append(
                {
                    "query_condition_id": qrow["condition_id"],
                    "query_condition_key": qrow["condition_key"],
                    "query_state_key": qrow["state_key"],
                    "query_external_key": qrow["external_key"],
                    "candidate_rank": rank,
                    "candidate_state_id": prow.get("state_id", stable_hash_id(cand_state_key, "state")),
                    "candidate_state_key": cand_state_key,
                    "candidate_condition_id": candidate_condition_id,
                    "candidate_condition_key": candidate_condition_key,
                    "candidate_level": prow.get("candidate_level", "unknown"),
                    "posterior_weight": posterior_w,
                    "state_similarity": float(prow["state_similarity"]),
                    "prior_prob": float(prow["prior_prob"]),
                    "likelihood_external_given_state": float(prow["likelihood_external_given_state"]),
                    "n_state": n_s,
                    "n_state_external": n_se,
                    "rho_exact_condition": rho,
                }
            )

            # Exact condition component: p(U | S=s, E=e_obs)
            if n_se > 0 and candidate_condition_key in condition_summary_by_key:
                w_exact = posterior_w * rho
                if w_exact > 0:
                    components.append(
                        component_from_summary(
                            condition_summary_by_key[candidate_condition_key],
                            component_weight=w_exact,
                            component_type="exact_condition",
                        )
                    )
                    exact_weight_total += w_exact

            # State-only shrinkage component: p(U | S=s)
            if cand_state_key in state_summary_by_key:
                w_state = posterior_w * (1.0 - rho)
                if w_state > 0:
                    components.append(
                        component_from_summary(
                            state_summary_by_key[cand_state_key],
                            component_weight=w_state,
                            component_type="state_only_shrinkage",
                        )
                    )
                    state_shrink_weight_total += w_state

        mix = mixture_summary_from_components(components)
        record = {
            "condition_id": qrow["condition_id"],
            "condition_key": qrow["condition_key"],
            "state_id": qrow["state_id"],
            "external_id": qrow["external_id"],
            "state_key": qrow["state_key"],
            "external_key": qrow["external_key"],
            "n_observed_exact_condition": int(qrow["n_condition"]),
            "n_posterior_candidate_states": len(posterior),
            "posterior_top_state_key": posterior.iloc[0]["state_key"] if len(posterior) else None,
            "posterior_top_weight": float(posterior.iloc[0]["posterior_weight"]) if len(posterior) else np.nan,
            "mixture_exact_condition_weight": exact_weight_total,
            "mixture_state_shrinkage_weight": state_shrink_weight_total,
        }

        for col in STATE_KEY_COLS_DEFAULT + EXTERNAL_KEY_COLS_DEFAULT:
            if col in qrow.index:
                record[col] = qrow[col]

        record.update(mix)
        bayes_summary_records.append(record)

    bayes_summary = pd.DataFrame(bayes_summary_records)
    posterior_weights = pd.DataFrame(posterior_records)

    return bayes_summary, posterior_weights


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-dir", type=str, default="./train")
    parser.add_argument("--out-dir", type=str, default="./bayesian_conditional_control_distribution")

    parser.add_argument(
        "--unit",
        type=str,
        default="last_frame",
        choices=["last_frame", "all_frames"],
        help=(
            "last_frame: 각 source_year/source_week/game_id/play_id/nfl_id마다 "
            "마지막 input frame만 사용. 예측 시점 분석에 적합.\n"
            "all_frames: 모든 input frame을 사용."
        ),
    )

    parser.add_argument(
        "--require-output",
        action="store_true",
        help="output csv와 매칭되는 trajectory만 사용.",
    )

    # Existing bin parameters retained.
    parser.add_argument("--x-bin", type=float, default=5.0)
    parser.add_argument("--y-bin", type=float, default=2.0)
    parser.add_argument("--ball-bin", type=float, default=5.0)
    parser.add_argument("--yardline-bin", type=float, default=5.0)
    parser.add_argument("--age-bin", type=float, default=3.0)
    parser.add_argument("--weight-bin", type=float, default=20.0)

    # Bayesian smoothing parameters.
    parser.add_argument("--alpha-state-prior", type=float, default=1.0)
    parser.add_argument("--beta-external-global", type=float, default=1.0)
    parser.add_argument("--lambda-external", type=float, default=20.0)
    parser.add_argument("--tau-control-shrinkage", type=float, default=30.0)

    # Candidate filtering radii; units are bin lower-bound units.
    parser.add_argument("--candidate-x-radius", type=float, default=10.0)
    parser.add_argument("--candidate-y-radius", type=float, default=4.0)
    parser.add_argument("--candidate-age-radius", type=float, default=6.0)
    parser.add_argument("--candidate-weight-radius", type=float, default=40.0)
    parser.add_argument("--min-candidate-states", type=int, default=20)
    parser.add_argument("--max-candidate-states", type=int, default=50)

    # Kernel bandwidths.
    parser.add_argument("--h-x", type=float, default=10.0)
    parser.add_argument("--h-y", type=float, default=4.0)
    parser.add_argument("--h-age", type=float, default=6.0)
    parser.add_argument("--h-weight", type=float, default=40.0)
    parser.add_argument("--cat-mismatch-penalty", type=float, default=0.05)

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="디버깅용. 앞에서부터 몇 개의 input/output pair만 사용할지 지정.",
    )
    parser.add_argument(
        "--max-query-conditions",
        type=int,
        default=None,
        help="디버깅용. Bayesian posterior summary를 앞에서부터 몇 condition에 대해서만 계산.",
    )

    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_input_output_pairs(train_dir)

    if args.max_files is not None:
        pairs = pairs[: args.max_files]

    if len(pairs) == 0:
        raise RuntimeError(f"No input_YYYY_wXX.csv files found in {train_dir}")

    print(f"[info] number of matched input files: {len(pairs)}")

    all_inputs = []

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

    for p in pairs:
        year = p["year"]
        week = p["week"]
        input_path = p["input_path"]
        output_path = p["output_path"]

        print(f"[info] reading input: {input_path.name}")

        inp = read_csv_selected(input_path, input_wanted_cols)

        inp["source_year"] = year
        inp["source_week"] = week

        for col in ID_COLS + ["frame_id"]:
            if col in inp.columns:
                inp[col] = pd.to_numeric(inp[col], errors="coerce")

        inp = inp.dropna(subset=ID_COLS + ["frame_id"])

        inp = add_age_and_bins(
            inp,
            source_year=year,
            x_bin=args.x_bin,
            y_bin=args.y_bin,
            ball_bin=args.ball_bin,
            yardline_bin=args.yardline_bin,
            age_bin=args.age_bin,
            weight_bin=args.weight_bin,
        )

        out_stats = summarize_output_file(output_path, year, week)

        if out_stats is not None:
            inp = inp.merge(out_stats, on=TRAJ_COLS, how="left")
        else:
            inp["output_num_frames"] = np.nan

        all_inputs.append(inp)

    all_rows = pd.concat(all_inputs, ignore_index=True)
    print(f"[info] total input rows: {len(all_rows):,}")

    if args.require_output:
        before = len(all_rows)
        all_rows = all_rows[all_rows["output_num_frames"].notna()].copy()
        after = len(all_rows)
        print(f"[info] require-output filter: {before:,} -> {after:,}")

    # Stable trajectory id retained from your scripts.
    all_rows["_trajectory_id"] = (
        all_rows["source_year"].astype(str)
        + "_w"
        + all_rows["source_week"].astype(str)
        + "_g"
        + all_rows["game_id"].astype(str)
        + "_p"
        + all_rows["play_id"].astype(str)
        + "_n"
        + all_rows["nfl_id"].astype(str)
    )

    state_key_cols = [c for c in STATE_KEY_COLS_DEFAULT if c in all_rows.columns]
    external_key_cols = [c for c in EXTERNAL_KEY_COLS_DEFAULT if c in all_rows.columns]

    all_rows, state_key_cols, external_key_cols = add_stable_condition_keys(
        all_rows,
        state_key_cols=state_key_cols,
        external_key_cols=external_key_cols,
    )

    if args.unit == "last_frame":
        df_unit = (
            all_rows.sort_values(TRAJ_COLS + ["frame_id"])
            .groupby(TRAJ_COLS, dropna=False, sort=False)
            .tail(1)
            .copy()
        )
    else:
        df_unit = all_rows.copy()

    print(f"[info] unit = {args.unit}")
    print(f"[info] rows used for conditional distribution: {len(df_unit):,}")
    print(f"[info] unique trajectories: {df_unit['_trajectory_id'].nunique():,}")
    print(f"[info] state key columns: {state_key_cols}")
    print(f"[info] external key columns: {external_key_cols}")

    # 1) state-only prior p(S)
    state_prior = build_state_prior(
        df_unit=df_unit,
        state_key_cols=state_key_cols,
        alpha=args.alpha_state_prior,
    )
    state_prior_path = out_dir / "state_prior.csv"
    state_prior.to_csv(state_prior_path, index=False)
    print(f"[saved] {state_prior_path}")

    # 2) global external registry p0(E)
    external_registry = build_external_registry(
        df_unit=df_unit,
        external_key_cols=external_key_cols,
        beta=args.beta_external_global,
    )
    external_registry_path = out_dir / "external_registry.csv"
    external_registry.to_csv(external_registry_path, index=False)
    print(f"[saved] {external_registry_path}")

    # 3) condition registry: stable state+external condition_key and condition_id
    condition_registry = build_condition_registry(
        df_unit=df_unit,
        state_key_cols=state_key_cols,
        external_key_cols=external_key_cols,
    )
    condition_registry = condition_registry.merge(
        state_prior[["state_key", "n_state", "prior_prob"]],
        on="state_key",
        how="left",
    )
    condition_registry_path = out_dir / "condition_registry.csv"
    condition_registry.to_csv(condition_registry_path, index=False)
    print(f"[saved] {condition_registry_path}")

    # 4) likelihood p(E | S) for observed state-external pairs
    external_likelihood = build_external_likelihood(
        condition_registry=condition_registry,
        state_prior=state_prior,
        external_registry=external_registry,
        lambda_external=args.lambda_external,
    )
    external_likelihood_path = out_dir / "external_likelihood.csv"
    external_likelihood.to_csv(external_likelihood_path, index=False)
    print(f"[saved] {external_likelihood_path}")

    # 5) exact condition and state-only control summaries
    state_control_summary = summarize_control_by_group(df_unit, ["state_key"], extra_count_cols=True)
    state_control_summary = state_prior.merge(state_control_summary, on="state_key", how="left")
    state_control_summary_path = out_dir / "state_control_distribution_summary.csv"
    state_control_summary.to_csv(state_control_summary_path, index=False)
    print(f"[saved] {state_control_summary_path}")

    exact_condition_summary = summarize_control_by_group(df_unit, ["condition_key"], extra_count_cols=True)
    exact_condition_summary = condition_registry.merge(exact_condition_summary, on="condition_key", how="left")
    exact_condition_summary_path = out_dir / "exact_condition_control_distribution_summary.csv"
    exact_condition_summary.to_csv(exact_condition_summary_path, index=False)
    print(f"[saved] {exact_condition_summary_path}")

    # 6) posterior p(S~ | S_obs, E_obs), then mixture p(U | S_obs, E_obs)
    lookups = build_lookup_dicts(
        state_prior=state_prior,
        external_registry=external_registry,
        condition_registry=condition_registry,
    )
    default_external_prob = 1.0 / max(1, len(external_registry))

    bayes_summary, posterior_weights = build_bayesian_conditional_summaries(
        condition_registry=condition_registry,
        state_prior=state_prior,
        state_control_summary=state_control_summary,
        exact_condition_summary=exact_condition_summary,
        lookups=lookups,
        args=args,
        default_external_prob=default_external_prob,
    )

    posterior_weights_path = out_dir / "posterior_state_weights.csv"
    posterior_weights.to_csv(posterior_weights_path, index=False)
    print(f"[saved] {posterior_weights_path}")

    bayes_summary_path = out_dir / "bayesian_conditional_control_distribution_summary.csv"
    bayes_summary.to_csv(bayes_summary_path, index=False)
    print(f"[saved] {bayes_summary_path}")

    # Backward-compatible name: now points to Bayesian summary, not exact ngroup summary.
    backward_summary_path = out_dir / "conditional_control_distribution_summary.csv"
    bayes_summary.to_csv(backward_summary_path, index=False)
    print(f"[saved] {backward_summary_path}")

    # 7) condition_id별 어떤 play_id/nfl_id가 들어갔는지
    membership = (
        df_unit.groupby(["condition_id"] + TRAJ_COLS, dropna=False, sort=False)
        .size()
        .reset_index(name="n_rows_in_condition")
    )
    membership = membership.merge(
        df_unit[["condition_id", "condition_key", "state_key", "external_key"]]
        .drop_duplicates("condition_id"),
        on="condition_id",
        how="left",
    )
    membership_path = out_dir / "conditional_trajectory_membership.csv"
    membership.to_csv(membership_path, index=False)
    print(f"[saved] {membership_path}")

    # 8) trajectory별 control variable의 시간축 분포
    trajectory_summary = summarize_trajectory_distribution(
        all_rows,
        state_key_cols=state_key_cols,
        external_key_cols=external_key_cols,
    )
    trajectory_summary_path = out_dir / "trajectory_control_distribution_summary.csv"
    trajectory_summary.to_csv(trajectory_summary_path, index=False)
    print(f"[saved] {trajectory_summary_path}")

    # 9) 설정 저장
    config = {
        "train_dir": str(train_dir),
        "out_dir": str(out_dir),
        "unit": args.unit,
        "require_output": args.require_output,
        "x_bin": args.x_bin,
        "y_bin": args.y_bin,
        "ball_bin": args.ball_bin,
        "yardline_bin": args.yardline_bin,
        "age_bin": args.age_bin,
        "weight_bin": args.weight_bin,
        "state_key_cols": state_key_cols,
        "external_key_cols": external_key_cols,
        "control_cols": CONTROL_COLS,
        "alpha_state_prior": args.alpha_state_prior,
        "beta_external_global": args.beta_external_global,
        "lambda_external": args.lambda_external,
        "tau_control_shrinkage": args.tau_control_shrinkage,
        "candidate_x_radius": args.candidate_x_radius,
        "candidate_y_radius": args.candidate_y_radius,
        "candidate_age_radius": args.candidate_age_radius,
        "candidate_weight_radius": args.candidate_weight_radius,
        "min_candidate_states": args.min_candidate_states,
        "max_candidate_states": args.max_candidate_states,
        "h_x": args.h_x,
        "h_y": args.h_y,
        "h_age": args.h_age,
        "h_weight": args.h_weight,
        "cat_mismatch_penalty": args.cat_mismatch_penalty,
        "max_files": args.max_files,
        "max_query_conditions": args.max_query_conditions,
    }

    config_path = out_dir / "run_config.txt"
    with open(config_path, "w", encoding="utf-8") as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")

    # Machine-readable config too.
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"[saved] {config_path}")
    print("[done] Bayesian conditional control distribution completed.")


if __name__ == "__main__":
    main()
