#!/usr/bin/env python3
# conditional_control_distribution_ver4_gif_improved.py
# 수정 완료: 한 GIF 안에 있는 서로 다른 trajectory(객체)마다 **완전히 다른 색상** + Legend 추가

import os
import re
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


ID_COLS = ["game_id", "play_id", "nfl_id"]
TRAJ_COLS = ["source_year", "source_week", "game_id", "play_id", "nfl_id"]

CONTROL_COLS = ["s", "a", "o", "dir"]

STATE_CATEGORICAL_COLS = ["player_position", "player_side", "player_role"]
EXTERNAL_CATEGORICAL_COLS = ["play_direction"]

STATE_NUMERIC_RAW_COLS = ["player_weight", "player_birth_date", "x", "y"]
EXTERNAL_NUMERIC_RAW_COLS = ["ball_land_x", "ball_land_y", "absolute_yardline_number"]


def discover_input_output_pairs(train_dir: Path):
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
        pairs.append({
            "year": year,
            "week": week,
            "input_path": input_path,
            "output_path": output_path,
        })
    return pairs


def read_csv_selected(path: Path, wanted_cols):
    header = pd.read_csv(path, nrows=0)
    available = list(header.columns)
    usecols = [c for c in wanted_cols if c in available]
    missing = [c for c in wanted_cols if c not in available]
    if missing:
        print(f"[warning] {path.name}: missing columns ignored: {missing}")
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def add_age_and_bins(df, source_year, x_bin=5.0, y_bin=2.0, ball_bin=5.0,
                     yardline_bin=5.0, age_bin=3.0, weight_bin=20.0):
    numeric_cols = ["player_weight", "x", "y", "ball_land_x", "ball_land_y",
                    "absolute_yardline_number", "s", "a", "o", "dir"]
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


def build_condition_id(df, condition_cols):
    df["condition_id"] = df.groupby(condition_cols, dropna=False, sort=False).ngroup().astype(int) + 1
    return df


def summarize_conditional_distribution(df_unit, condition_cols):
    for col in CONTROL_COLS:
        if col in df_unit.columns:
            df_unit[col] = pd.to_numeric(df_unit[col], errors="coerce")
    agg = {"n_rows": ("condition_id", "size"), "n_trajectories": ("_trajectory_id", "nunique"),
           "n_games": ("game_id", "nunique"), "n_plays": ("play_id", "nunique"),
           "n_players": ("nfl_id", "nunique")}
    for c in CONTROL_COLS:
        if c not in df_unit.columns: continue
        agg[f"{c}_mean"] = (c, "mean")
        agg[f"{c}_std"] = (c, "std")
        agg[f"{c}_min"] = (c, "min")
        agg[f"{c}_q05"] = (c, lambda z: z.quantile(0.05))
        agg[f"{c}_q25"] = (c, lambda z: z.quantile(0.25))
        agg[f"{c}_q50"] = (c, lambda z: z.quantile(0.50))
        agg[f"{c}_q75"] = (c, lambda z: z.quantile(0.75))
        agg[f"{c}_q95"] = (c, lambda z: z.quantile(0.95))
        agg[f"{c}_max"] = (c, "max")
        if c in ["o", "dir"]:
            agg[f"{c}_circular_mean"] = (c, circular_mean_deg)
            agg[f"{c}_circular_std"] = (c, circular_std_deg)
    summary = df_unit.groupby("condition_id", dropna=False, sort=False).agg(**agg).reset_index()
    condition_values = df_unit.groupby("condition_id", dropna=False, sort=False)[condition_cols].first().reset_index()
    summary = condition_values.merge(summary, on="condition_id", how="left")
    return summary


def summarize_trajectory_distribution(all_rows, condition_cols):
    all_rows = all_rows.sort_values(TRAJ_COLS + ["frame_id"])
    agg = {"n_input_frames": ("frame_id", "nunique"),
           "input_first_frame": ("frame_id", "min"),
           "input_last_frame": ("frame_id", "max")}
    for c in CONTROL_COLS:
        if c not in all_rows.columns: continue
        agg[f"{c}_mean_over_input"] = (c, "mean")
        agg[f"{c}_std_over_input"] = (c, "std")
        agg[f"{c}_min_over_input"] = (c, "min")
        agg[f"{c}_q25_over_input"] = (c, lambda z: z.quantile(0.25))
        agg[f"{c}_q50_over_input"] = (c, lambda z: z.quantile(0.50))
        agg[f"{c}_q75_over_input"] = (c, lambda z: z.quantile(0.75))
        agg[f"{c}_max_over_input"] = (c, "max")
        agg[f"{c}_first_input"] = (c, "first")
        agg[f"{c}_last_input"] = (c, "last")
    traj_summary = all_rows.groupby(TRAJ_COLS, dropna=False, sort=False).agg(**agg).reset_index()
    last_rows = all_rows.sort_values(TRAJ_COLS + ["frame_id"]).groupby(TRAJ_COLS, dropna=False, sort=False).tail(1)
    extra_cols = ["condition_id", *condition_cols, "output_num_frames", "output_first_frame",
                  "output_last_frame", "output_first_x", "output_first_y", "output_end_x", "output_end_y"]
    existing_extra_cols = [c for c in extra_cols if c in last_rows.columns]
    last_info = last_rows[TRAJ_COLS + existing_extra_cols].drop_duplicates(TRAJ_COLS)
    traj_summary = traj_summary.merge(last_info, on=TRAJ_COLS, how="left")
    return traj_summary


def save_single_histogram(task):
    cid, row, sub, out_plot_dir = task
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()
    for ax, c in zip(axes, CONTROL_COLS):
        if c not in sub.columns:
            ax.axis("off")
            continue
        vals = pd.to_numeric(sub[c], errors="coerce").dropna()
        if len(vals) == 0:
            ax.set_title(f"{c}: no data")
            continue
        ax.hist(vals, bins=30)
        ax.set_title(f"{c}, n={len(vals)}")
        ax.set_xlabel(c)
        ax.set_ylabel("count")
    title = f"condition_id={cid}, n_trajectories={int(row['n_trajectories'])}, n_rows={int(row['n_rows'])}"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    out_path = out_plot_dir / f"condition_{cid}_control_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_random_plots(df_unit, summary, out_plot_dir, max_plots=100,
                      min_trajectories_for_plot=5, random_seed=42):
    if max_plots <= 0:
        return
    out_plot_dir.mkdir(parents=True, exist_ok=True)

    summary_sorted = summary.sort_values("n_trajectories", ascending=False).copy()
    eligible = summary_sorted[summary_sorted["n_trajectories"] >= min_trajectories_for_plot]
    if len(eligible) == 0:
        eligible = summary_sorted.head(max_plots)
    else:
        eligible = eligible.head(max_plots)

    print(f"[info] Generating {len(eligible)} histogram plots in parallel...")

    tasks = []
    for _, row in eligible.iterrows():
        cid = int(row["condition_id"])
        sub = df_unit[df_unit["condition_id"] == cid].copy()
        tasks.append((cid, row, sub, out_plot_dir))

    if len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(8, len(tasks))) as executor:
            list(executor.map(save_single_histogram, tasks))
    else:
        for t in tasks:
            save_single_histogram(t)

    print(f"[saved] random plots under: {out_plot_dir}")


def generate_animated_gif(task):
    """한 GIF 안에 있는 서로 다른 trajectory(객체)마다 **완전히 다른 색상** + Legend 표시"""
    cid, row, sampled_traj_ids, sub_traj, condition_cols, out_plot_dir, max_trajectories_per_plot = task

    n_traj_plot = min(max_trajectories_per_plot, len(sampled_traj_ids))
    fig, ax = plt.subplots(figsize=(14, 9))

    # 서로 다른 trajectory마다 명확히 구분되는 색상 (tab20 colormap)
    colors = plt.cm.tab20(np.linspace(0, 1, n_traj_plot))

    lines = []
    start_points = []
    end_points = []

    for i in range(n_traj_plot):
        color = colors[i]
        # Line
        line, = ax.plot([], [], color=color, alpha=0.85, linewidth=2.0, label=f'Traj {i+1}')
        lines.append(line)

        # Start marker (고정 초록색)
        start = ax.scatter([], [], color='lime', s=70, marker='o', edgecolors='black', zorder=7)
        start_points.append(start)

        # End marker (trajectory와 동일한 색상)
        end = ax.scatter([], [], color=color, s=70, marker='s', edgecolors='black', zorder=7)
        end_points.append(end)

    ax.set_xlim(0, 120)
    ax.set_ylim(-5, 60)
    ax.set_xlabel("Field X (yards)", fontsize=12)
    ax.set_ylabel("Field Y (yards)", fontsize=12)
    ax.grid(True, alpha=0.4)
    ax.set_aspect("equal", adjustable="box")

    # Condition 정보
    cond_parts = []
    for c in condition_cols[:6]:
        if c in row and pd.notna(row[c]):
            val = row[c]
            val_str = f"{val:.0f}" if isinstance(val, float) and "_bin" in str(c) else str(val)
            cond_parts.append(f"{c}={val_str}")
    cond_label = " | ".join(cond_parts) if cond_parts else "N/A"

    ax.set_title(f"Conditional Trajectories Animation — condition_id={cid}\n{cond_label}", fontsize=13)

    # Legend (각 trajectory 색상과 함께 표시)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95, title="Trajectories")

    frames = 45

    def animate(frame_idx):
        progress = min(1.0, (frame_idx + 1) / frames)
        for i, tid in enumerate(sampled_traj_ids):
            traj_data = sub_traj[sub_traj["_trajectory_id"] == tid]
            if len(traj_data) < 2:
                continue
            max_idx = max(1, int(len(traj_data) * progress))
            x_data = traj_data["x"].iloc[:max_idx].values
            y_data = traj_data["y"].iloc[:max_idx].values

            lines[i].set_data(x_data, y_data)

            # End point 업데이트
            end_points[i].set_offsets([[x_data[-1], y_data[-1]]])

            # Start point (처음에만 한 번)
            if frame_idx == 0:
                start_points[i].set_offsets([[x_data[0], y_data[0]]])

        return lines + end_points + start_points

    ani = FuncAnimation(fig, animate, frames=frames, interval=35, blit=False)

    out_path = out_plot_dir / f"condition_{cid}_trajectories.gif"
    writer = PillowWriter(fps=18, metadata=dict(artist='NFL Big Data Bowl 2026'))
    ani.save(str(out_path), writer=writer)
    plt.close(fig)
    print(f"[GIF saved] condition_{cid}_trajectories.gif ({n_traj_plot} trajectories)")
    return cid


def make_conditional_trajectory_plots(all_rows, df_unit, summary, condition_cols,
                                      out_plot_dir, max_plots=20,
                                      max_trajectories_per_plot=10,
                                      min_trajectories_for_plot=8,
                                      random_seed=42):
    if max_plots <= 0:
        return
    out_plot_dir.mkdir(parents=True, exist_ok=True)

    traj_per_cond = df_unit.groupby("condition_id")["_trajectory_id"].apply(lambda x: x.unique().tolist()).to_dict()

    summary_sorted = summary.sort_values("n_trajectories", ascending=False).copy()
    eligible = summary_sorted[summary_sorted["n_trajectories"] >= min_trajectories_for_plot]
    if len(eligible) == 0:
        eligible = summary_sorted.head(max_plots)
    else:
        eligible = eligible.head(max_plots)

    print(f"[info] Generating {len(eligible)} animated GIF trajectory visualizations in parallel...")

    rng = np.random.default_rng(random_seed)
    tasks = []

    for _, row in eligible.iterrows():
        cid = int(row["condition_id"])
        traj_ids = traj_per_cond.get(cid, [])
        if len(traj_ids) == 0:
            continue
        n_traj_plot = min(max_trajectories_per_plot, len(traj_ids))
        sampled_traj_ids = rng.choice(traj_ids, size=n_traj_plot, replace=False)
        sub_traj = all_rows[all_rows["_trajectory_id"].isin(sampled_traj_ids)].copy()
        if len(sub_traj) == 0:
            continue
        sub_traj = sub_traj.sort_values(TRAJ_COLS + ["frame_id"])
        tasks.append((cid, row, sampled_traj_ids, sub_traj, condition_cols, out_plot_dir, max_trajectories_per_plot))

    if len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(6, len(tasks))) as executor:
            list(executor.map(generate_animated_gif, tasks))
    else:
        for t in tasks:
            generate_animated_gif(t)

    print(f"[saved] {len(tasks)} animated GIFs under: {out_plot_dir}")


def process_input_pair(serial_p, input_wanted_cols, x_bin, y_bin, ball_bin, yardline_bin, age_bin, weight_bin):
    year = serial_p["year"]
    week = serial_p["week"]
    input_path = Path(serial_p["input_path"])
    output_path = Path(serial_p["output_path"]) if serial_p.get("output_path") else None

    print(f"[info] reading input (parallel): {input_path.name}")

    inp = read_csv_selected(input_path, input_wanted_cols)
    inp["source_year"] = year
    inp["source_week"] = week
    for col in ID_COLS + ["frame_id"]:
        if col in inp.columns:
            inp[col] = pd.to_numeric(inp[col], errors="coerce")
    inp = inp.dropna(subset=ID_COLS + ["frame_id"])
    inp = add_age_and_bins(inp, source_year=year, x_bin=x_bin, y_bin=y_bin,
                           ball_bin=ball_bin, yardline_bin=yardline_bin,
                           age_bin=age_bin, weight_bin=weight_bin)

    out_stats = summarize_output_file(output_path, year, week)
    if out_stats is not None:
        inp = inp.merge(out_stats, on=TRAJ_COLS, how="left")
    else:
        inp["output_num_frames"] = np.nan
    return inp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, default="./train")
    parser.add_argument("--out-dir", type=str, default="./conditional_control_distribution")
    parser.add_argument("--unit", type=str, default="last_frame", choices=["last_frame", "all_frames"])
    parser.add_argument("--require-output", action="store_true")
    parser.add_argument("--x-bin", type=float, default=5.0)
    parser.add_argument("--y-bin", type=float, default=2.0)
    parser.add_argument("--ball-bin", type=float, default=5.0)
    parser.add_argument("--yardline-bin", type=float, default=5.0)
    parser.add_argument("--age-bin", type=float, default=3.0)
    parser.add_argument("--weight-bin", type=float, default=20.0)
    parser.add_argument("--max-plots", type=int, default=80)
    parser.add_argument("--min-trajectories-for-plot", type=int, default=5)
    parser.add_argument("--max-traj-plots", type=int, default=20)
    parser.add_argument("--max-trajectories-per-plot", type=int, default=10)
    parser.add_argument("--min-trajectories-for-traj-plot", type=int, default=8)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None)

    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = discover_input_output_pairs(train_dir)
    if args.max_files is not None:
        pairs = pairs[:args.max_files]

    if len(pairs) == 0:
        raise RuntimeError(f"No input_YYYY_wXX.csv files found in {train_dir}")

    print(f"[info] number of matched input files: {len(pairs)}")

    input_wanted_cols = list(set(
        ID_COLS + ["frame_id"] + STATE_CATEGORICAL_COLS + EXTERNAL_CATEGORICAL_COLS +
        STATE_NUMERIC_RAW_COLS + EXTERNAL_NUMERIC_RAW_COLS + CONTROL_COLS
    ))

    if args.max_workers is None:
        args.max_workers = multiprocessing.cpu_count()
    max_workers = max(1, min(args.max_workers, len(pairs)))
    print(f"[info] using {max_workers} workers for parallel file processing")

    serial_pairs = []
    for p in pairs:
        serial_p = p.copy()
        serial_p["input_path"] = str(p["input_path"])
        serial_p["output_path"] = str(p["output_path"]) if p["output_path"] else None
        serial_pairs.append(serial_p)

    if max_workers > 1 and len(pairs) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            worker_func = partial(process_input_pair,
                                  input_wanted_cols=input_wanted_cols,
                                  x_bin=args.x_bin, y_bin=args.y_bin,
                                  ball_bin=args.ball_bin, yardline_bin=args.yardline_bin,
                                  age_bin=args.age_bin, weight_bin=args.weight_bin)
            all_inputs = list(executor.map(worker_func, serial_pairs))
    else:
        all_inputs = []
        for p in pairs:
            serial_p = {"year": p["year"], "week": p["week"],
                        "input_path": str(p["input_path"]),
                        "output_path": str(p["output_path"]) if p["output_path"] else None}
            all_inputs.append(process_input_pair(serial_p, input_wanted_cols, args.x_bin,
                                                 args.y_bin, args.ball_bin, args.yardline_bin,
                                                 args.age_bin, args.weight_bin))

    all_rows = pd.concat(all_inputs, ignore_index=True)
    print(f"[info] total input rows: {len(all_rows):,}")

    if args.require_output:
        before = len(all_rows)
        all_rows = all_rows[all_rows["output_num_frames"].notna()].copy()
        print(f"[info] require-output filter: {before:,} -> {len(all_rows):,}")

    condition_cols = [c for c in [
        "player_position", "player_side", "player_role", "play_direction",
        "player_weight_bin", "player_age_bin", "x_bin", "y_bin",
        "ball_land_x_bin", "ball_land_y_bin", "absolute_yardline_bin"
    ] if c in all_rows.columns]

    all_rows["_trajectory_id"] = (
        all_rows["source_year"].astype(str) + "_w" + all_rows["source_week"].astype(str) +
        "_g" + all_rows["game_id"].astype(str) + "_p" + all_rows["play_id"].astype(str) +
        "_n" + all_rows["nfl_id"].astype(str)
    )

    all_rows = build_condition_id(all_rows, condition_cols)

    if args.unit == "last_frame":
        df_unit = all_rows.sort_values(TRAJ_COLS + ["frame_id"]).groupby(TRAJ_COLS, dropna=False, sort=False).tail(1).copy()
    else:
        df_unit = all_rows.copy()

    print(f"[info] unit = {args.unit}")
    print(f"[info] rows used for conditional distribution: {len(df_unit):,}")
    print(f"[info] unique trajectories: {df_unit['_trajectory_id'].nunique():,}")

    conditional_summary = summarize_conditional_distribution(df_unit, condition_cols)
    conditional_summary.to_csv(out_dir / "conditional_control_distribution_summary.csv", index=False)

    membership = df_unit.groupby(["condition_id"] + TRAJ_COLS, dropna=False, sort=False).size().reset_index(name="n_rows_in_condition")
    membership.to_csv(out_dir / "conditional_trajectory_membership.csv", index=False)

    trajectory_summary = summarize_trajectory_distribution(all_rows, condition_cols)
    trajectory_summary.to_csv(out_dir / "trajectory_control_distribution_summary.csv", index=False)

    make_random_plots(df_unit, conditional_summary, out_dir / "random_condition_plots",
                      max_plots=args.max_plots,
                      min_trajectories_for_plot=args.min_trajectories_for_plot,
                      random_seed=args.random_seed)

    make_conditional_trajectory_plots(
        all_rows=all_rows, df_unit=df_unit, summary=conditional_summary,
        condition_cols=condition_cols,
        out_plot_dir=out_dir / "conditional_trajectory_plots",
        max_plots=args.max_traj_plots,
        max_trajectories_per_plot=args.max_trajectories_per_plot,
        min_trajectories_for_plot=args.min_trajectories_for_traj_plot,
        random_seed=args.random_seed
    )

    config = {k: getattr(args, k) for k in vars(args)}
    with open(out_dir / "run_config.txt", "w", encoding="utf-8") as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")
    print(f"[saved] {out_dir / 'run_config.txt'}")


if __name__ == "__main__":
    main()