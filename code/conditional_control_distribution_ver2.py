#!/usr/bin/env python3
# conditional_control_distribution.py

import os
import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ID_COLS = ["game_id", "play_id", "nfl_id"]
TRAJ_COLS = ["source_year", "source_week", "game_id", "play_id", "nfl_id"]

CONTROL_COLS = ["s", "a", "o", "dir"]

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


def discover_input_output_pairs(train_dir: Path):
    """
    input_2023_w01.csv -> output_2023_w01.csv 자동 매칭
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

    # numeric 변환
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

    # birth_date -> age
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

    # categorical 결측 처리
    categorical_cols = STATE_CATEGORICAL_COLS + EXTERNAL_CATEGORICAL_COLS
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("MISSING").astype(str)

    return df


def summarize_output_file(output_path: Path, source_year: int, source_week: int):
    """
    output csv는 미래 위치 x, y만 있으므로 control variable 계산에는 직접 쓰지 않음.
    대신 input trajectory와 매칭해서 future target 존재 여부 및 landing/future endpoint 정보를 저장.
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
    """
    같은 condition combination에 동일한 condition_id 부여.
    """
    df["condition_id"] = (
        df.groupby(condition_cols, dropna=False, sort=False)
        .ngroup()
        .astype(int)
        + 1
    )
    return df


def summarize_conditional_distribution(df_unit, condition_cols):
    """
    condition_id별 control variable 조건부 분포 요약.
    df_unit의 관측 단위는 --unit에 의해 결정됨.
    기본값 last_frame이면 trajectory당 1개 row.
    """

    for col in CONTROL_COLS:
        if col in df_unit.columns:
            df_unit[col] = pd.to_numeric(df_unit[col], errors="coerce")

    agg = {
        "n_rows": ("condition_id", "size"),
        "n_trajectories": ("_trajectory_id", "nunique"),
        "n_games": ("game_id", "nunique"),
        "n_plays": ("play_id", "nunique"),
        "n_players": ("nfl_id", "nunique"),
    }

    for c in CONTROL_COLS:
        if c not in df_unit.columns:
            continue

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

    summary = (
        df_unit.groupby("condition_id", dropna=False, sort=False)
        .agg(**agg)
        .reset_index()
    )

    condition_values = (
        df_unit.groupby("condition_id", dropna=False, sort=False)[condition_cols]
        .first()
        .reset_index()
    )

    summary = condition_values.merge(summary, on="condition_id", how="left")

    return summary


def summarize_trajectory_distribution(all_rows, condition_cols):
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
        "condition_id",
        *condition_cols,
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


def make_random_plots(
    df_unit,
    summary,
    out_plot_dir,
    max_plots=100,
    min_trajectories_for_plot=20,
    random_seed=42,
):
    """
    condition_id를 랜덤 샘플링하여 control variables histogram 저장.
    """
    if max_plots <= 0:
        return

    out_plot_dir.mkdir(parents=True, exist_ok=True)

    eligible = summary[summary["n_trajectories"] >= min_trajectories_for_plot].copy()

    if len(eligible) == 0:
        print("[warning] no condition has enough trajectories for plotting.")
        return

    n_sample = min(max_plots, len(eligible))
    sampled = eligible.sample(n=n_sample, random_state=random_seed)

    for _, row in sampled.iterrows():
        cid = int(row["condition_id"])
        sub = df_unit[df_unit["condition_id"] == cid]

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

        title = (
            f"condition_id={cid}, "
            f"n_trajectories={int(row['n_trajectories'])}, "
            f"n_rows={int(row['n_rows'])}"
        )
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()

        out_path = out_plot_dir / f"condition_{cid}_control_distribution.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def make_conditional_trajectory_plots(
    all_rows: pd.DataFrame,
    df_unit: pd.DataFrame,
    summary: pd.DataFrame,
    condition_cols: list,
    out_plot_dir: Path,
    max_plots: int = 30,
    max_trajectories_per_plot: int = 15,
    min_trajectories_for_plot: int = 30,
    random_seed: int = 42,
):
    """
    특정 state 조건(condition_id)에 해당하는 여러 trajectory를 한 plot에 overlay하여 시각화.
    조건부 trajectory 분포(위치 경로의 다양성)를 직관적으로 보여줌.
    df_unit의 condition_id (last_frame 또는 all_frames 기준)를 사용해 trajectory를 매핑.
    """
    if max_plots <= 0:
        return

    out_plot_dir.mkdir(parents=True, exist_ok=True)

    # condition_id별 해당 trajectory 목록 (unique 처리)
    traj_per_cond = (
        df_unit.groupby("condition_id")["_trajectory_id"]
        .apply(lambda x: x.unique().tolist())
        .to_dict()
    )

    eligible = summary[summary["n_trajectories"] >= min_trajectories_for_plot].copy()

    if len(eligible) == 0:
        print("[warning] no condition has enough trajectories for trajectory plotting.")
        return

    n_sample = min(max_plots, len(eligible))
    rng = np.random.default_rng(random_seed)
    sampled = eligible.sample(n=n_sample, random_state=random_seed)

    print(f"[info] generating {n_sample} conditional trajectory visualization plots...")

    for _, row in sampled.iterrows():
        cid = int(row["condition_id"])
        traj_ids = traj_per_cond.get(cid, [])
        if len(traj_ids) < min_trajectories_for_plot:
            continue

        # Sample trajectories for this condition
        n_traj_plot = min(max_trajectories_per_plot, len(traj_ids))
        sampled_traj_ids = rng.choice(traj_ids, size=n_traj_plot, replace=False)

        # Extract FULL trajectory data (x, y over all input frames) for sampled trajs
        sub_traj = all_rows[all_rows["_trajectory_id"].isin(sampled_traj_ids)].copy()
        if len(sub_traj) == 0:
            continue
        sub_traj = sub_traj.sort_values(TRAJ_COLS + ["frame_id"])

        fig, ax = plt.subplots(figsize=(12, 8))

        colors = plt.cm.tab20(np.linspace(0, 1, n_traj_plot))

        for i, tid in enumerate(sampled_traj_ids):
            traj_data = sub_traj[sub_traj["_trajectory_id"] == tid]
            if len(traj_data) < 2:
                continue
            color = colors[i]
            ax.plot(
                traj_data["x"],
                traj_data["y"],
                color=color,
                alpha=0.65,
                linewidth=1.5,
                label=f"traj_{i+1}",
            )
            # Start (green circle) and end (red square) markers
            ax.scatter(
                traj_data["x"].iloc[0],
                traj_data["y"].iloc[0],
                color="green",
                s=50,
                marker="o",
                edgecolors="black",
                zorder=6,
            )
            ax.scatter(
                traj_data["x"].iloc[-1],
                traj_data["y"].iloc[-1],
                color="red",
                s=50,
                marker="s",
                edgecolors="black",
                zorder=6,
            )

        # NFL field approximate bounds
        ax.set_xlim(0, 120)
        ax.set_ylim(-5, 60)
        ax.set_xlabel("Field X (yards)")
        ax.set_ylabel("Field Y (yards)")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

        # Condition summary for title
        cond_parts = []
        for c in condition_cols:
            if c in row and pd.notna(row[c]):
                val = row[c]
                if isinstance(val, float) and "_bin" in c:
                    val_str = f"{val:.0f}"
                else:
                    val_str = str(val)
                cond_parts.append(f"{c}={val_str}")
            if len(cond_parts) >= 5:
                break
        cond_label = " | ".join(cond_parts) if cond_parts else "N/A"

        ax.set_title(
            f"Conditional Trajectories — condition_id={cid}\n"
            f"{n_traj_plot} sampled trajectories (total {len(traj_ids)} available)\n"
            f"{cond_label}",
            fontsize=11,
        )

        if n_traj_plot <= 10:
            ax.legend(loc="upper right", fontsize=8)

        fig.tight_layout()

        out_path = out_plot_dir / f"condition_{cid}_trajectories.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    print(f"[saved] {n_sample} conditional trajectory plots under: {out_plot_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-dir", type=str, default="./train")
    parser.add_argument("--out-dir", type=str, default="./conditional_control_distribution")

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

    parser.add_argument("--x-bin", type=float, default=5.0)
    parser.add_argument("--y-bin", type=float, default=2.0)
    parser.add_argument("--ball-bin", type=float, default=5.0)
    parser.add_argument("--yardline-bin", type=float, default=5.0)
    parser.add_argument("--age-bin", type=float, default=3.0)
    parser.add_argument("--weight-bin", type=float, default=20.0)

    parser.add_argument("--max-plots", type=int, default=100)
    parser.add_argument("--min-trajectories-for-plot", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=42)

    # ===== NEW: 조건부 trajectory 분포 저장 + 시각화 옵션 =====
    parser.add_argument(
        "--max-traj-plots",
        type=int,
        default=30,
        help="조건부 trajectory 시각화에 사용할 condition 개수 (0이면 스킵)",
    )
    parser.add_argument(
        "--max-trajectories-per-plot",
        type=int,
        default=15,
        help="한 plot당 overlay할 trajectory 최대 개수",
    )
    parser.add_argument(
        "--min-trajectories-for-traj-plot",
        type=int,
        default=30,
        help="trajectory plot을 생성할 최소 n_trajectories 조건",
    )
    # =======================================================

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="디버깅용. 앞에서부터 몇 개의 input/output pair만 사용할지 지정.",
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
            inp = inp.merge(
                out_stats,
                on=TRAJ_COLS,
                how="left",
            )
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

    # condition variables
    condition_cols = [
        "player_position",
        "player_side",
        "player_role",
        "play_direction",
        "player_weight_bin",
        "player_age_bin",
        "x_bin",
        "y_bin",
        "ball_land_x_bin",
        "ball_land_y_bin",
        "absolute_yardline_bin",
    ]

    condition_cols = [c for c in condition_cols if c in all_rows.columns]

    # trajectory id
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

    # condition id는 전체 row 기준으로 먼저 부여
    all_rows = build_condition_id(all_rows, condition_cols)

    # 분석 단위 선택
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

    # 1) condition별 control variable 조건부 분포
    conditional_summary = summarize_conditional_distribution(df_unit, condition_cols)

    conditional_summary_path = out_dir / "conditional_control_distribution_summary.csv"
    conditional_summary.to_csv(conditional_summary_path, index=False)

    print(f"[saved] {conditional_summary_path}")

    # 2) condition_id별 어떤 play_id/nfl_id가 들어갔는지
    membership = (
        df_unit.groupby(["condition_id"] + TRAJ_COLS, dropna=False, sort=False)
        .size()
        .reset_index(name="n_rows_in_condition")
    )

    membership_path = out_dir / "conditional_trajectory_membership.csv"
    membership.to_csv(membership_path, index=False)

    print(f"[saved] {membership_path}")

    # 3) trajectory별 control variable의 시간축 분포
    trajectory_summary = summarize_trajectory_distribution(all_rows, condition_cols)

    trajectory_summary_path = out_dir / "trajectory_control_distribution_summary.csv"
    trajectory_summary.to_csv(trajectory_summary_path, index=False)

    print(f"[saved] {trajectory_summary_path}")

    # 4) control variable histogram (기존)
    plot_dir = out_dir / "random_condition_plots"

    make_random_plots(
        df_unit=df_unit,
        summary=conditional_summary,
        out_plot_dir=plot_dir,
        max_plots=args.max_plots,
        min_trajectories_for_plot=args.min_trajectories_for_plot,
        random_seed=args.random_seed,
    )

    print(f"[saved] random plots under: {plot_dir}")

    # ===== NEW: 5) 조건부 trajectory 분포 시각화 (여러 trajectory를 한 plot에 overlay) =====
    traj_plot_dir = out_dir / "conditional_trajectory_plots"
    make_conditional_trajectory_plots(
        all_rows=all_rows,
        df_unit=df_unit,
        summary=conditional_summary,
        condition_cols=condition_cols,
        out_plot_dir=traj_plot_dir,
        max_plots=args.max_traj_plots,
        max_trajectories_per_plot=args.max_trajectories_per_plot,
        min_trajectories_for_plot=args.min_trajectories_for_traj_plot,
        random_seed=args.random_seed,
    )
    # ===================================================================================

    # 6) 설정 저장
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
        "condition_cols": condition_cols,
        "control_cols": CONTROL_COLS,
        "max_traj_plots": args.max_traj_plots,
        "max_trajectories_per_plot": args.max_trajectories_per_plot,
        "min_trajectories_for_traj_plot": args.min_trajectories_for_traj_plot,
    }

    config_path = out_dir / "run_config.txt"
    with open(config_path, "w", encoding="utf-8") as f:
        for k, v in config.items():
            f.write(f"{k}: {v}\n")

    print(f"[saved] {config_path}")


if __name__ == "__main__":
    main()