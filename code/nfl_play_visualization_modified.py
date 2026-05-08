"""
NFL player-location GIF generator
=================================

This script takes two CSV files:

1) input CSV, usually ending with *_input.csv
   Required columns:
   game_id, play_id, nfl_id, frame_id, x, y, ball_land_x, ball_land_y
   Optional columns are preserved when present, e.g. player_name,
   player_to_predict, player_position, player_side, player_role, etc.

2) output CSV
   Required columns:
   game_id, play_id, nfl_id, frame_id, x, y

For each (game_id, play_id), the script concatenates player locations in
chronological order: input frames first, then output frames. The join key for
adding metadata from input to output is (game_id, play_id, nfl_id). frame_id is
used as the within-phase timestamp.

Example:
    python3 nfl_play_visualization_modified.py \
        --input-csv ./csv/train_input.csv \
        --output-csv ./csv/train_output.csv \
        --out-gif ./plots_nfl_play_gif \
        --out-csv ./csv_nfl_play_combined \
        --fps 10

Generate only selected plays:
    python3 nfl_play_visualization_modified.py \
        --input-csv ./csv/train_input.csv \
        --output-csv ./csv/train_output.csv \
        --selected-keys "2022090800:56,2022090800:80"
"""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# =========================================================
# 기본 경로: argparse로 덮어쓸 수 있음
# =========================================================
INPUT_CSV_PATH = "./csv/train_input.csv"
OUTPUT_CSV_PATH = "./csv/train_output.csv"
OUTPUT_DIR_GIF = "./plots_nfl_play_gif"
OUTPUT_DIR_CSV = "./csv_nfl_play_combined"

# =========================================================
# 사용자가 만들고 싶은 GIF 대상 지정
# =========================================================
# None 또는 빈 리스트면 전체 생성
# 형식: [(game_id, play_id), ...]
SELECTED_KEYS = [
    # 예시:
    # ("2022090800", "56"),
]

# True면 SELECTED_KEYS만 생성
# False면 전체 생성
USE_SELECTED_KEYS_ONLY = False

# GIF 표시 옵션
DEFAULT_FPS = 10
DEFAULT_MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)
USE_NFL_FIELD_LIMITS = True
SHOW_TRAILS = True
SHOW_PLAYER_LABELS = False
LABEL_ONLY_TARGETS = True

# NFL field coordinates in Big Data Bowl style data
FIELD_X_MIN = 0.0
FIELD_X_MAX = 120.0
FIELD_Y_MIN = 0.0
FIELD_Y_MAX = 53.3

KEY_COLS = ["game_id", "play_id", "nfl_id"]
PLAY_COLS = ["game_id", "play_id"]

INPUT_REQUIRED_COLS = [
    "game_id",
    "play_id",
    "nfl_id",
    "frame_id",
    "x",
    "y",
    "ball_land_x",
    "ball_land_y",
]

OUTPUT_REQUIRED_COLS = ["game_id", "play_id", "nfl_id", "frame_id", "x", "y"]

# output CSV에는 없는 경우가 많으므로 input에서 metadata를 가져와 붙임
OPTIONAL_METADATA_COLS = [
    "player_to_predict",
    "play_direction",
    "absolute_yardline_number",
    "player_name",
    "player_height",
    "player_weight",
    "player_birth_date",
    "player_position",
    "player_side",
    "player_role",
    "s",
    "a",
    "dir",
    "o",
    "num_frames_output",
    "ball_land_x",
    "ball_land_y",
]


def sanitize_filename(value):
    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "-")
    )


def normalize_id_value(value):
    """Make ID columns comparable across files.

    pandas sometimes reads integer IDs as floats when missing values are present,
    so this removes trailing '.0'.
    """
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def normalize_key_tuple(key_tuple):
    return tuple(normalize_id_value(x) for x in key_tuple)


def parse_selected_keys(text):
    """Parse 'game_id:play_id,game_id:play_id' into normalized tuples."""
    if not text:
        return []

    selected = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "--selected-keys 형식은 'game_id:play_id,game_id:play_id' 입니다. "
                f"잘못된 항목: {item}"
            )
        game_id, play_id = item.split(":", 1)
        selected.append(normalize_key_tuple((game_id, play_id)))
    return selected


def read_csv_auto(path):
    """Read comma CSV, with a fallback for tab-separated files."""
    df = pd.read_csv(path, low_memory=False)
    if len(df.columns) == 1 and "\t" in str(df.columns[0]):
        df = pd.read_csv(path, sep="\t", low_memory=False)
    return df


def require_columns(df, required_cols, label):
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{label} 파일에 다음 열이 없습니다: {missing_cols}")


def normalize_key_columns(df):
    df = df.copy()
    for col in KEY_COLS:
        df[col] = df[col].map(normalize_id_value)
    return df


def to_bool_series(series):
    """Convert player_to_predict-like values to bool."""
    if series is None:
        return pd.Series(False, index=[])

    true_values = {"1", "true", "t", "yes", "y"}
    false_values = {"0", "false", "f", "no", "n", ""}

    def _one(v):
        if pd.isna(v):
            return False
        text = str(v).strip().lower()
        if text in true_values:
            return True
        if text in false_values:
            return False
        return bool(v)

    return series.map(_one)


def build_combined_dataframe(input_csv, output_csv):
    input_df = read_csv_auto(input_csv)
    output_df = read_csv_auto(output_csv)

    require_columns(input_df, INPUT_REQUIRED_COLS, "input")
    require_columns(output_df, OUTPUT_REQUIRED_COLS, "output")

    input_df = normalize_key_columns(input_df)
    output_df = normalize_key_columns(output_df)

    # 숫자형 열 정리
    for df in (input_df, output_df):
        for col in ["frame_id", "x", "y"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["ball_land_x", "ball_land_y"]:
        input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    input_df = input_df.dropna(subset=KEY_COLS + ["frame_id", "x", "y"]).copy()
    output_df = output_df.dropna(subset=KEY_COLS + ["frame_id", "x", "y"]).copy()

    # output에는 player_name, player_side, ball_land_x/y 등이 없을 수 있으므로
    # input의 선수별 metadata를 붙인다. frame_id는 timestamp이므로 merge key에 넣지 않는다.
    metadata_cols = [col for col in OPTIONAL_METADATA_COLS if col in input_df.columns]
    metadata_cols = [col for col in metadata_cols if col not in {"x", "y", "frame_id"}]

    metadata = (
        input_df.sort_values(KEY_COLS + ["frame_id"])
        .drop_duplicates(KEY_COLS, keep="first")[KEY_COLS + metadata_cols]
        .copy()
    )

    output_enriched = output_df.merge(metadata, on=KEY_COLS, how="left")

    # input/output 공통 스키마 구성
    base_cols = KEY_COLS + ["frame_id", "x", "y"] + metadata_cols
    base_cols = list(dict.fromkeys(base_cols))

    input_part = input_df[[col for col in base_cols if col in input_df.columns]].copy()
    output_part = output_enriched[[col for col in base_cols if col in output_enriched.columns]].copy()

    for col in base_cols:
        if col not in input_part.columns:
            input_part[col] = pd.NA
        if col not in output_part.columns:
            output_part[col] = pd.NA

    input_part = input_part[base_cols]
    output_part = output_part[base_cols]

    input_part["phase"] = "input"
    input_part["phase_order"] = 0
    output_part["phase"] = "output"
    output_part["phase_order"] = 1

    combined = pd.concat([input_part, output_part], ignore_index=True)

    if "player_to_predict" in combined.columns:
        combined["is_target"] = to_bool_series(combined["player_to_predict"])
    else:
        combined["is_target"] = False

    # input -> output 순서가 보장되도록 phase_order를 먼저 사용한다.
    # 각 phase 안에서는 frame_id를 timestamp로 사용한다.
    combined = combined.sort_values(
        PLAY_COLS + ["phase_order", "frame_id", "nfl_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    return combined


def add_animation_frame_index(play_df):
    frame_table = (
        play_df[["phase", "phase_order", "frame_id"]]
        .drop_duplicates()
        .sort_values(["phase_order", "frame_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    frame_table["animation_frame_index"] = np.arange(len(frame_table), dtype=int)
    out = play_df.merge(frame_table, on=["phase", "phase_order", "frame_id"], how="left")
    out = out.sort_values(["animation_frame_index", "nfl_id"], kind="mergesort").reset_index(drop=True)
    return out, frame_table


def draw_nfl_field(ax):
    ax.set_xlim(FIELD_X_MIN, FIELD_X_MAX)
    ax.set_ylim(FIELD_Y_MIN, FIELD_Y_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # field boundary
    ax.plot(
        [FIELD_X_MIN, FIELD_X_MAX, FIELD_X_MAX, FIELD_X_MIN, FIELD_X_MIN],
        [FIELD_Y_MIN, FIELD_Y_MIN, FIELD_Y_MAX, FIELD_Y_MAX, FIELD_Y_MIN],
        linewidth=1,
    )

    # yard lines
    for x in range(10, 111, 10):
        ax.axvline(x, linewidth=0.5, alpha=0.35)

    # hash-like horizontal guide lines
    ax.axhline(FIELD_Y_MAX / 2, linewidth=0.5, alpha=0.25)
    ax.grid(False)


def set_dynamic_axis(ax, play_df):
    x = play_df["x"].to_numpy(dtype=float)
    y = play_df["y"].to_numpy(dtype=float)

    ball_x = pd.to_numeric(play_df.get("ball_land_x", pd.Series(dtype=float)), errors="coerce")
    ball_y = pd.to_numeric(play_df.get("ball_land_y", pd.Series(dtype=float)), errors="coerce")

    x_values = np.concatenate([x, ball_x.dropna().to_numpy(dtype=float)]) if len(ball_x) else x
    y_values = np.concatenate([y, ball_y.dropna().to_numpy(dtype=float)]) if len(ball_y) else y

    margin_x = max((np.nanmax(x_values) - np.nanmin(x_values)) * 0.08, 2.0)
    margin_y = max((np.nanmax(y_values) - np.nanmin(y_values)) * 0.08, 2.0)

    ax.set_xlim(np.nanmin(x_values) - margin_x, np.nanmax(x_values) + margin_x)
    ax.set_ylim(np.nanmin(y_values) - margin_y, np.nanmax(y_values) + margin_y)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)


def get_ball_land_point(play_df):
    if "ball_land_x" not in play_df.columns or "ball_land_y" not in play_df.columns:
        return None

    ball_df = play_df[["ball_land_x", "ball_land_y"]].dropna().drop_duplicates()
    if len(ball_df) == 0:
        return None

    first = ball_df.iloc[0]
    return float(first["ball_land_x"]), float(first["ball_land_y"])


def make_offsets(df):
    if df is None or len(df) == 0:
        return np.empty((0, 2))
    return df[["x", "y"]].to_numpy(dtype=float)


def make_label(row):
    name = row.get("player_name", None)
    nfl_id = row.get("nfl_id", "")
    position = row.get("player_position", "")

    if pd.notna(name) and str(name).strip():
        base = str(name).strip()
    else:
        base = str(nfl_id)

    if pd.notna(position) and str(position).strip():
        return f"{base} ({position})"
    return base


def process_group(args):
    (
        key,
        group,
        output_dir_gif,
        output_dir_csv,
        fps,
        use_nfl_field_limits,
        show_trails,
        show_player_labels,
        label_only_targets,
    ) = args

    game_id, play_id = key
    group = group.copy()
    group, frame_table = add_animation_frame_index(group)

    if len(group) == 0 or len(frame_table) == 0:
        return None

    safe_game_id = sanitize_filename(game_id)
    safe_play_id = sanitize_filename(play_id)
    base_name = f"game_{safe_game_id}_play_{safe_play_id}"

    out_csv = os.path.join(output_dir_csv, f"{base_name}.csv")
    group.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(12, 6.8))

    if use_nfl_field_limits:
        draw_nfl_field(ax)
    else:
        set_dynamic_axis(ax, group)

    ball_land = get_ball_land_point(group)
    if ball_land is not None:
        bx, by = ball_land
        ax.scatter([bx], [by], marker="X", s=140, label="ball land point", zorder=5)
        ax.axvline(bx, linestyle="--", linewidth=0.8, alpha=0.45)
        ax.axhline(by, linestyle="--", linewidth=0.8, alpha=0.45)
        ax.text(bx + 0.5, by + 0.5, "ball land", fontsize=8)

    title = f"game_id={game_id}, play_id={play_id}"
    ax.set_title(title)

    # 현재 frame의 점들
    scat_non_target = ax.scatter([], [], s=35, label="input/non-target", alpha=0.65, zorder=3)
    scat_target_input = ax.scatter([], [], s=50, label="input/target", alpha=0.95, zorder=4)
    scat_output = ax.scatter([], [], s=60, marker="s", label="output", alpha=0.95, zorder=4)

    # 선수별 trajectory line
    line_map = {}
    if show_trails:
        for nfl_id in sorted(group["nfl_id"].dropna().astype(str).unique()):
            line, = ax.plot([], [], linewidth=0.8, alpha=0.28, zorder=2)
            line_map[nfl_id] = line

    text = ax.text(
        0.01,
        0.99,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
    )

    label_texts = []
    ax.legend(loc="lower right", fontsize=8)

    def clear_labels():
        while label_texts:
            label_texts.pop().remove()

    def init():
        scat_non_target.set_offsets(np.empty((0, 2)))
        scat_target_input.set_offsets(np.empty((0, 2)))
        scat_output.set_offsets(np.empty((0, 2)))
        for line in line_map.values():
            line.set_data([], [])
        text.set_text("")
        clear_labels()
        return [scat_non_target, scat_target_input, scat_output, text, *line_map.values()]

    def update(frame_idx):
        clear_labels()

        current = group[group["animation_frame_index"] == frame_idx].copy()
        history = group[group["animation_frame_index"] <= frame_idx].copy()

        current_input = current[current["phase"] == "input"]
        current_output = current[current["phase"] == "output"]
        current_input_target = current_input[current_input["is_target"]]
        current_input_non_target = current_input[~current_input["is_target"]]

        scat_non_target.set_offsets(make_offsets(current_input_non_target))
        scat_target_input.set_offsets(make_offsets(current_input_target))
        scat_output.set_offsets(make_offsets(current_output))

        if show_trails:
            for nfl_id, line in line_map.items():
                player_hist = history[history["nfl_id"].astype(str) == str(nfl_id)]
                line.set_data(player_hist["x"].to_numpy(), player_hist["y"].to_numpy())

        frame_info = frame_table.iloc[frame_idx]
        phase = frame_info["phase"]
        frame_id = frame_info["frame_id"]

        n_current = len(current)
        n_output = len(current_output)
        n_target = int(current.get("is_target", pd.Series(dtype=bool)).sum())

        text.set_text(
            f"phase={phase} | frame_id={frame_id:g} | "
            f"animation frame={frame_idx + 1}/{len(frame_table)}\n"
            f"current players={n_current} | target rows={n_target} | output rows={n_output}"
        )

        if show_player_labels:
            label_source = current
            if label_only_targets and "is_target" in label_source.columns:
                label_source = label_source[label_source["is_target"]]

            for _, row in label_source.iterrows():
                label_texts.append(
                    ax.text(
                        row["x"] + 0.35,
                        row["y"] + 0.35,
                        make_label(row),
                        fontsize=6,
                        zorder=6,
                    )
                )

        return [scat_non_target, scat_target_input, scat_output, text, *line_map.values(), *label_texts]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frame_table),
        init_func=init,
        interval=int(1000 / max(fps, 1)),
        blit=False,
        repeat=False,
    )

    out_gif = os.path.join(output_dir_gif, f"{base_name}.gif")
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)

    return {
        "key": key,
        "gif": out_gif,
        "csv": out_csv,
        "n_rows": len(group),
        "n_frames": len(frame_table),
        "n_players": group["nfl_id"].nunique(),
    }


def build_tasks(
    combined,
    target_keys_df,
    output_dir_gif,
    output_dir_csv,
    fps,
    use_nfl_field_limits,
    show_trails,
    show_player_labels,
    label_only_targets,
):
    grouped = combined.groupby(PLAY_COLS, sort=False)

    group_map = {}
    for key, group in grouped:
        norm_key = normalize_key_tuple(key)
        group_map[norm_key] = group.copy()

    tasks = []
    for _, row in target_keys_df.iterrows():
        key = normalize_key_tuple((row["game_id"], row["play_id"]))
        if key in group_map:
            tasks.append(
                (
                    key,
                    group_map[key],
                    output_dir_gif,
                    output_dir_csv,
                    fps,
                    use_nfl_field_limits,
                    show_trails,
                    show_player_labels,
                    label_only_targets,
                )
            )

    return tasks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=str, default=INPUT_CSV_PATH)
    parser.add_argument("--output-csv", type=str, default=OUTPUT_CSV_PATH)
    parser.add_argument("--out-gif", type=str, default=OUTPUT_DIR_GIF)
    parser.add_argument("--out-csv", type=str, default=OUTPUT_DIR_CSV)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument(
        "--selected-keys",
        type=str,
        default="",
        help="'game_id:play_id,game_id:play_id' 형식. 지정하면 해당 play만 생성합니다.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="테스트용. 0이면 제한 없음, 양수면 앞에서부터 해당 개수의 play만 생성.",
    )
    parser.add_argument(
        "--dynamic-axis",
        action="store_true",
        help="NFL field 고정 축 대신 각 play의 x/y 범위에 맞춰 축을 동적으로 설정합니다.",
    )
    parser.add_argument("--no-trails", action="store_true", help="선수별 이동 궤적선을 표시하지 않습니다.")
    parser.add_argument("--show-labels", action="store_true", help="선수 이름 또는 nfl_id 라벨을 표시합니다.")
    parser.add_argument(
        "--label-all-players",
        action="store_true",
        help="--show-labels 사용 시 target 선수뿐 아니라 모든 선수 라벨을 표시합니다.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.out_gif, exist_ok=True)
    os.makedirs(args.out_csv, exist_ok=True)

    combined = build_combined_dataframe(args.input_csv, args.output_csv)

    if len(combined) == 0:
        print("input/output을 합친 결과가 비어 있습니다.")
        return

    unique_keys_df = (
        combined[PLAY_COLS]
        .drop_duplicates()
        .sort_values(PLAY_COLS, kind="mergesort")
        .reset_index(drop=True)
    )

    total_possible_gif_count = len(unique_keys_df)

    print("=" * 80)
    print(f"전체 생성 가능한 GIF 개수(= game_id, play_id 조합 수): {total_possible_gif_count}")
    print(f"합쳐진 전체 row 수: {len(combined)}")
    print("=" * 80)

    selected_from_cli = parse_selected_keys(args.selected_keys)
    selected_from_code = [normalize_key_tuple(k) for k in SELECTED_KEYS]
    selected_keys = selected_from_cli if selected_from_cli else selected_from_code

    use_selected = bool(selected_keys) or USE_SELECTED_KEYS_ONLY

    if use_selected:
        selected_key_set = set(selected_keys)
        unique_keys_df["_norm_key"] = list(
            zip(unique_keys_df["game_id"].map(normalize_id_value), unique_keys_df["play_id"].map(normalize_id_value))
        )

        filtered_keys_df = unique_keys_df[unique_keys_df["_norm_key"].isin(selected_key_set)].copy()
        not_found_keys = selected_key_set - set(filtered_keys_df["_norm_key"].tolist())

        print(f"사용자가 요청한 play 개수: {len(selected_key_set)}")
        print(f"실제로 데이터에 존재하는 play 개수: {len(filtered_keys_df)}")

        if not_found_keys:
            print("데이터에 없는 play:")
            for k in sorted(not_found_keys):
                print(f"  {k}")

        target_keys_df = filtered_keys_df[PLAY_COLS].copy()
    else:
        target_keys_df = unique_keys_df[PLAY_COLS].copy()
        print("전체 play에 대해 GIF/CSV를 생성합니다.")

    if args.limit and args.limit > 0:
        target_keys_df = target_keys_df.head(args.limit).copy()
        print(f"테스트용 limit 적용: 앞에서부터 {len(target_keys_df)}개 play만 생성합니다.")

    selected_gif_count = len(target_keys_df)
    print(f"이번 실행에서 생성 대상인 GIF 개수: {selected_gif_count}")

    if selected_gif_count == 0:
        print("생성할 대상이 없습니다.")
        return

    tasks = build_tasks(
        combined=combined,
        target_keys_df=target_keys_df,
        output_dir_gif=args.out_gif,
        output_dir_csv=args.out_csv,
        fps=args.fps,
        use_nfl_field_limits=not args.dynamic_axis,
        show_trails=not args.no_trails,
        show_player_labels=args.show_labels,
        label_only_targets=not args.label_all_players,
    )

    print(f"실제 작업(task) 개수: {len(tasks)}")

    max_workers = max(1, int(args.max_workers))
    max_workers = min(max_workers, len(tasks))
    print(f"병렬 프로세스 수: {max_workers}")

    if max_workers == 1:
        results = [process_group(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_group, tasks))

    results = [r for r in results if r is not None]

    print("=" * 80)
    print(f"완료: {len(results)}개 play 처리 완료")
    print(f"GIF 저장 폴더: {args.out_gif}")
    print(f"CSV 저장 폴더: {args.out_csv}")
    print("=" * 80)

    for r in results[:20]:
        print(
            f"[생성완료] key={r['key']} | players={r['n_players']} | "
            f"frames={r['n_frames']} | rows={r['n_rows']} | "
            f"gif={r['gif']} | csv={r['csv']}"
        )

    if len(results) > 20:
        print(f"... 외 {len(results) - 20}개")


if __name__ == "__main__":
    main()
