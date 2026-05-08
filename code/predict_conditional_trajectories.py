#!/usr/bin/env python3
# predict_conditional_trajectories.py
# NFL Big Data Bowl 2026 - Conditional Clustering 기반 x, y 예측 스크립트
# 원본 conditional_control_distribution_ver3.py의 clustering 로직을 **100% 재사용**
# → train 데이터로 condition_id별 future trajectory를 학습 → test_input.csv 예측

import os
import re
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless

# ====================== 원본 스크립트에서 그대로 가져온 상수 & 함수 ======================
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
        output_name = f"output_{year}_w{week_str}.csv"
        output_path = train_dir / output_name
        if not output_path.exists():
            print(f"[warning] output file not found for {input_path.name}")
            output_path = None
        pairs.append({
            "year": year,
            "week": int(week_str),
            "input_path": input_path,
            "output_path": output_path,
        })
    return pairs


def read_csv_selected(path: Path, wanted_cols):
    """원본 함수 + wanted_cols=None 일 때 전체 컬럼 로드 지원 (test_input.csv용)"""
    if wanted_cols is None or len(wanted_cols) == 0:
        print(f"[info] reading ALL columns from {path.name}")
        return pd.read_csv(path, low_memory=False)

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


def build_condition_id(df, condition_cols):
    df["condition_id"] = df.groupby(condition_cols, dropna=False, sort=False).ngroup().astype(int) + 1
    return df


# ====================== PREDICTION 전용 함수 ======================
def load_full_output(output_path: Path):
    """output_*.csv 전체를 로드 (미래 프레임 포함)"""
    if output_path is None or not output_path.exists():
        return None
    wanted = ID_COLS + ["frame_id", "x", "y"]
    out = read_csv_selected(output_path, wanted)
    for col in ["frame_id", "x", "y"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=ID_COLS)


def process_train_pair_for_prediction(serial_p, input_wanted_cols, x_bin, y_bin, ball_bin, yardline_bin, age_bin, weight_bin):
    """train 한 파일을 처리하면서 condition_id별 future delta를 수집"""
    year = serial_p["year"]
    week = serial_p["week"]
    input_path = Path(serial_p["input_path"])
    output_path = Path(serial_p["output_path"]) if serial_p.get("output_path") else None

    print(f"[train] processing {input_path.name}")

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

    # trajectory ID 생성
    inp["_trajectory_id"] = (
        inp["source_year"].astype(str) + "_w" + inp["source_week"].astype(str) +
        "_g" + inp["game_id"].astype(str) + "_p" + inp["play_id"].astype(str) +
        "_n" + inp["nfl_id"].astype(str)
    )

    # condition_id 생성
    condition_cols = [c for c in [
        "player_position", "player_side", "player_role", "play_direction",
        "player_weight_bin", "player_age_bin", "x_bin", "y_bin",
        "ball_land_x_bin", "ball_land_y_bin", "absolute_yardline_bin"
    ] if c in inp.columns]
    inp = build_condition_id(inp, condition_cols)

    # 마지막 input frame만 추출 (conditioning point)
    last_input = inp.sort_values(TRAJ_COLS + ["frame_id"]).groupby(TRAJ_COLS, dropna=False).tail(1).copy()

    # output 전체 로드
    out_df = load_full_output(output_path)
    if out_df is None:
        return None, condition_cols

    # output에도 trajectory_id 생성
    out_df["_trajectory_id"] = (
        str(year) + "_w" + str(week) +
        "_g" + out_df["game_id"].astype(str) + "_p" + out_df["play_id"].astype(str) +
        "_n" + out_df["nfl_id"].astype(str)
    )

    future_data = []
    for _, last_row in last_input.iterrows():
        tid = last_row["_trajectory_id"]
        cond_id = int(last_row["condition_id"])
        curr_x, curr_y = last_row["x"], last_row["y"]

        traj_out = out_df[out_df["_trajectory_id"] == tid].sort_values("frame_id")
        if len(traj_out) == 0:
            continue

        deltas = []
        for _, row_out in traj_out.iterrows():
            dx = row_out["x"] - curr_x
            dy = row_out["y"] - curr_y
            deltas.append((dx, dy, row_out["frame_id"]))

        future_data.append({
            "condition_id": cond_id,
            "deltas": deltas,
            "n_future": len(deltas)
        })

    return pd.DataFrame(future_data), condition_cols


def build_condition_lookup(train_dir: Path, x_bin=5.0, y_bin=2.0, ball_bin=5.0,
                           yardline_bin=5.0, age_bin=3.0, weight_bin=20.0, max_files=None):
    """train 전체를 처리하여 condition_id → future deltas lookup 생성"""
    pairs = discover_input_output_pairs(train_dir)
    if max_files:
        pairs = pairs[:max_files]

    input_wanted_cols = list(set(
        ID_COLS + ["frame_id"] + STATE_CATEGORICAL_COLS + EXTERNAL_CATEGORICAL_COLS +
        STATE_NUMERIC_RAW_COLS + EXTERNAL_NUMERIC_RAW_COLS + CONTROL_COLS
    ))

    max_workers = min(8, multiprocessing.cpu_count())
    serial_pairs = []
    for p in pairs:
        serial_p = p.copy()
        serial_p["input_path"] = str(p["input_path"])
        serial_p["output_path"] = str(p["output_path"]) if p["output_path"] else None
        serial_pairs.append(serial_p)

    print(f"[info] Building lookup from {len(serial_pairs)} train files...")

    all_futures = []
    condition_cols = None

    if len(serial_pairs) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            worker_func = partial(process_train_pair_for_prediction,
                                  input_wanted_cols=input_wanted_cols,
                                  x_bin=x_bin, y_bin=y_bin, ball_bin=ball_bin,
                                  yardline_bin=yardline_bin, age_bin=age_bin,
                                  weight_bin=weight_bin)
            results = list(executor.map(worker_func, serial_pairs))
    else:
        results = [process_train_pair_for_prediction(p, input_wanted_cols, x_bin, y_bin,
                                                     ball_bin, yardline_bin, age_bin, weight_bin)
                   for p in serial_pairs]

    for res, cols in results:
        if res is not None and len(res) > 0:
            all_futures.append(res)
            if condition_cols is None:
                condition_cols = cols

    if not all_futures:
        raise RuntimeError("No training data with output found!")

    lookup_df = pd.concat(all_futures, ignore_index=True)

    lookup = {}
    for cid, group in lookup_df.groupby("condition_id"):
        lookup[int(cid)] = group["deltas"].tolist()

    print(f"[lookup built] {len(lookup)} unique conditions, total future examples: {len(lookup_df):,}")
    return lookup, condition_cols


def predict_for_test(test_path: Path, lookup, condition_cols,
                     x_bin=5.0, y_bin=2.0, ball_bin=5.0, yardline_bin=5.0,
                     age_bin=3.0, weight_bin=20.0):
    """test_input.csv를 읽고 player_to_predict=True인 선수에 대해 미래 x,y 예측"""
    print(f"[test] loading {test_path.name}")
    test_df = read_csv_selected(test_path, None)   # ← ALL columns (fixed)

    # source_year 추출 (test 파일에 다양한 연도가 있을 수 있음)
    test_df["source_year"] = test_df["game_id"].astype(str).str[:4].astype(int)

    test_df = add_age_and_bins(test_df, source_year=2024, x_bin=x_bin, y_bin=y_bin,
                               ball_bin=ball_bin, yardline_bin=yardline_bin,
                               age_bin=age_bin, weight_bin=weight_bin)

    test_df = build_condition_id(test_df, condition_cols)

    targets = test_df[test_df["player_to_predict"] == True].copy()
    if len(targets) == 0:
        print("[warning] No player_to_predict=True rows found!")
        return pd.DataFrame()

    print(f"[predict] {len(targets):,} target rows to predict")

    predictions = []

    for traj_key, traj_group in targets.groupby(["game_id", "play_id", "nfl_id"]):
        traj_group = traj_group.sort_values("frame_id")
        last_frame = traj_group.iloc[-1]

        cid = int(last_frame["condition_id"])
        curr_x = float(last_frame["x"])
        curr_y = float(last_frame["y"])
        num_future = int(last_frame.get("num_frames_output", 30))

        if cid in lookup and len(lookup[cid]) > 0:
            hist_deltas = lookup[cid][0]  # 가장 첫 번째 historical trajectory 사용
            hist_deltas = sorted(hist_deltas, key=lambda t: t[2])
        else:
            print(f"[fallback] condition_id={cid} not found → zero motion")
            hist_deltas = [(0.0, 0.0, i) for i in range(1, num_future + 1)]

        future_frames = []
        for i in range(min(num_future, len(hist_deltas))):
            dx, dy, _ = hist_deltas[i]
            pred_x = curr_x + dx
            pred_y = curr_y + dy
            future_frames.append({
                "game_id": int(last_frame["game_id"]),
                "play_id": int(last_frame["play_id"]),
                "nfl_id": int(last_frame["nfl_id"]),
                "frame_id": i + 1,
                "x": round(pred_x, 4),
                "y": round(pred_y, 4)
            })

        # 부족한 프레임은 마지막 위치 반복
        while len(future_frames) < num_future:
            last = future_frames[-1]
            future_frames.append({
                "game_id": last["game_id"],
                "play_id": last["play_id"],
                "nfl_id": last["nfl_id"],
                "frame_id": len(future_frames) + 1,
                "x": last["x"],
                "y": last["y"]
            })

        predictions.extend(future_frames)

    pred_df = pd.DataFrame(predictions)
    print(f"[done] Generated {len(pred_df):,} prediction rows")
    return pred_df


def main():
    parser = argparse.ArgumentParser(description="Conditional clustering 기반 NFL trajectory 예측")
    parser.add_argument("--train-dir", type=str, default="./train", help="train 데이터 디렉토리")
    parser.add_argument("--test-path", type=str, default="test_input.csv", help="test_input.csv 경로")
    parser.add_argument("--output", type=str, default="predictions.csv", help="예측 결과 저장 경로")
    parser.add_argument("--x-bin", type=float, default=5.0)
    parser.add_argument("--y-bin", type=float, default=2.0)
    parser.add_argument("--ball-bin", type=float, default=5.0)
    parser.add_argument("--yardline-bin", type=float, default=5.0)
    parser.add_argument("--age-bin", type=float, default=3.0)
    parser.add_argument("--weight-bin", type=float, default=20.0)
    parser.add_argument("--max-train-files", type=int, default=None, help="디버깅용")
    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    test_path = Path(args.test_path)

    lookup, condition_cols = build_condition_lookup(
        train_dir=train_dir,
        x_bin=args.x_bin, y_bin=args.y_bin, ball_bin=args.ball_bin,
        yardline_bin=args.yardline_bin, age_bin=args.age_bin, weight_bin=args.weight_bin,
        max_files=args.max_train_files
    )

    pred_df = predict_for_test(
        test_path=test_path,
        lookup=lookup,
        condition_cols=condition_cols,
        x_bin=args.x_bin, y_bin=args.y_bin, ball_bin=args.ball_bin,
        yardline_bin=args.yardline_bin, age_bin=args.age_bin, weight_bin=args.weight_bin
    )

    if not pred_df.empty:
        pred_df.to_csv(args.output, index=False)
        print(f"\n✅ 예측 완료! 결과 저장: {args.output}")
        print(pred_df.head())
        print(f"Total rows: {len(pred_df)}")
    else:
        print("❌ 예측 결과가 없습니다.")


if __name__ == "__main__":
    main()