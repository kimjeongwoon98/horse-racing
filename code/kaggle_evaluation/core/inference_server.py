#!/usr/bin/env python3
# inference_server.py
# NFL Big Data Bowl 2026 - Precomputed predictions 기반 제출 코드

import pandas as pd
from pathlib import Path

from kaggle_evaluation.core.relay import define_server


class NFLInferenceServer:
    def __init__(self):
        # 미리 생성한 predictions.csv 로드 (한 번만)
        self.predictions = self._load_predictions()

    def _load_predictions(self):
        """predictions.csv 로드 (Kaggle Dataset 또는 local)"""
        paths_to_try = [
            Path("/kaggle/input/nfl-predictions/predictions.csv"),   # Kaggle Dataset 경로 (필요시 변경)
            Path("/kaggle/input/predictions/predictions.csv"),
            Path("predictions.csv"),                                 # local test
            Path("/home/workdir/attachments/predictions.csv")        # sandbox
        ]
        for p in paths_to_try:
            if p.exists():
                print(f"[info] Loaded predictions from: {p}")
                return pd.read_csv(p)
        raise FileNotFoundError("predictions.csv not found. Please upload it as Dataset.")

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Kaggle Gateway가 호출하는 함수"""
        # player_to_predict 컬럼이 없으면 전체 예측 (안전장치)
        if "player_to_predict" not in test_df.columns:
            test_df["player_to_predict"] = True

        # 예측 대상 trajectory만 추출
        targets = test_df[test_df["player_to_predict"] == True].copy()
        if targets.empty:
            return pd.DataFrame(columns=["game_id", "play_id", "nfl_id", "frame_id", "x", "y"])

        # predictions.csv에서 해당 trajectory의 예측값 가져오기
        pred_subset = pd.merge(
            targets[["game_id", "play_id", "nfl_id"]].drop_duplicates(),
            self.predictions,
            on=["game_id", "play_id", "nfl_id"],
            how="left"
        )

        # 필요한 컬럼만 반환 (Kaggle이 요구하는 형식)
        result = pred_subset[["game_id", "play_id", "nfl_id", "frame_id", "x", "y"]].copy()
        result = result.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])

        print(f"[predict] Returned {len(result)} prediction rows")
        return result


# ====================== Kaggle 서버 등록 ======================
def main():
    server = NFLInferenceServer()
    define_server(server.predict).serve()


if __name__ == "__main__":
    main()