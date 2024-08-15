import pickle
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import yaml

DATA_DIR = Path("..", "data")
INPUT_DIR = DATA_DIR / "02_features"
OUTPUT_DIR = DATA_DIR / "03_train"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


class Trainer:
    def __init__(
        self,
        features_filepath: Path = INPUT_DIR / "features.csv",
        config_filepath: Path = "config.yaml",
        output_dir: Path = OUTPUT_DIR,
    ):
        self.features = pd.read_csv(features_filepath, sep="\t")
        with open(config_filepath, "r") as f:
            self.feature_cols = yaml.safe_load(f)["features"]
        self.output_dir = output_dir

    def create_dataset(self, test_start_date: str):
        """
        test_start_dateをYYYY-MM-DD形式で指定すると、
        その日付以降のデータをテストデータに、
        それより前のデータを学習データに分割する関数。
        """
        # 目的変数
        self.features["target"] = (self.features["rank"] == 1).astype(int)
        # 学習データとテストデータに分割
        self.train_df = self.features.query("date < @test_start_date")
        self.test_df = self.features.query("date >= @test_start_date")

    def train(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        importance_filename: str,
        model_filename: str,
    ) -> pd.DataFrame:
        # データセットの作成
        lgb_train = lgb.Dataset(train_df[self.feature_cols], train_df["target"])
        lgb_test = lgb.Dataset(test_df[self.feature_cols], test_df["target"])
        # パラメータの設定
        params = {
            "objective": "binary",  # 二値分類
            "metric": "binary_logloss",  # 予測誤差
            "random_state": 100,  # 実行ごとに同じ結果を得るための設定
            "verbosity": -1,  # 学習中のログを非表示
        }
        # 学習の実行
        model = lgb.train(
            params=params,
            train_set=lgb_train,
            valid_sets=[lgb_train, lgb_test],
            callbacks=[lgb.log_evaluation(100)],
        )
        with open(self.output_dir / model_filename, "wb") as f:
            pickle.dump(model, f)
        # 特徴量重要度の可視化
        lgb.plot_importance(model, importance_type="gain", figsize=(12, 6))
        plt.savefig(self.output_dir / importance_filename)
        plt.close()
        # テストデータに対してスコアリング
        evaluation_df = test_df[
            [
                "race_id",
                "horse_id",
                "target",
                "rank",
                "tansho_odds",
                "popularity",
                "umaban",
            ]
        ].copy()
        evaluation_df["pred"] = model.predict(test_df[self.feature_cols])
        return evaluation_df

    def run(
        self,
        test_start_date: str,
        importance_filename: str = "importance.png",
        model_filename: str = "model.pkl",
    ):
        """
        学習処理を実行する。
        test_start_dateをYYYY-MM-DD形式で指定すると、
        その日付以降のデータをテストデータに、
        それより前のデータを学習データに分割する
        """
        self.create_dataset(test_start_date)
        evaluation_df = self.train(
            self.train_df, self.test_df, importance_filename, model_filename
        )
        return evaluation_df
