import json
from pathlib import Path

import pandas as pd

COMMON_DATA_DIR = Path("..", "..", "common", "data")
RAWDF_DIR = COMMON_DATA_DIR / "rawdf"
MAPPING_DIR = COMMON_DATA_DIR / "mapping"
OUTPUT_DIR = Path("..", "data", "01_preprocessed")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# カテゴリ変数を数値に変換するためのマッピング
with open(MAPPING_DIR / "sex.json", "r") as f:
    sex_mapping = json.load(f)
with open(MAPPING_DIR / "race_type.json", "r") as f:
    race_type_mapping = json.load(f)
with open(MAPPING_DIR / "around.json", "r") as f:
    around_mapping = json.load(f)
with open(MAPPING_DIR / "weather.json", "r") as f:
    weather_mapping = json.load(f)
with open(MAPPING_DIR / "ground_state.json", "r") as f:
    ground_state_mapping = json.load(f)
with open(MAPPING_DIR / "race_class.json", "r") as f:
    race_class_mapping = json.load(f)


def process_results(
    input_dir: Path = RAWDF_DIR,
    output_dir: Path = OUTPUT_DIR,
    save_filename: str = "results.csv",
    sex_mapping: dict = sex_mapping,
) -> pd.DataFrame:
    """
    未加工のレース結果テーブルをinput_dirから読み込んで加工し、
    output_dirに保存する関数。
    """
    df = pd.read_csv(input_dir / save_filename, sep="\t")
    df["rank"] = pd.to_numeric(df["着順"], errors="coerce")
    df.dropna(subset=["rank"], inplace=True)
    df["rank"] = df["rank"].astype(int)
    df["umaban"] = df["馬番"].astype(int)
    df["tansho_odds"] = df["単勝"].astype(float)
    df["popularity"] = df["人気"].astype(int)
    df["impost"] = df["斤量"].astype(float)
    df["wakuban"] = df["枠番"].astype(int)
    df["sex"] = df["性齢"].str[0].map(sex_mapping)
    df["age"] = df["性齢"].str[1:].astype(int)
    df["weight"] = df["馬体重"].str.extract(r"(\d+)").astype(int)
    df["weight_diff"] = df["馬体重"].str.extract(r"\((.+)\)").astype(int)
    # データが着順に並んでいることによるリーク防止のため、各レースを馬番順にソートする
    df = df.sort_values(["race_id", "umaban"])
    # 使用する列を選択
    df = df[
        [
            "race_id",
            "horse_id",
            "jockey_id",
            "trainer_id",
            "owner_id",
            "rank",
            "umaban",
            "wakuban",
            "tansho_odds",
            "popularity",
            "impost",
            "sex",
            "age",
            "weight",
            "weight_diff",
        ]
    ]
    df.to_csv(output_dir / save_filename, sep="\t", index=False)
    return df


def process_race_info(
    input_dir: Path = RAWDF_DIR,
    output_dir: Path = OUTPUT_DIR,
    save_filename: str = "race_info.csv",
    race_type_mapping: dict = race_type_mapping,
    around_mapping: dict = around_mapping,
    weather_mapping: dict = weather_mapping,
    ground_state_mapping: dict = ground_state_mapping,
    race_class_mapping: dict = race_class_mapping,
) -> pd.DataFrame:
    """
    未加工のレース情報テーブルをinput_dirから読み込んで加工し、
    output_dirに保存する関数。
    """
    df = pd.read_csv(input_dir / save_filename, sep="\t")
    # evalで文字列型の列をリスト型に変換し、一時的な列を作成
    df["tmp"] = df["info1"].map(lambda x: eval(x)[0])
    # ダートor芝or障害
    df["race_type"] = df["tmp"].str[0].map(race_type_mapping)
    # 右or左or直線
    df["around"] = df["tmp"].str[1].map(around_mapping)
    df["course_len"] = df["tmp"].str.extract(r"(\d+)")
    df["weather"] = df["info1"].str.extract(r"天候:(\w+)")[0].map(weather_mapping)
    df["ground_state"] = (
        df["info1"].str.extract(r"(芝|ダート|障害):(\w+)")[1].map(ground_state_mapping)
    )
    df["date"] = pd.to_datetime(
        df["info2"].map(lambda x: eval(x)[0]), format="%Y年%m月%d日"
    )
    regex_race_class = "|".join(race_class_mapping)
    df["race_class"] = (
        df["title"]
        .str.extract(rf"({regex_race_class})")
        # タイトルからレース階級情報が取れない場合はinfo2から取得
        .fillna(df["info2"].str.extract(rf"({regex_race_class})"))[0]
        .map(race_class_mapping)
    )
    df["place"] = df["race_id"].astype(str).str[4:6].astype(int)
    # 使用する列を選択
    df = df[
        [
            "race_id",
            "date",
            "race_type",
            "around",
            "course_len",
            "weather",
            "ground_state",
            "race_class",
            "place",
        ]
    ]
    df.to_csv(output_dir / save_filename, sep="\t", index=False)
    return df


def process_horse_results(
    input_dir: Path = RAWDF_DIR,
    output_dir: Path = OUTPUT_DIR,
    save_filename: str = "horse_results.csv",
    race_type_mapping: dict = race_type_mapping,
    weather_mapping: dict = weather_mapping,
    ground_state_mapping: dict = ground_state_mapping,
    race_class_mapping: dict = race_class_mapping,
) -> pd.DataFrame:
    """
    未加工の馬の過去成績テーブルをinput_dirから読み込んで加工し、
    output_dirに保存する関数。
    """
    df = pd.read_csv(input_dir / save_filename, sep="\t")
    df["rank"] = pd.to_numeric(df["着順"], errors="coerce")
    df.dropna(subset=["rank"], inplace=True)
    df["date"] = pd.to_datetime(df["日付"])
    df["weather"] = df["天気"].map(weather_mapping)
    df["race_type"] = df["距離"].str[0].map(race_type_mapping)
    df["course_len"] = df["距離"].str.extract(r"(\d+)").astype(int)
    df["ground_state"] = df["馬場"].map(ground_state_mapping)
    # 着差は1着以外は「1着との差」を表すが、1着のみ「2着との差」のデータが入っている
    df["rank_diff"] = df["着差"].map(lambda x: 0 if x < 0 else x)
    df["prize"] = df["賞金"].fillna(0)
    regex_race_class = "|".join(race_class_mapping)
    df["race_class"] = (
        df["レース名"].str.extract(rf"({regex_race_class})")[0].map(race_class_mapping)
    )
    df.rename(columns={"頭数": "n_horses"}, inplace=True)
    # 使用する列を選択
    df = df[
        [
            "horse_id",
            "date",
            "rank",
            "prize",
            "rank_diff",
            "weather",
            "race_type",
            "course_len",
            "ground_state",
            "race_class",
            "n_horses",
        ]
    ]
    df.to_csv(output_dir / save_filename, sep="\t", index=False)
    return df
