"""Download public UCI datasets and save the CSV schemas used by the labs.

Run from the project root with: .venv/bin/python data/prepare_datasets.py
"""
from __future__ import annotations

import gzip
import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


def uci(identifier: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = fetch_ucirepo(id=identifier)
    return dataset.data.features.copy(), dataset.data.targets.copy()


def csv(features: pd.DataFrame, targets: pd.DataFrame, filename: str,
        feature_names=None, target_name=None) -> pd.DataFrame:
    x = features.copy()
    if feature_names is not None:
        x.columns = feature_names
    y = targets.iloc[:, 0].copy()
    y.name = target_name or targets.columns[0]
    frame = pd.concat([x.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    frame.to_csv(DATA / filename, index=False)
    return frame


def download(url: str) -> bytes:
    with urlopen(url, timeout=60) as response:
        return response.read()


def main() -> None:
    # A different dataset from the example wine-classification lab: UCI's
    # dedicated red-wine file is used for both labs 01 and 02.
    wine_zip = zipfile.ZipFile(io.BytesIO(download(
        "https://archive.ics.uci.edu/static/public/186/wine+quality.zip")))
    wine_member = next(n for n in wine_zip.namelist() if n.lower().endswith("winequality-red.csv"))
    clean = pd.read_csv(wine_zip.open(wine_member), sep=";")
    clean.columns = [name.replace(" ", "_") for name in clean.columns]
    clean.to_csv(DATA / "wine_quality_red.csv", index=False)
    # Low (<=5), medium (=6), and high (>=7) quality classes.
    classes = pd.cut(clean.quality, [-np.inf, 5, 6, np.inf], labels=[1, 2, 3]).astype(int)
    clean.drop(columns="quality").assign(**{"class": classes}).to_csv(
        DATA / "knn_wine_quality.csv", index=False)

    # Regression on concrete compressive strength (UCI 165).
    x, y = uci(165)
    names = ["cement", "slag", "fly_ash", "water", "superplasticizer",
             "coarse_aggregate", "fine_aggregate", "age_days"]
    y.columns = ["strength_mpa"]
    csv(x, y, "concrete.csv", names, "strength_mpa")

    # Diabetes progression data used in the original lab description.
    base = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/"
    xx = pd.read_csv(io.BytesIO(gzip.decompress(download(base + "diabetes_data_raw.csv.gz"))),
                     sep=r"\s+", header=None)
    yy = pd.read_csv(io.BytesIO(gzip.decompress(download(base + "diabetes_target.csv.gz"))), header=None)
    xx.columns = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
    yy.columns = ["progression"]
    csv(xx, yy, "diabetes.csv")

    # SMS spam text corpus (UCI 228).
    payload = download("https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        member = next(n for n in archive.namelist() if n.lower().endswith("readme") is False and "sms" in n.lower())
        raw = pd.read_csv(archive.open(member), sep="\t", header=None, names=["label", "message"],
                          encoding="latin-1", quoting=3, on_bad_lines="skip")
    raw.to_csv(DATA / "sms_spam.csv", index=False)

    # Breast Cancer Wisconsin Diagnostic (UCI 17).
    x, y = uci(17)
    names = [c.lower().replace("_", "_") for c in x.columns]
    x.columns = names
    y.columns = ["diagnosis"]
    rename = {c: c.replace("radius1", "radius_mean").replace("texture1", "texture_mean")
                .replace("radius3", "radius_worst").replace("texture3", "texture_worst")
              for c in x.columns}
    x = x.rename(columns=rename)
    x.insert(0, "id", np.arange(1, len(x) + 1))
    csv(x, y, "breast_cancer.csv")

    # Banknote Authentication (UCI 267).
    x, y = uci(267)
    x.columns = ["variance", "skewness", "curtosis", "entropy"]
    y.columns = ["class"]
    csv(x, y, "banknote.csv")

    # Optical Digits (UCI 80); deterministic 80/20 train/test split.
    x, y = uci(80)
    rng = np.random.default_rng(42)
    labels = y.iloc[:, 0].to_numpy()
    train_idx, test_idx = [], []
    for label in np.unique(labels):
        idx = rng.permutation(np.flatnonzero(labels == label))
        ntest = round(len(idx) * 0.2)
        test_idx.extend(idx[:ntest]); train_idx.extend(idx[ntest:])
    digit_frame = x.copy()
    digit_frame.columns = [f"pixel_{i}" for i in range(x.shape[1])]
    digit_frame["digit"] = labels
    digit_frame.iloc[train_idx].to_csv(DATA / "optdigits_train.csv", index=False)
    digit_frame.iloc[test_idx].to_csv(DATA / "optdigits_test.csv", index=False)

    # Seeds (UCI 236), distributed as a whitespace-delimited text file.
    seed_zip = zipfile.ZipFile(io.BytesIO(download("https://archive.ics.uci.edu/static/public/236/seeds.zip")))
    seed_file = next(n for n in seed_zip.namelist() if n.lower().endswith("seeds_dataset.txt"))
    seeds = pd.read_csv(seed_zip.open(seed_file), sep=r"\s+", header=None,
                        names=["area", "perimeter", "compactness", "kernel_length", "kernel_width",
                               "asymmetry", "groove_length", "variety"])
    seeds.to_csv(DATA / "seeds.csv", index=False)

    # Sonar, Mines vs Rocks (UCI 151).
    x, y = uci(151)
    x.columns = [f"band_{i+1}" for i in range(x.shape[1])]
    y.columns = ["label"]
    csv(x, y, "sonar.csv")

    # Auto MPG (UCI 9); translate the integer origin codes into the script's
    # expected USA=1, Europe=2, Japan=3 convention.
    x, y = uci(9)
    origin_col = next(c for c in x.columns if c.lower() == "origin")
    x = x.rename(columns={c: c.lower().replace(" ", "_") for c in x.columns})
    x["origin"] = pd.to_numeric(x["origin"], errors="coerce")
    y.columns = ["mpg"]
    frame = x.join(y).dropna(subset=["horsepower"])
    frame.to_csv(DATA / "auto_mpg.csv", index=False)

    # Car Evaluation (UCI 19), entirely categorical.
    x, y = uci(19)
    y.columns = ["class"]
    csv(x, y, "car_evaluation.csv")

    # Ionosphere (UCI 52).
    x, y = uci(52)
    x.columns = [f"feature_{i+1}" for i in range(x.shape[1])]
    y.columns = ["label"]
    csv(x, y, "ionosphere.csv")

    print(f"Wrote datasets to {DATA}")
    for path in sorted(DATA.glob("*.csv")):
        print(f"  {path.name}: {sum(1 for _ in path.open()) - 1} rows")


if __name__ == "__main__":
    main()
