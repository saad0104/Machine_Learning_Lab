CSE 615 ML Lab
====================================

Environment setup (Linux/macOS):
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt

Download/prepare public datasets (requires an internet connection):
    .venv/bin/python data/prepare_datasets.py

Run any lab from the project root, for example:
    .venv/bin/python 01_KNN/knn.py

The scripts write results.txt and figures/ beside each script. Dataset CSVs are
stored in data/. The raw datasets come from the UCI Machine Learning Repository;
the diabetes files are from scikit-learn's public dataset mirror. See
data/prepare_datasets.py for source URLs, transformations, and train/test split.

Dependencies used by the lab scripts: NumPy, Pandas, Matplotlib. ucimlrepo is
used by the preparation script to download several UCI datasets.

The LaTeX report files were copied from the starter material. They have not yet
been updated to match all datasets and newly generated results; do not treat the
existing PDF or chapter results as results from these revised scripts.
