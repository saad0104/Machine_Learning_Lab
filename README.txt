CSE 616 ML Lab
====================================

Environment setup (Linux):
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

The concise report source is in chapters_report/ and is included by
main_share.tex. Compile twice with XeLaTeX from this directory:
    xelatex main_share.tex
    xelatex main_share.tex
The existing main_share.pdf is the starter PDF and has not been regenerated.

For the report's full source-code listings, Overleaf must also receive all 13
lab folders containing the .py files, plus chapters_report/, fonts/, the figure
PDFs under each lab's figures/ directory, and main_share.tex. The data/ folder
and virtual environment are not needed just to compile the document.
