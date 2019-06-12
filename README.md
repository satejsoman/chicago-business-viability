# chicago-business-viability

[![Travis (.org)](https://img.shields.io/travis/satejsoman/chicago-business-viability.svg?label=tests&logoColor=black&style=for-the-badge&logoColor=gray&logo=travis-ci)](https://travis-ci.com/satejsoman/chicago-business-viability) [![python.org](https://img.shields.io/badge/made%20with-python-%233776AB.svg?style=for-the-badge&logo=python&logoColor=ffdf76)](https://www.python.org)

## structure

- `code` holds all of the core pipeline code, as well as specific scripts for this project.
    - `code/data-assembly` holds intermediate files required to assemble our final datasets from their component raw data files.
    - `code/output` holds evaluation files and charts for each of our model runs.
    - `code/pipeline` holds the library implementation and unit tests.
    - `chicago_business.py` is the primary script for running our pipeline.
    - `config.yml` holds specified parameters for the pipeline to be run.
    - `data_cleaning.py` holds project-specific data cleaning methods.
    - `feature_generation.py` holds project-specific feature generation methods.
    - `requirements.txt` holds required libraries to run the project code.
- `data` holds all the raw data file that are accessed by `glue.py` to assemble the final processed datasets.
- `explorations` holds ad-hoc and one-off files for data exploration conducted while completing this project.

Instructions for running the pipeline are listed below.

## running the pipeline
The following commands should be run in the root directory:

### 1. set up a virtual environment
```
python3 -mvenv env
source ./env/bin/activate
```

### 2. install requirements
```
pip3 install -r code/requirements.txt
```

### 3. assemble dataset
- In `data`, unzip `licenses_joined.csv.gz`, `Business_Licenses.csv.gz`, and `merged_business_govtdata.csv.gz`.
- Then, from `code`, run `python3 data-assembly/glue.py`.
- This creates `joined_table.csv` in the `data` folder, which the pipeline takes as input.

### 4. execute pipeline from /code
```
python3 chicago_business.py
```
### 5. view results in /code/output

The evaluation file and precision-recall curves for the classifiers and hyperparameters specified in `config.yml` will be in the folder marked `LATEST`.

### 6. run bias/fairness audit
From a pipeline, save a model (using `joblib`) and its associated test or train set (using `pd.to_csv(..)`). Point the paths of `code/aequitas_audit.py` to the locations of the saved artifacts and run the script to see the cross-tabulations by `num_renewals`.

## other actions

### run unit tests and end-to-end smoketest
from the root directory, with the virtual environment activated, run:
```
pytest
```
