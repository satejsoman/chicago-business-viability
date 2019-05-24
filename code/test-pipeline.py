import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn.datasets
import yaml
from sklearn.ensemble import (BaggingClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from pipeline.core import Pipeline
from pipeline.grid import Grid


def main(config_path):
    warnings.filterwarnings('ignore')

    with open(config_path, 'rb') as config_file:
        config = yaml.safe_load(config_file.read())
    
    input_path    = config["data"]["input_path"]
    output_dir    = config["data"]["output_dir"]
    pipeline_name = config["pipeline"]["name"]
    target        = config["pipeline"]["target"]
    features      = config["pipeline"]["features"]
    models        = config["models"]
    
    if not input_path.exists():
        iris = sklearn.datasets.load_iris()
        pd.DataFrame(
            data = np.c_[iris['data'], (iris['target'] > 1).astype(int)], 
            columns = [_.replace(' (cm)', '') for _ in iris['feature_names']] + ['target']
        ).to_csv(input_path)

    pipeline = Pipeline(
        csv_path        = Path(input_path), 
        target          = target, 
        model_grid      = Grid.from_config(models),
        name            = pipeline_name,
        features        = features,
        output_root_dir = Path(output_dir))
    pipeline.run()

if __name__ == "__main__":
    main("test_config.yaml")
