import configparser
import datetime
from itertools import cycle
from pathlib import Path
from types import MethodType

import matplotlib2tikz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import (BaggingClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from pipeline.core import Pipeline
from pipeline.transformation import (Transformation, to_datetime)
from pipeline.grid import Grid

from data_cleaning import (clean_data, filter_out_2019_data)
from feature_generation import (make_features, reshape_and_create_label,
                                count_by_zip_year, count_by_dist_radius,
                                balance_features, make_dummy_vars)

def clean_chicago_business_data(self):
    self.logger.info("    Running cleaning steps on raw data")
    self.dataframe = clean_data(self.dataframe, self.data_cleaning)
    return self


def make_chicago_business_features(self):

    # Make features for each dataset in train_sets, test_sets
    n = len(self.feature_generators)
    for df_list in (self.test_sets, self.train_sets):
        for i in range(len(df_list)):
            self.logger.info("    Creating %s features on test-train set %s", n, i+1)
            df_list[i] = make_features(df_list[i], self.feature_generators)

    # Check for feature balance on each test-train pair
    for i in range(len(self.test_sets)):
        self.logger.info("    Balancing features for test-train set %s", i+1)
        self.train_sets[i], self.test_sets[i] = balance_features(self.train_sets[i], self.test_sets[i])

    return self


def main(config_path):
    script_dir = Path(__file__).parent

    with open(script_dir.resolve()/config_path, 'rb') as config_file:
        config = yaml.safe_load(config_file.read())

    input_path    = script_dir/config["data"]["input_path"]
    output_dir    = script_dir/config["data"]["output_dir"]
    target        = config["pipeline"]["target"]
    pipeline_name = config["pipeline"]["name"]
    model_grid    = Grid.from_config(config["models"])

    pipeline = Pipeline(
        input_source    = input_path,
        target          = target,
        name            = pipeline_name,
        output_root_dir = output_dir,
        model_grid      = model_grid,
        data_cleaning   = [
            to_datetime("LICENSE TERM EXPIRATION DATE"),
            to_datetime("DATE ISSUED"),
            filter_out_2019_data
        ],
        feature_generators = [
            count_by_zip_year,      # num_not_renewed_zip
            count_by_dist_radius,   # num_not_renewed_1km
            make_dummy_vars         # CITY, STATE, APPLICATION TYPE
        ])

    pipeline.clean_data        = MethodType(clean_chicago_business_data, pipeline)
    pipeline.generate_features = MethodType(make_chicago_business_features, pipeline)

    pipeline.run()


if __name__ == "__main__":
    main("config.yml")
