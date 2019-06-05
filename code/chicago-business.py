import configparser
import datetime
from itertools import cycle
from pathlib import Path
from types import MethodType

import matplotlib
matplotlib.use('TkAgg')
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
<<<<<<< HEAD
from pipeline.transformation import (Transformation, to_datetime)
=======
from pipeline.transformation import (Transformation, binarize, categorize,
                                     hash_string, replace_missing_with_value,
                                     scale_by_max, to_datetime)
>>>>>>> first-model
from pipeline.grid import Grid

from data_cleaning import (clean_data, filter_out_2019_data)
from feature_generation import (make_features, reshape_and_create_label,
                                count_by_zip_year, count_by_dist_radius,
<<<<<<< HEAD
                                balance_features, make_dummy_vars)
=======
                                balance_features)

def explore():
    pass

def clean():
    pass

def predict():
    pass

>>>>>>> first-model

def clean_chicago_business_data(self):
    self.logger.info("    Running cleaning steps on raw data")
    self.dataframe = clean_data(self.dataframe, self.data_cleaning)
    return self


def make_chicago_business_features(self):

    old_cols = set(self.test_sets[0].columns)

    # Make features for each dataset in train_sets, test_sets
    n = len(self.feature_generators)
    for df_list in (self.test_sets, self.train_sets):
        for i in range(len(df_list)):
            self.logger.info("    Creating features on test-train set %s", i+1)
            # Cecile deleted n in the above log line because she can't tell what it's supposed to be

            df_list[i] = make_features(df_list[i], self.feature_generators)

    # Check for feature balance on each test-train pair
    for i in range(len(self.test_sets)):
        self.logger.info("    Balancing features for test-train set %s", i+1)
        self.train_sets[i], self.test_sets[i] = balance_features(self.train_sets[i], self.test_sets[i])

<<<<<<< HEAD
=======
    # Add newly-generated features to self.features
    new_cols = set(self.test_sets[0].columns)
    self.features = list(new_cols - old_cols) # set difference
    print(self.features)

>>>>>>> first-model
    return self


# def get_pipeline(config, original_df=original_df):
#     pipeline = Pipeline(
#         original_df,
#         "not_renewed_2yrs",
#         data_cleaning = [
#             to_datetime("LICENSE TERM EXPIRATION DATE"),
#             to_datetime("DATE ISSUED")
#         ],
#         feature_generators=[
#             count_by_zip_year,
#             count_by_dist_radius
#         ],
#         model_grid=Grid.from_config(config["models"]),
#         name="quick-pipeline-lr-only",
#         output_root_dir=Path("output/"))

#     pipeline.generate_features = MethodType(make_chicago_business_features, pipeline)
#     return pipeline

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
<<<<<<< HEAD
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
=======
        Path(config["data"]["input_path"]),
        config["pipeline"]["target"],
        data_cleaning=[
            to_datetime("LICENSE TERM EXPIRATION DATE"),
            to_datetime("DATE ISSUED")
        ],
        feature_generators=[
            count_by_zip_year,
            count_by_dist_radius
        ],
        summarize=False,
        model_grid=Grid.from_config(config["models"]),
        name="quick-pipeline-lr-only-" + config["description"],
        output_root_dir=Path("output/"))
>>>>>>> first-model
    pipeline.generate_features = MethodType(make_chicago_business_features, pipeline)
    pipeline.run()

if __name__ == "__main__":
    main("config.yml")
