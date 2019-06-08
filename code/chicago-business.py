import configparser
import datetime
import os
from itertools import cycle
from pathlib import Path
from types import MethodType
import argparse
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib2tikz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import (BaggingClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from data_cleaning import clean_data, filter_out_2019_data
from feature_generation import (balance_features, count_by_dist_radius,
                                count_by_zip_year, make_dummy_vars,
                                make_features, reshape_and_create_label)
from pipeline.core import Pipeline
from pipeline.grid import Grid
from pipeline.transformation import (Transformation, to_datetime,
                                     replace_missing_with_mean)
from pipeline.splitter import Splitter
from joblib import dump, load


def clean_chicago_business_data(self):
    self.logger.info("    Running cleaning steps on raw data")
    self.dataframe = clean_data(self.dataframe, self.data_cleaning)
    return self


def make_chicago_business_features(self):

    original_features = self.features[:]
    old_cols = set(self.test_sets[0].columns)

    # Make features for each dataset in train_sets, test_sets
    n = len(self.feature_generators)
    for df_list in (self.test_sets, self.train_sets):
        for i in range(len(df_list)):
            self.logger.info("    Creating %s features on test-train set %s", n, i+1)

            # Filter out licenses for Jan 1, 2019 onwards
            df_list[i] = df_list[i].loc[df_list[i]['DATE ISSUED'] <= pd.to_datetime('12/31/2018')]
            df_list[i] = make_features(df_list[i], self.feature_generators, original_features)

    # Check for feature balance on each test-train pair
    for i in range(len(self.test_sets)):
        self.logger.info("    Balancing features for test-train set %s", i+1)
        self.train_sets[i], self.test_sets[i] = balance_features(self.train_sets[i], self.test_sets[i])

        # Filter only for business-years in 1 year test window
        self.test_sets[i] = self.test_sets[i].loc[
            self.test_sets[i]['YEAR'] == self.test_sets[i]['YEAR'].max()
        ]

    # Add newly-generated features to self.features
    new_cols = set(self.test_sets[0].columns)
    self.features += list(new_cols - old_cols) # set difference

    self.features = list(set(self.features) - set([self.target]))

    return self


def get_pipeline(config_path):
    try:
        script_dir = Path(__file__).parent
    except NameError:
        script_dir = Path(os.path.abspath(''))

    with open(script_dir.resolve()/config_path, 'rb') as config_file:
        config = yaml.safe_load(config_file.read())

    input_path    = script_dir/config["data"]["input_path"]
    output_dir    = script_dir/config["data"]["output_dir"]
    target        = config["pipeline"]["target"]
    pipeline_name = config["pipeline"]["name"]
    features      = config["pipeline"]["precomputed_features"]
    model_grid    = Grid.from_config(config["models"])
    splitter      = Splitter.from_config(config["pipeline"]["test_train"])

    pipeline = Pipeline(
        input_source    = input_path,
        target          = target,
        name            = pipeline_name,
        output_root_dir = output_dir,
        model_grid      = model_grid,
        splitter        = splitter,
        features        = features,
        data_cleaning   = [
            to_datetime("LICENSE TERM EXPIRATION DATE"),
            to_datetime("DATE ISSUED"),
            replace_missing_with_mean('medhhinc'),
            replace_missing_with_mean('a35to64_share'),
            replace_missing_with_mean('share_BA+'),
            replace_missing_with_mean('total_pop'),
            replace_missing_with_mean('metro_GDP'),
            replace_missing_with_mean('Cook_U3_ann_avg'),
            replace_missing_with_mean('num_sites'),
            replace_missing_with_mean('in_ssa'),
            replace_missing_with_mean('which_ssa'),
            replace_missing_with_mean('num_renewals')
        ],
        feature_generators=[
            count_by_zip_year,
            make_dummy_vars
        ])

    # pipeline.clean_data        = MethodType(clean_chicago_business_data, pipeline)
    pipeline.generate_features = MethodType(make_chicago_business_features, pipeline)
    return pipeline

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help = "config file path")
    args = parser.parse_args()

    pipeline = get_pipeline(args.config)
    pipeline.run()

    try:
        for (description, model) in pipeline.trained_models.items():
            dump(model, "models/" + args.config["description"] + "_" + description + ".joblib" )
    except Exception as e:
        print(e)
    # pipeline = (get_pipeline("config.yml")
    #              .load_data()
    #              .clean_data()
    #              .summarize_data()
    #              .generate_test_train()
    #              .preprocess_data()
    #              .generate_features())
