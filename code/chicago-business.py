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
from pipeline.transformation import (Transformation, binarize, categorize,
                                     hash_string, replace_missing_with_value,
                                     scale_by_max)
from pipeline.grid import Grid

from data_cleaning import (clean_data, filter_out_2019_data)
from feature_generation import (make_features, reshape_and_create_label,
                                count_by_zip_year, count_by_dist_radius)

def explore():
    pass

def clean():
    pass

def predict():
    pass


def clean_chicago_business_data(self):
    self.logger.info("    Running cleaning steps on raw data")
    self.dataframe = clean_data(self.dataframe, self.data_cleaning)
    return self


def make_chicago_business_features(self):
    for df_list in (self.test_sets, self.train_sets):
        for i in range(len(df_list)):

            self.logger.info("    Creating %s features on test-train set %s", n, i+1)
            df_list[i] = make_features(df_list[i], self.feature_generators)

    return self


def main(config_path):
    with open(config_path, 'rb') as config_file:
        config = yaml.safe_load(config_file.read())

    pipeline = Pipeline(
            Path(config["data"]["imputed_path"]),
            config["pipeline"]["target"],
            data_cleaning=[
                filter_out_2019_data
            ],
            data_preprocessors=[
                hash_string('LEGAL NAME'),
                hash_string('DOING BUSINESS AS NAME'),
                hash_string('ADDRESS'),
                hash_string('LICENSE DESCRIPTION'),
                hash_string('BUSINESS ACTIVITY'),
                hash_string('LICENSE STATUS'),
                hash_string('SSA'),
            ],
            feature_generators=[
                count_by_zip_year,      # num_not_renewed_zip
                count_by_dist_radius,   # num_not_renewed_1km
                make_dummy_vars         # CITY, STATE, APPLICATION TYPE
            ],
            summarize=False,
            model=model,
            name="quick-pipeline-lr-only-" + description,
            output_root_dir=Path("output/"))

    pipeline.clean_data = Methodtype(clean_chicago_business_data, pipeline)
    pipeline.generate_features = MethodType(make_chicago_business_features, pipeline)

    pipeline.run()


if __name__ == "__main__":
    main("config.yml")
