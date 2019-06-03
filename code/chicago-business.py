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
                                     scale_by_max, to_datetime)
from pipeline.grid import Grid

from feature_generation import (make_features, reshape_and_create_label,
                                count_by_zip_year, count_by_dist_radius)

def explore():
    pass

def clean():
    pass

def predict():
    pass


def make_chicago_business_features(self):
    for df_list in (self.test_sets, self.train_sets):
        for i in range(len(df_list)):
            self.logger.info("    Creating features on test-train set %s", i+1) 
            # Cecile deleted n in the above log line because she can't tell what it's supposed to be
            df_list[i] = make_features(df_list[i], self.feature_generators)
            print(self.feature_generators)

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
    with open(config_path, 'rb') as config_file:
        config = yaml.safe_load(config_file.read())

    pipeline = Pipeline(  
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
        model_grid=config["models"],
        name="quick-pipeline-lr-only-" + config["description"],
        output_root_dir=Path("output/"))
    pipeline.generate_features = MethodType(make_chicago_business_features, pipeline)
    pipeline.run()

if __name__ == "__main__":
    main("config.yml")
