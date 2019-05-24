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

def explore():
    pass 

def clean():
    pass 

def predict():
    pass

def evaluate_models(grid, pipeline_generator):
    evaluations = []
    for (description, model) in grid.models.items():
        pipeline = pipeline_generator(description, model)
        pipeline.run()
        evaluations += pipeline.model_evaluations
    return evaluations

def main(config_path):
    with open(config_path, 'rb') as config_file:
        config = yaml.safe_load(config_file.read())
    
    def model_parametrized_pipeline(description, model):
        return Pipeline(
            Path(config["data"]["imputed_path"]), 
            config["pipeline"]["target"], 
            data_preprocessors=[
                hash_string('LEGAL NAME'),
                hash_string('DOING BUSINESS AS NAME'),
                hash_string('ADDRESS'),
                categorize('CITY'),
                categorize('STATE'),
                hash_string('LICENSE DESCRIPTION'),
                hash_string('BUSINESS ACTIVITY'),
                categorize('APPLICATION TYPE'),
                hash_string('LICENSE STATUS'),
                hash_string('SSA'),
            ],
            features=[
                'LEGAL NAME_hashed',
                'DOING BUSINESS AS NAME_hashed',
                'ADDRESS_hashed',
                'CITY_categorical',
                'STATE_categorical',
                'LICENSE DESCRIPTION_hashed',
                'BUSINESS ACTIVITY_hashed',
                'APPLICATION TYPE_categorical',
                'LICENSE STATUS_hashed',
                'SSA_hashed',
                'LICENSE TERM EXPIRATION DATE',
                'ID',
                'LICENSE ID',
                'ACCOUNT NUMBER',
                'SITE NUMBER',
                'ZIP CODE',
                'WARD',
                'PRECINCT',
                'WARD PRECINCT',
                'POLICE DISTRICT',
                'LICENSE CODE',
                'BUSINESS ACTIVITY ID',
                'LICENSE NUMBER',
                'APPLICATION CREATED DATE',
                'APPLICATION REQUIREMENTS COMPLETE',
                'PAYMENT DATE',
                'CONDITIONAL APPROVAL',
                'LICENSE TERM START DATE',
                'LICENSE APPROVED FOR ISSUANCE',
                'DATE ISSUED',
                'LICENSE STATUS CHANGE DATE',
                'LATITUDE',
                'LONGITUDE'],
            summarize=False,
            model=model,
            name="quick-pipeline-lr-only-" + description,
            output_root_dir=Path("output/"))
    evaluations = evaluate_models(Grid.from_config(config["models"]), model_parametrized_pipeline)
    pd.DataFrame(evaluations).to_csv("evaluations.csv")

if __name__ == "__main__":
    main("config.yaml")
