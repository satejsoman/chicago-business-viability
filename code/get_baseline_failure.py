import configparser
import datetime
import os
from itertools import cycle
from pathlib import Path
from types import MethodType
import argparse
import matplotlib
import matplotlib2tikz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from chicago_business import get_config, get_pipeline

def compute_annual_failure_rate(pipeline):
    '''
    Takes pipeline object to get test and train inputs before modeling
    Computes the failure rate per year
    '''

    dfs = []
    cols = ['ACCOUNT NUMBER', 'SITE NUMBER', 'YEAR', 'not_renewed_2yrs']

    for df in [pipeline.train_sets[0], pipeline.test_sets[0]]:

        rv = df[cols] \
                .groupby(['YEAR', pipeline.target])['ACCOUNT NUMBER'] \
                .count() \
                .unstack() \
                .reset_index()

        rv['share_not_renewed'] = rv[1] / (rv[0] + rv[1])
        dfs.append(rv)

    return pd.concat(dfs)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help = "config file path", default = "final-prediction-config.yml")
    args = parser.parse_args()

    config, script_dir = get_config("final-prediction-config.yml")

    pipeline = (get_pipeline(config, script_dir)
                 .load_data()
                 .clean_data()
                 .summarize_data()
                 .generate_test_train()
                 .preprocess_data()
                 .generate_features())
    
    baseline_rates = compute_annual_failure_rate(pipeline)
    baseline_rates.to_csv("baseline_failure_rates2.csv")
    
    

