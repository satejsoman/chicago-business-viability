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
from chicago_business import get_pipeline

def compute_annual_failure_rate(self):
    pass

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help = "config file path")
    args = parser.parse_args()

    # pipeline = get_pipeline(args.config)
    
    pipeline = (get_pipeline("config.yml")
                 .load_data()
                 .clean_data()
                 .summarize_data()
                 .generate_test_train()
                 .preprocess_data()
                 .generate_features())
    

