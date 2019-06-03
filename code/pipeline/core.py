import datetime
import logging
import os
import shutil
import sys
import uuid
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from .utils import get_git_hash, get_sha256_sum
from .evaluation import evaluate, score_function_overrides

DEFAULT_K_VALUES = [_/100.0 for _ in [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90]]

class Pipeline:
    def __init__(self,
                input_source,
                target,
                name="ML Pipeline",
                summarize=False,
                data_cleaning=None,
                data_preprocessors=None,
                feature_generators=None,
                features=None,
                model_grid=None,
                splitter=None,
                positive_label=1,
                k_values=None,
                output_root_dir=".",
                verbose=True):
        warnings.filterwarnings("ignore")

        self.name = name

        self.in_memory = not isinstance(input_source, Path)

        self.input_source = input_source
        self.target = target
        self.summarize = summarize
        self.data_cleaning = data_cleaning
        self.data_preprocessors = data_preprocessors
        self.feature_generators = feature_generators
        self.model_grid = model_grid
        self.splitter = splitter
        self.positive_label = positive_label

        self.dataframe = None

        self.all_columns_are_features = False
        if not features:
            if not feature_generators: #assume all columns are features
                self.all_columns_are_features = True
            self.features = []
        else:
            self.features = features

        self.train_sets   = []
        self.test_sets    = []
        self.split_names  = []

        self.trained_models    = OrderedDict()
        self.model_evaluations = []

        self.output_root_dir = Path(output_root_dir)

        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        if verbose and not self.logger.hasHandlers():
            self.logger.addHandler(logging.StreamHandler(sys.stdout))
        else:
            self.logger.addHandler(logging.NullHandler())

        self.k_values = k_values if k_values else DEFAULT_K_VALUES

    def load_data(self):
        self.logger.info("Loading data")
        if self.in_memory:
            self.dataframe = self.input_source
        else:
            self.dataframe = pd.read_csv(self.input_source)
        if self.all_columns_are_features:
            self.features = list([col for col in self.dataframe.columns if col != self.target])
        return self

    def summarize_data(self):
        if self.summarize:
            self.logger.info("Summarizing data.")
            self.dataframe.describe().to_csv(self.output_dir/"summary.csv")
            self.dataframe.corr().to_csv(self.output_dir/"correlation.csv")
        return self

    def run_transformations(self, transformations, purpose=None):
        if not transformations:
            return self
        self.logger.info("")
        self.logger.info("Running transformations %s", ("for " + purpose) if purpose else "")
        n = len(transformations)
        generated_columns = []
        for (i, transformation) in enumerate(transformations):
            self.logger.info("    Applying transformation (%s/%s): %s ",  i+1, n, transformation.name)
            self.logger.info("    %s -> %s", transformation.input_column_names, transformation.output_column_name)
            if not self.test_sets or purpose == "cleaning":
                self.dataframe[transformation.output_column_name] = transformation(self.dataframe[transformation.input_column_names])
            else:
                for dataset in (self.test_sets, self.train_sets):
                    for dataframe in dataset:
                        dataframe[transformation.output_column_name] = transformation(dataframe[transformation.input_column_names])
                generated_columns.append(transformation.output_column_name)
        self.logger.info("")
        if purpose == "feature generation":
            self.features = list(set(self.features + generated_columns))

        return self

    def clean_data(self):
        return self.run_transformations(self.data_cleaning, purpose="cleaning")

    def preprocess_data(self):
        return self.run_transformations(self.data_preprocessors, purpose="preprocessing")

    def generate_features(self):
        return self.run_transformations(self.feature_generators, purpose="feature generation")

    def generate_test_train(self):
        if self.splitter is None:
            return self.test_train_all()
        self.train_sets, self.test_sets, self.split_names = self.splitter.split(self.dataframe)

        return self

    def test_train_all(self):
        self.logger.info("Using entire dataset for training and testing.")
        self.logger.info("Columns: %s", self.dataframe.columns)
        self.train_sets = [self.dataframe]
        self.test_sets  = [self.dataframe]
        self.split_names = ["entire dataset"]
        return self

    def run_model(self, description, model):
        self.logger.info("    Training model %s", description)
        n = len(self.train_sets)
        for (index, (split_name, train_set)) in enumerate(zip(self.split_names, self.train_sets)):
            self.logger.info("        Training on training set \"%s\" (%s/%s)", split_name, index + 1, n)
            if description in self.trained_models.keys():
                self.trained_models[description]+= [model.fit(X=train_set[self.features], y=train_set[self.target])]
            else:
                print(train_set)
                self.trained_models[description] = [model.fit(X=train_set[self.features], y=train_set[self.target])]
        return self

    def evaluate_models(self, description, models):
        X, y = self.features, self.target

        self.logger.info("    Evaluating model %s", description)
        n = len(self.test_sets)
        for (index, (model, split_name, test_set)) in enumerate(zip(models, self.split_names, self.test_sets)):
            self.logger.info("        Evaluating on testing set \"%s\" (%s/%s):", split_name, index + 1, n)
            score = model.score(X=test_set[X], y=test_set[y])
            y_true = test_set[y]
            if type(model) in score_function_overrides.keys():
                score_function = score_function_overrides[type(model)]
                y_score = score_function(self=model, X=test_set[X])
            else:
                y_score = np.array([_[self.positive_label] for _ in model.predict_proba(test_set[X])])

            evaluation, (precision, recall, _) = evaluate(self.positive_label, self.k_values, y_true, y_score)
            evaluation["name"] = description
            evaluation["test_train_index"] = index + 1

            # save raw PR data
            pd.DataFrame({"precision": precision, "recall": recall}).to_csv(self.output_dir/(description+"_pr-data_" + str(index + 1) + ".csv"))
            self.model_evaluations.append(evaluation)

            # save PR curve for model as a figure
            plt.figure()
            plt.step(recall, precision, color='b', alpha=0.2, where='post')
            plt.fill_between(recall, precision, alpha=0.2, color='b')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('2-class Precision-Recall curve')
            png_filename = description + "_" + str(index + 1) + "_prcurve.png"
            svg_filename = description + "_" + str(index + 1) + "_prcurve.svg"
            plt.savefig(self.output_dir/png_filename, dpi=300)
            plt.savefig(self.output_dir/svg_filename, dpi=300)
        return self

    def run_model_grid(self):
        if self.model_grid is None:
            return self
        self.logger.info("Training models.")
        self.logger.info("Features: %s", self.features)
        self.logger.info("Fitting: %s", self.target)
        for (description, model) in self.model_grid:
            self.run_model(description, model)
        return self

    def evaluate_model_grid(self):
        if self.model_grid is None:
            return self
        self.logger.info("Testing models.")
        for (description, models) in self.trained_models.items():
            self.evaluate_models(description, models)
        pd.DataFrame(self.model_evaluations).to_csv(self.output_dir/"evaluations.csv")
        return self

    def run(self):
        run_id = str(uuid.uuid4())
        self.output_dir = self.output_root_dir/(self.name + "-" + run_id)
        if not self.output_dir.exists():
            os.makedirs(self.output_dir)

        run_handler = logging.FileHandler(self.output_dir/"pipeline.run")
        self.logger.addHandler(run_handler)

        self.logger.info("Starting pipeline %s (%s) at %s", self.name, run_id, datetime.datetime.now())
        if self.in_memory:
            self.logger.info("Input data: (in-memory dataframe)")
        else:
            self.logger.info("Input data: %s (SHA-256: %s)", self.input_source.resolve(), get_sha256_sum(self.input_source))
        self.logger.info("Pipeline library version: %s", get_git_hash())
        self.logger.info("")
        self.logger.info("Pipeline settings:")
        self.logger.info("    summarize: %s", self.summarize)
        self.logger.info("    data_preprocessors: %s", self.data_preprocessors)
        self.logger.info("    feature_generators: %s", self.feature_generators)
        self.logger.info("    models: %s", self.model_grid)
        self.logger.info("    name: %s", self.name)
        self.logger.info("    output_root_dir: %s", self.output_root_dir.resolve())
        self.logger.info("")

        (self
            .load_data()
            .clean_data()
            .summarize_data()
            .generate_test_train()
            .preprocess_data()
            .generate_features()
            .run_model_grid()
            .evaluate_model_grid()
        )

        self.logger.info("Copying artifacts to stable path")
        latest_dir = self.output_root_dir/(self.name + "-LATEST")
        if latest_dir.exists():
            shutil.rmtree(latest_dir)
        shutil.copytree(self.output_dir, latest_dir)

        self.logger.info("Finished at %s", datetime.datetime.now())
        self.logger.removeHandler(run_handler)
