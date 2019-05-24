import datetime
import logging
import os
import shutil
import sys
import uuid
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score)

from .utils import get_git_hash, get_sha256_sum


class Pipeline:
    def __init__(self, 
        csv_path,
        target,
        summarize=False,
        data_preprocessors=None,
        feature_generators=None,
        model_grid=None,
        name=None,
        output_root_dir=".", 
        features=None):
        self.csv_path = csv_path 
        self.target = target
        self.summarize = summarize
        self.data_preprocessors = data_preprocessors
        self.feature_generators = feature_generators
        self.model_grid = model_grid

        self.dataframe = None

        if not name:
            self.name = "ML Pipeline"
        else:
            self.name = name

        self.all_columns_are_features = False
        if not features:
            if not feature_generators: #assume all columns are features
                self.all_columns_are_features = True
            self.features = []
        else:
            self.features = features

        self.train_sets = []
        self.test_sets  = []

        self.trained_models    = OrderedDict()
        self.model_evaluations = []

        self.output_root_dir = Path(output_root_dir)

        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        if not self.logger.hasHandlers():
            self.logger.addHandler(logging.StreamHandler(sys.stdout))
    
    def load_data(self):
        self.logger.info("Loading data")
        self.dataframe = pd.read_csv(self.csv_path)
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
            self.dataframe[transformation.output_column_name] = transformation(self.dataframe[transformation.input_column_names])
            generated_columns.append(transformation.output_column_name)
        self.logger.info("")
        if purpose == "feature generation":
            self.feature_generators = list(set(self.features + generated_columns))
        
        return self

    def preprocess_data(self):
        return self.run_transformations(self.data_preprocessors, purpose="preprocessing")

    def generate_features(self):
        return self.run_transformations(self.feature_generators, purpose="feature generation")

    def generate_test_train(self):
        self.logger.info("Columns: %s", self.dataframe.columns)
        self.train_sets = [{"X" : self.dataframe[self.features], "y": self.dataframe[self.target]}]
        self.test_sets  = [{"X" : self.dataframe[self.features], "y": self.dataframe[self.target]}]
        return self

    def run_model(self, description, model):
        self.logger.info("    Training model %s", description)
        n = len(self.train_sets)
        for (index, train_set) in enumerate(self.train_sets):
            self.logger.info("        Training on training set (%s/%s)", index + 1, n)
            if description in self.trained_models.keys():
                self.trained_models[description]+= [model.fit(**train_set)]
            else:
                self.trained_models[description] = [model.fit(**train_set)]
        return self

    def evaluate_models(self, description, models):
        X, y = "X", "y"
        thresholds = [1, 2, 5, 10, 20, 20, 50]
        self.logger.info("    Evaluating model %s", description)
        n = len(self.test_sets)
        for (index, (model, test_set)) in enumerate(zip(models, self.test_sets)):
            self.logger.info("        Evaluating on testing set (%s/%s)", index + 1, n)
            score = model.score(**test_set)
            y_true = test_set[y]
            y_score = np.array([_[1] for _ in model.predict_proba(test_set[X])])
            auc_roc = roc_auc_score(y_true, y_score)
            evaluation = {
                "name"             : description,
                "test_train_index" : index + 1,
                "score"            : score,
                "auc_roc"          : auc_roc
            }

            for k in thresholds:
                report = classification_report(y_true, apply_threshold(k/100.0, y_score), output_dict=True)
                evaluation.update({
                    metric + "-" + str(k): value 
                    for (metric, value) 
                    in report['1.0'].items()})

            self.model_evaluations.append(evaluation)
            self.logger.info("    Model score: %s", score)

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
        self.logger.info("Input data: %s (SHA-256: %s)", self.csv_path.resolve(), get_sha256_sum(self.csv_path))
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
            .summarize_data()
            .preprocess_data()
            .generate_features()
            .generate_test_train()
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

# utils 
def apply_threshold(threshold, scores):
    return np.where(scores > threshold, 1, 0)
