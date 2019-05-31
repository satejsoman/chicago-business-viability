from collections import OrderedDict
from itertools import product

import yaml
from sklearn.ensemble import (BaggingClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def generate_name(model, keys, values):
    return "-".join([model] + ["{}{}".format(k, v) for (k, v) in zip(keys, values)])

class Grid():
    def __init__(self, model_parameters):
        models = {}
        for model in model_parameters:
            models.update(model)
        self.models = models

    # quick grid for testing pipeline works
    @staticmethod
    def default():
        return Grid([
            {"log-reg-{}-c{}".format(p, c)     : LogisticRegression(penalty=p, C=c)          for c in (0.001, 0.1, 1 , 10, 100) for p in ('l1', 'l2')},
            {"knn-k{}".format(k)               : KNeighborsClassifier(n_neighbors=k)         for k in (10, 50, 100)},
            {"decision-tree-{}".format(c)      : DecisionTreeClassifier(criterion=c)         for c in ("gini", "entropy")},
            {"boost-alpha{}".format(a)         : GradientBoostingClassifier(learning_rate=a) for a in (0.1, 0.5, 2.0)},
            {"bagging-sample-frac{}".format(f) : BaggingClassifier(max_samples=f)            for f in (0.1, 0.5, 1.0)},
            {"random-forest"                   : RandomForestClassifier()},
        ])

    @staticmethod
    def from_config(config_dict):
        model_parameters = []
        for (model, params) in config_dict.items():
            constructor = globals()[model]
            model_parameters += [{generate_name(model, params.keys(), vals): constructor(**dict(zip(params.keys(), vals)))} for vals in product(*params.values())]
        return Grid(model_parameters)

    def __repr__(self):
        return self.models.__repr__()

    def __iter__(self):
        return iter(self.models.items())

if __name__ == "__main__":
    with open('config.yaml') as config_file:
        config = yaml.safe_load(config_file)
    for each in Grid.from_config(config["models"]).models.items():
        print(each)