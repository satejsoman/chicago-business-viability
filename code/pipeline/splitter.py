import datetime

import yaml
from pandas import date_range

date_format = "%m/%d/%Y"

def parse_datetimes(split):
    return {k:datetime.datetime.strptime(v, date_format) for (k, v) in split.items()}

class Splitter():
    def __init__(self, column, train_splits, test_splits, split_names):
        self.column = column
        self.train_splits = train_splits
        self.test_splits = test_splits
        self.split_names = split_names

    def split(self, df):
        train_sets, test_sets = [], []
        for (train, test) in zip(self.train_splits, self.test_splits):
            train_sets.append(df[df[self.column].isin(train)])
            test_sets.append(df[df[self.column].isin(test)])
        return (train_sets, test_sets, self.split_names)

    @staticmethod
    def from_config(config_dict):
        column = config_dict["split_column"]
        train_splits, test_splits, names = [], [], []
        for (i, split) in enumerate(config_dict["splits"]):
            train_splits.append(date_range(**parse_datetimes(split["train"])))
            test_splits.append(date_range(**parse_datetimes(split["test"])))
            names.append(split.get("name", "split " + str(i)))

        return Splitter(
            column       = column,
            train_splits = train_splits,
            test_splits  = test_splits,
            split_names  = names)

    def __iter__(self):
        return iter(zip(self.split_names, self.train_splits, self.test_splits))

    def __repr__(self):
        return repr((self.column, repr(list(iter(self)))))

if __name__ == "__main__":
    with open('config.yaml') as config_file:
        config = yaml.safe_load(config_file)
    splitter = Splitter.from_config(config["pipeline"]["test_train"])
    print(splitter)
