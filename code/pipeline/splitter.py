from collections import namedtuple
import yaml
import datetime 

Bounds = namedtuple('Bounds', ["start", "end"])

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
        for (train, test) in zip(train_splits, test_splits):
            train_sets.append(df[df.where((train.start <= df[self.column]) & (df[self.column] <= train.end))])
            test_sets.append(df[df.where((test.start  <= df[self.column]) & (df[self.column] <= test.end))])
        return (train_sets, test_sets, self.names)

    @staticmethod
    def from_config(config_dict):
        column = config_dict["split_column"]
        train_splits, test_splits, names = [], [], []
        for split in config_dict["splits"]:
            train_splits.append(Bounds(**parse_datetimes(split["train"])))
            test_splits.append(Bounds(**parse_datetimes(split["test"])))
            names.append(split["name"])

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