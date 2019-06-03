import datetime
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.core import Pipeline
from pipeline.splitter import Splitter
from pipeline.transformation import replace_missing_with_mean, to_datetime


def get_pipeline():
    splitter = Splitter.from_config({
        "split_column": "date",
        "splits" : [
            {
                "name": "initial",
                "train": {
                    "start": "01/01/2000",
                    "end":   "01/31/2000"
                },
                "test": {
                    "start": "03/01/2000",
                    "end"  : "03/31/2000"
                }
            }, {
                "train": {
                    "start": "04/01/2000",
                    "end":   "04/30/2000"
                },
                "test": {
                    "start": "06/01/2000",
                    "end"  : "06/30/2000"
                }
            }, {
                "name" : "overlapping",
                "train": {
                    "start": "01/01/2000",
                    "end":   "01/31/2000"
                },
                "test": {
                    "start": "01/01/2000",
                    "end"  : "03/31/2000"
                }
            }
        ]
    })

    return Pipeline(
        input_source       = Path(__file__).parent/'test_split_mock_data.csv',
        target             = "label",
        data_cleaning      = [to_datetime("date")],
        data_preprocessors = [replace_missing_with_mean("raw1")],
        feature_generators = [replace_missing_with_mean("raw2")],
        splitter           = splitter,
        verbose            = False)

train_slice_1  = slice(0, 14)
test_slice_1 = slice(24, 43)

train_slice_2  = slice(43, 65)
test_slice_2 = slice(85, None)

train_slice_3  = slice(0, 14)
test_slice_3 = slice(0, 43)

slices = [train_slice_1, train_slice_2, train_slice_3, test_slice_1, test_slice_2, test_slice_3]

class TestTemporalSplits(unittest.TestCase):
    def test_no_imputation_before_split(self):
        pipeline = (get_pipeline().load_data())
        self.assertNotIn("raw1_clean", pipeline.dataframe.columns)
        self.assertNotIn("raw2_clean", pipeline.dataframe.columns)

    def test_splits(self):
        pipeline = (get_pipeline().load_data()
                                  .clean_data()
                                  .generate_test_train())
        df = pipeline.dataframe.copy()

        # test names picked up correctly
        self.assertEqual(pipeline.split_names, ["initial", "split 1", "overlapping"])

        # test generated splits match known split indices
        for (dataset, sl) in zip(pipeline.train_sets + pipeline.test_sets, slices):
            self.assertTrue(dataset.equals(df[sl]))

    def test_data_processing_run_on_test_and_train(self):
        pipeline = (get_pipeline().load_data()
                                  .clean_data()
                                  .generate_test_train()
                                  .preprocess_data()
                                  .generate_features())
        df = pipeline.dataframe

        datasets = pipeline.train_sets + pipeline.test_sets

        # make sure columns generated in test/train sets
        for dataset in datasets:
            self.assertIn("raw1_clean", dataset.columns)
            self.assertIn("raw2_clean", dataset.columns)

        # make sure value imputation occurred per-set
        for col in ("raw1", "raw2"):
            missing_index = df[col].isna()
            col_clean = col+"_clean"
            for (dataset, sl) in zip(datasets, slices):
                imputed_values = dataset[missing_index][col_clean].unique()
                if len(imputed_values) > 0:
                    # some roundoff error during imputation
                    self.assertAlmostEqual(imputed_values[0], df[col][sl][df[col][sl].notna()].mean())

if __name__ == "__main__":
    unittest.main()
