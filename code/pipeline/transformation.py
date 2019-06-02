import pandas as pd
import numpy as np


class Transformation:
    def __init__(self, name, input_column_names, function, output_column_name=None):
        self.name = name
        self.input_column_names = input_column_names
        self.output_column_name = output_column_name if output_column_name else name
        self.function = function

    def __call__(self, dataframe):
        return self.function(dataframe)

    def __repr__(self):
        return self.name

def hash_string(column):
    input_col  = column
    output_col = column + "_hashed"
    return Transformation(
        name="hashed-" + input_col,
        input_column_names=[input_col],
        output_column_name=output_col,
        function=lambda df: df[input_col].apply(hash))

def categorize(column):
    input_col  = column
    output_col = column + "_categorical"
    return Transformation(
        name="categorize-" + input_col,
        input_column_names=[input_col],
        output_column_name=output_col,
        function=lambda col: col[input_col].astype("category").cat.codes)

def binarize(column, true_value):
    input_col  = column
    output_col = column + "_binary"
    return Transformation(
        name="binarize-" + input_col,
        input_column_names=[input_col],
        output_column_name=output_col,
        function=lambda df: np.where(df[input_col] == true_value, 1, 0))

def replace_missing_with_value(column, value):
    input_col  = column
    output_col = column + "_clean"
    def replace(dataframe):
        return dataframe[input_col].fillna(value)

    return Transformation(
        name="replace-missing-values-with-value({},{})".format(column, value),
        input_column_names=[input_col],
        output_column_name=output_col,
        function=replace)

def scale_by_max(column):
    input_col  = column
    output_col = column + "_scaled"
    def scale(dataframe):
        return dataframe[input_col]/(dataframe[input_col].max())

    return Transformation(
        name="scale-by-max({})".format(column),
        input_column_names=[input_col],
        output_column_name=output_col,
        function=scale)

def replace_missing_with_mean(column):
    input_col  = column
    output_col = column+"_clean"
    def replace(dataframe):
        avg = dataframe[input_col].mean()
        return dataframe[input_col].fillna(avg)

    return Transformation(
        name="replace-missing-values-with-mean({})".format(column),
        input_column_names=[input_col],
        output_column_name=output_col,
        function=replace)


def to_datetime(column):
    return Transformation(
        name = "convert-" + column + "-to-datetime",
        input_column_names=column,
        output_column_name=column,
        function = pd.to_datetime)
