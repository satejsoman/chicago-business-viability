import numpy as np
import pandas as pd


def clean_data(input_df, cleaning_steps):
    '''
    Input:  input_df - input pandas DataFrame to apply cleaning steps to
            cleaning_steps - list of functions to apply to input_df
    Output: result_df - input_df after applying all functions in cleaning_steps
    '''
    result_df = input_df.copy(deep=True)
    for step in cleaning_steps:
        result_df = step(result_df)
    return result_df


def filter_out_2019_data(input_df):
    '''
    Filters out all licenses issued on Jan 1, 2019 onwards.
    '''
    df = input_df.loc[df['DATE ISSUED'] <= pd.to_datetime('12/31/2018')]
    return df
