#==============================================================================#
# USEFUL FUNCTIONS FOR READING, CLEANING, AND PREPARING DATA
# CAPP 30254 - MACHINE LEARNING FOR PUBLIC POLICY
#
# Cecile Murray
#==============================================================================#

# dependencies
import numpy as np
import pandas as pd 
import seaborn as sns
import plotnine as p9

#==============================================================================#
# READ IN DATA
#==============================================================================#


# TO DO: implement additional options
def read_data(filename, file_type):
    '''Takes string name of csv or excel file (for now) and returns pandas data frame 
        If a data dict is provided it will rename vars'''
    
    if file_type == "csv":
        return pd.read_csv('data/' + filename + '.csv')
    elif file_type == "excel": 
        return pd.read_excel('data/' + filename + '.xlsx')
    else:
        print("filetype not supported")
        return



#==============================================================================#
# PROCESS DATA
#==============================================================================#

def find_cols_with_missing(df):
    ''' Returns columns that contain missing values with count of missing'''

    return df.isna().sum()[df.isna().sum() > 0]


# some pre-processing to convert numeric NA's to the median of the column
def replace_missing(df, method = 'mean', *args):
    ''' Takes data frame, replacement method, and arbitrary column names;
        replaces missing values using specified method;
        returns data frame '''

    if method == 'median': 
        for c in list(args):
            df[c] = df[c].fillna(df[c].median())
    
    elif method == "mean":
        for c in list(args):
            df[c] = df[c].fillna(df[c].mean())

    elif method == "zero":
        for c in list(args):
            df[c] = df[c].fillna(0)

    else:
        pass
    
    return df


def convert_to_boolean(df, cols, true_val, false_val):
    ''' Takes data frame, column name/list of names, and values for True/False
        Returns: data frame with that column converted to boolean

        Note: true/false vals must be the same for all columns
    '''

    if isinstance(cols, str):
        cols = [cols]

    for col in cols:
        df[col] = np.where(df[col] == true_val, True, False)
    
    return df

#==============================================================================#
# GENERATE FEATURES
#==============================================================================#

def bin_continuous(df, var, new_varname, breaks, labels = False):
    ''' Convert continuous variable to discrete/categorical '''


    # handle case where upper bound less than max of variable
    if breaks[-1] < df[var].max():
        breaks.append(df[var].max())
        
    # handle case where lower bound greater than min of variable
    if breaks[0] > df[var].min():
        breaks.insert(0, df[var].min())

    if labels:
        df[new_varname] = pd.cut(df[var], np.array(breaks), labels = labels)
    else:
        df[new_varname] = pd.cut(df[var], np.array(breaks))
    
    return df


def make_cat_dummy(df, cols):
    '''Convert categorical variable or list of categorical variables into binary '''

    return pd.get_dummies(df, dummy_na=True, columns=cols)

#==============================================================================#
# MISC
#==============================================================================#

def move_last_col_to_front(df):
    ''' Takes the rightmost column in a data frame and makes it the left-most '''

    cols = list(df.columns)
    return df[[cols[-1]] + cols[:-1]]