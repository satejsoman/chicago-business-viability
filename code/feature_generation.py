import itertools
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances

MERGE_KEYS = ['ACCOUNT NUMBER', 'SITE NUMBER', 'YEAR']

def make_features(input_df, feature_generators, existing_features):
    '''
    Takes an input license-level dataframe and a list of feature generation
    functions to apply, and returns a df of transformed business-year data.
    make_features() will be called on every test and train dataset separately.

    Input:  input_df - raw licence-level data, after test-train split
            feature_generators - a list of functions to apply to each test
                or train df that return the transformed df
    Output: result_df - df of business-year data with all features created
    '''

    base = reshape_and_create_label(input_df) # business-year data

    generated_features = base.copy(deep=True) # accumulate features
    for feature_generator in feature_generators:
        print("        Applying function:", feature_generator.__name__)
        feature = feature_generator(base, input_df)
        generated_features = generated_features.merge(feature,
            how='left', on=MERGE_KEYS)

    # Merge on all generated feature onto the base
    result_df = base \
        .drop(labels=['not_renewed_2yrs'], axis=1) \
        .merge(generated_features,
               how='left',
               on=MERGE_KEYS)

    # Merge on existing features
    result_df["JOIN_YEAR"] = result_df["YEAR"] - 1

    # Get unique account-site level data for existing features
    existing_df = input_df[MERGE_KEYS + existing_features].drop_duplicates()
    full_result = result_df \
        .drop(labels=['YEAR'], axis=1) \
        .merge(existing_df,
            how='left',
            right_on=MERGE_KEYS,
            left_on=['ACCOUNT NUMBER', 'SITE NUMBER', 'JOIN_YEAR']) \
        .drop(labels=['YEAR'], axis=1)

    # Fix year indexing
    full_result['YEAR'] = full_result["JOIN_YEAR"] + 1
    full_result = full_result.drop(labels=['JOIN_YEAR'], axis=1)

    return full_result


# Reshape license data to business-year data, create label
def reshape_and_create_label(input_df):
    '''
    Processes raw business license-level dataframe into account-site-year level
     dataframe. Extracts years from min/max year and expands dataframe into
     account-site-year level.

    Currently hardcoded to require columns for ACCOUNT NUMBER, SITE NUMBER,
     DATE ISSUED, and LICENSE TERM EXPIRATION DATE

    Input:  input_df - license-level dataframe
    Output: result_df - business-year-level df with not_renewed_2yrs label
    '''

    # Aggregate by account-site and get min/max/expiry dates for licenses
    df = input_df.copy(deep=True) \
        .groupby(['ACCOUNT NUMBER', 'SITE NUMBER']) \
        .agg({'DATE ISSUED': ['min', 'max'],
              'LICENSE TERM EXPIRATION DATE': 'max'}) \
        .reset_index(col_level=1)

    # Flatten column names into something usable
    df.columns = df.columns.to_flat_index()

    df = df.rename(columns={
        ('', 'ACCOUNT NUMBER'): "account",
        ('' , 'SITE NUMBER'): 'site',
        ('DATE ISSUED', 'min'): 'min_license_date',
        ('DATE ISSUED', 'max'): 'max_license_date',
        ('LICENSE TERM EXPIRATION DATE', 'max'): 'expiry'})

    # Extract min/max license dates into list of years_open
    df['years_open'] = pd.Series(map(lambda x, y: [z for z in range(x, y+2)],
                                     df['min_license_date'].dt.year,
                                     df['max_license_date'].dt.year))

    # make account-site id var
    # melt step below doesn't work well without merging these two cols
    df['account_site'] = df['account'].astype('str') + "-" + df['site'].astype('str')
    df = df[df.columns.tolist()[-1:] + df.columns.tolist()[:-1]]
    df = df.drop(labels=['account', 'site'], axis=1)

    # Expand list of years_open into one row for each account-site-year
    # https://mikulskibartosz.name/how-to-split-a-list-inside-a-dataframe-cell-into-rows-in-pandas-9849d8ff2401
    df = df \
        .years_open \
        .apply(pd.Series) \
        .merge(df, left_index=True, right_index=True) \
        .drop(labels=['years_open'], axis=1) \
        .melt(id_vars=['account_site', 'min_license_date', 'max_license_date', 'expiry'], value_name='YEAR') \
        .drop(labels=['variable'], axis=1) \
        .dropna() \
        .sort_values(by=['account_site', 'YEAR'])

    # Split account_site back into ACCOUNT NUMBER, SITE NUMBER
    df['ACCOUNT NUMBER'], df['SITE NUMBER'] = df['account_site'].str.split('-', 1).str
    df['ACCOUNT NUMBER'] = df['ACCOUNT NUMBER'].astype('int')
    df['SITE NUMBER'] = df['SITE NUMBER'].astype('int')

    # reorder columns
    df['YEAR'] = df['YEAR'].astype('int')
    df = df[['ACCOUNT NUMBER', 'SITE NUMBER', 'account_site', 'YEAR',
             'min_license_date', 'max_license_date', 'expiry']] \
        .sort_values(by=['ACCOUNT NUMBER', 'SITE NUMBER'])

    grouping = df.groupby(["ACCOUNT NUMBER", "YEAR"])["SITE NUMBER"].count()
    df["num_sites"] = df.apply(lambda row: grouping[row["ACCOUNT NUMBER"], row["YEAR"]], axis=1)

    # Assume buffer period is last 2 years of input data
    threshold_year = input_df['DATE ISSUED'].dt.year.max() - 1
    buffer_df = input_df.loc[input_df['DATE ISSUED'].dt.year >= threshold_year]

    # Get list of account-site numbers in buffer
    buffer_ids = buffer_df['ACCOUNT NUMBER'].astype('str') \
        + '-' + buffer_df['SITE NUMBER'].astype('str')

    # Generate label
    # If expiry before buffer, label = 1 if year is 1 year after last license,
    #   else 0
    # If expiry within or after buffer, AND buffer contains renewal, label = 0,
    #   else 0 except for year after last license
    df['not_renewed_2yrs'] = np.where(df['expiry'].dt.year < threshold_year,
        np.where(df['YEAR'] >= df['max_license_date'].dt.year + 1, 1, 0),
        np.where(df['account_site'].isin(buffer_ids),
            0,
            np.where(df['YEAR'] >= df['max_license_date'].dt.year + 1, 1, 0)
            )
        )
    # Drop unnecessary columns
    # Drop all years that we can't predict on, i.e. buffer years onwards
    df = df.drop(labels=['account_site', 'min_license_date','max_license_date',
                         'expiry'], axis=1) \
        .loc[df['YEAR'] < threshold_year] \
        .reset_index(drop=True)
    return df


# Get unique addresses
def get_locations(input_df):
    '''
    Takes license-level data and returns a dataframe with location attributes
        for each account-site.

    Input:  input_df - license-level data with specified location columns.
    Output: df - unique addresses for each account-site.
    '''
    # Columns to return
    LOCATION_COLS = ['ACCOUNT NUMBER', 'SITE NUMBER', 'ADDRESS', 'CITY',
                     'STATE', 'ZIP CODE', 'LATITUDE', 'LONGITUDE', "which_ssa"]

    # Drop rows if these columns have NA
    NA_COLS = ['LATITUDE', 'LONGITUDE']

    df = input_df.copy(deep=True)[LOCATION_COLS] \
        .dropna(subset=NA_COLS) \
        .drop_duplicates() \
        .sort_values(by=['ACCOUNT NUMBER', 'SITE NUMBER']) \
        .reset_index(drop=True)

    return df


# Count nonrenewals by zipcode
def count_by_zip_year(input_df, license_data):
    '''
    Takes business-year-level data from reshape_and_create_label(), counts the
    number of nonrenewals in the same zipcode-year, and returns the original
    df with num_not_renewed_zip.

    Currently hardcoded to require columns for ACCOUNT NUMBER, SITE NUMBER,
     YEAR, and ZIP CODE.

    Input:  input_df - business-year level data with label.
            license_data - licence-level data that provides locational info
    Output: results_df - input_df with num_not_renewed_zip appended.
    '''

    # print("Before address merge", input_df.shape)

    # Get locations from license data and merge onto business-year data
    addresses = get_locations(license_data)
    df = input_df.copy(deep=True) \
        .merge(addresses, how='left', on=['ACCOUNT NUMBER', 'SITE NUMBER'])

    # print("After address merge", df.shape)

    # Setting and resetting index serves the purpose of expanding rows to all
    #   years for each zipcode in the data, then filling the "missing" rows
    #   with a count of 0. This lets us handle implicit imputation here, then
    #   merge it onto the original data without introducing NAs at that stage.
    counts_by_zip = df.loc[df['not_renewed_2yrs'] == 1] \
        .groupby(['ZIP CODE', 'YEAR']).size().reset_index() \
        .set_index(['ZIP CODE', 'YEAR']) \
        .reindex(pd.MultiIndex.from_tuples(
            itertools.product(df['ZIP CODE'].unique(), df['YEAR'].unique())
        )) \
        .reset_index() \
        .rename(columns={'level_0': 'ZIP CODE',
                         'level_1': 'YEAR',
                         0: 'num_not_renewed_zip'}) \
        .fillna(0) \
        .sort_values(by=['ZIP CODE', 'YEAR'])

    # Change from float to int
    counts_by_zip['num_not_renewed_zip'] = \
        counts_by_zip['num_not_renewed_zip'].astype('int')

    # Merge zip-year level data onto base
    results_df = df[['ACCOUNT NUMBER', 'SITE NUMBER', 'YEAR', 'ZIP CODE']] \
        .merge(counts_by_zip, how='left', on=['ZIP CODE', 'YEAR']) \
        .drop(labels=['ZIP CODE'], axis=1) \
        .fillna(0) \
        .sort_values(by=MERGE_KEYS)

    return results_df


# Count nonrenewals by distance radius
def count_by_dist_radius(input_df, license_data):
    '''
    Counts the number of business nonrenewals within a specified distance in km
    for each business-year.

    Currently hardcoded to require columns for ACCOUNT NUMBER, SITE NUMBER,
     YEAR, LATITUDE, and LONGITUDE.

    Input: input_df - business-year level data with label.
    Output: results_df - input df with num_not_renewed_1km appended.
    '''

    # Get locations from license data and merge onto business-year data
    addresses = get_locations(license_data)
    df = input_df.copy(deep=True) \
        .merge(addresses, how='left', on=['ACCOUNT NUMBER', 'SITE NUMBER'])

    # Select columns, transforms lat/long in degrees to radians
    df = df[['ACCOUNT NUMBER', 'SITE NUMBER', 'YEAR', 'LATITUDE', 'LONGITUDE',
             'not_renewed_2yrs']]
    df['LATITUDE_rad'] = np.radians(df['LATITUDE'])
    df['LONGITUDE_rad'] = np.radians(df['LONGITUDE'])
    R = 6371 # circumference of the Earth in km

    year_dfs = []
    years = df['YEAR'].unique()

    for i in sorted(df['YEAR'].unique()):
        year_df = df.loc[df['YEAR'] == i]
        fails_only = year_df.loc[year_df['not_renewed_2yrs'] == 1]

        # if no businesses failed that year, return a count of 0 for all
        if len(fails_only) == 0:
            year_df[0] = np.zeros(len(year_df)).astype('int')
            year_df = year_df \
                .reset_index(drop=True) \
                .drop(labels=['LATITUDE', 'LONGITUDE', 'LATITUDE_rad',
                              'LONGITUDE_rad', 'not_renewed_2yrs'], axis=1)
        else:
            # Get pairwise distance between all businesses that year and all
            # nonrenewals that year. Then count number of nonrenewals within
            # threshold distance (using row-wise sum) and join back on year_df
            dist_df = R * haversine_distances(
                year_df[['LATITUDE_rad', 'LONGITUDE_rad']],
                fails_only[['LATITUDE_rad', 'LONGITUDE_rad']]
            )
            dist_df = pd.DataFrame(np.where(dist_df <= 1, 1, 0).sum(axis=1))
            year_df = year_df \
                .reset_index(drop=True) \
                .join(dist_df) \
                .drop(labels=['LATITUDE', 'LONGITUDE', 'LATITUDE_rad',
                              'LONGITUDE_rad', 'not_renewed_2yrs'], axis=1)

        year_dfs.append(year_df)
    # Concatenate all year-specific dfs to get counts for all business-years
    # Then merge onto original df by business-year id cols
    all_years_df = pd.concat(year_dfs)
    results_df = input_df.merge(all_years_df,
                                how='left',
                                on=MERGE_KEYS) \
        .rename(columns={0: 'num_not_renewed_1km'}) \
        [['ACCOUNT NUMBER', 'SITE NUMBER', 'YEAR', 'num_not_renewed_1km' ]]

    return results_df


def make_dummy_vars(base, license_data):
    '''
    Wrapper for the pandas get_dummies() method. Takes a pandas DataFrame and
    a string variable label as inputs, and returns a new DataFrame with new
    binary variables for every unique value in var.
    Inputs: df - pandas DataFrame
    Output: new_df - pandas DataFrame with new variables named "[var]_[value]"
    '''
    VARS_TO_DUMMIFY = ['CITY', 'STATE', "which_ssa"]
    base_cols = base.columns.tolist()

    # Get locations from license data and merge onto business-year data
    addresses = get_locations(license_data)
    df = base.merge(addresses, how='left', on=['ACCOUNT NUMBER', 'SITE NUMBER'])

    # Select only relevant features to dummify
    # ACCOUNT NUM, SITE NUM, YEAR, not_renewed_2yrs, CITY, STATE
    df = df[base_cols + VARS_TO_DUMMIFY]
    new_df = pd.get_dummies(df, columns=VARS_TO_DUMMIFY, dtype=np.int64) \
        .drop(labels=['not_renewed_2yrs'], axis=1)

    return new_df


# Balance features between test and train sets
def balance_features(train_df, test_df):
    '''
    Making dummy variables from categorical features may result in different
    dummy features occuring between test and training sets if different values
    are present. This takes 2 dataframes, checks for feature balance, then
    applies the following:

    1. If a feature is in train but not in test, its value did not occur in the
    categorical feature in test and can be added as 0s to test.
    2. If a feature is in test but not in train, a classifier would not have
    been trained on it and it can be safely dropped from test.

    Inputs: train_df - pandas Dataframe of training data
            test_df - pandas Dataframe of test data
    Output: train_df - unchanged from input
            test_df - pandas Dataframe with feature balancing corrections.
    '''

    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    in_train_not_test = train_cols - test_cols # set difference
    in_test_not_train = test_cols - train_cols # set difference

    # Drop cols in test but not train
    new_test_df = test_df.copy(deep=True) \
        .drop(labels=list(in_test_not_train), axis=1)

    # Add cols in train but not test as 0s
    for i in in_train_not_test:
        new_test_df[i] = np.zeros(len(test_df))

    return train_df, new_test_df

#
