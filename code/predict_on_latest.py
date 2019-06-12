import yaml
import configparser
import numpy as np
import pandas as pd
from feature_generation import (balance_features, make_dummy_vars,
                                classify_business_activity)
from chicago_business import get_config, get_pipeline
from joblib import dump, load


MERGE_KEYS = ['ACCOUNT NUMBER', 'SITE NUMBER', 'YEAR']

def make_features(input_df, feature_generators, existing_features):
    '''
    Takes an input license-level dataframe and a list of feature generation
    functions to apply, and returns a df of transformed business-year data.

    Input:  input_df - raw licence-level data, after test-train split
            feature_generators - a list of functions to apply to each test
                or train df that return the transformed df
    Output: result_df - df of business-year data with all features created
    '''

    base = reshape_to_business_year(input_df) # business-year data

    # print("base is \n", base.head())
    print("1 ", existing_features)
        
    generated_features = base.copy(deep=True) # accumulate features
    for feature_generator in feature_generators:
        print("        Applying function:", feature_generator.__name__)
        feature = feature_generator(base, input_df)
        generated_features = generated_features.merge(feature,
            how='left', on=MERGE_KEYS)

    # print("generated features is \n", generated_features.head())    
    # print("2 ", existing_features)

    # Merge on all generated feature onto the base
    result_df = base \
        .drop(labels=['not_renewed_2yrs'], axis=1, errors='ignore') \
        .merge(generated_features,
               how='left',
               on=MERGE_KEYS) \
        .drop(columns = ["num_sites_x"]) \
        .rename(columns = {"num_sites_y": "num_sites"})
    # Merge on existing features
    result_df["JOIN_YEAR"] = result_df["YEAR"] - 1

    # print("results df is \n", result_df.head())

    # print(existing_features)

    # Get unique account-site level data for existing features
    existing_df = input_df[MERGE_KEYS + existing_features].drop_duplicates()
    full_result = result_df \
        .drop(labels=['YEAR'], axis=1) \
        .merge(existing_df,
            how='left',
            right_on=MERGE_KEYS,
            left_on=['ACCOUNT NUMBER', 'SITE NUMBER', 'JOIN_YEAR']) \
        .drop(labels=['YEAR'], axis=1)

    # print("full result: \n", full_result.head())
    # print(full_result.columns)
    # Fix year indexing
    full_result['YEAR'] = full_result["JOIN_YEAR"] + 1
    full_result = full_result.drop(labels=['JOIN_YEAR'], axis=1)

    return full_result


# Reshape license data to business-year data
def reshape_to_business_year(input_df):
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
            #  <--- goes above .sort_Values

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

    # Drop unnecessary columns
    df = df.drop(labels=['account_site', 'min_license_date','max_license_date',
                         'expiry'], axis=1) 

    return df


def make_business_features_for_prediction(pipeline, df, original_features):

    # old_cols = set(pipeline.test_sets[0].columns) # dunno about this
    print(original_features)

    # Filter out licenses for Jan 1, 2019 onwards
    df = df.loc[df['DATE ISSUED'] <= pd.to_datetime('12/31/2018')]
    df = make_features(df, [make_dummy_vars, classify_business_activity], original_features)

    print(df.columns)

    # Check for feature balance on each test-train pair
    train_set, df = balance_features(pipeline.train_sets[0], df)

    # Filter only for business-years in 2018
    df = df.loc[df['YEAR'] == 2018]
    df['num_not_renewed_geo'] = 0

    return df



if __name__ == "__main__":
    
    config, script_dir = get_config("final-prediction-config.yml")

    original_features = list(config["pipeline"]["precomputed_features"])
    print(original_features)

    # generate a pipeline object as though test set would be 2018
    pipeline = (get_pipeline(config, script_dir)
                 .load_data()
                 .clean_data()
                 .summarize_data()
                 .generate_test_train()
                 .preprocess_data()
                 .generate_features()
                 .run_model_grid())
    
    # grab the preprocessed data from that pipeline
    df = pipeline.dataframe

    # now make features on the df containing just 1/1/2018-12/31/2018
    df = make_business_features_for_prediction(pipeline, df, original_features)

    # extract trained model from the pipeline
    dt = pipeline.trained_models['DecisionTreeClassifier-max_depth1'][0]
    
    scores = dt2.predict_proba(df.fillna(0)[pipeline.features])
    ranks = pd.DataFrame(scores[:,1]).rank(method="first")
    k = math.floor(ranks.count() * 0.02)
    predictions = np.where(ranks < k, 1, 0)

    cols = ['ACCOUNT NUMBER', 'SITE NUMBER', 'LEGAL NAME', 'DOING BUSINESS AS NAME']
    result = pd.merge(predicted_failures,
                        pipeline.dataframe[cols].drop_duplicates(),
                        on=["ACCOUNT NUMBER", 'SITE NUMBER'],
                        how="left")