#==============================================================================#
# CONCATENATE GOVT DATA
#
# Cecile Murray
#==============================================================================#

import re
import os
import argparse
import numpy as np
import pandas as pd
from functools import reduce

TOTAL_POP_VAR = "B01001_001E"
MEDINC_VAR = "B19013_001E"
BA_SHARE_VAR = "S1501_C02_015E"
AGE_VARS = list(map(lambda x: 'S0101_C01_' + str(x).zfill(3) + 'E', range(9, 15)))
S0101_VARS = list(map(lambda x: 'S0101_C01_' + str(x).zfill(3) + 'E', range(1, 15)))
S0101_CONVERSION = dict(zip(list(map(lambda x: 'S0101_C02_' + str(x).zfill(3) + 'E', range(1, 15))),
                            list(map(lambda x: 'S0101_C01_' + str(x).zfill(3) + 'E', range(1, 15)))))

BA_VARS_2000 = ["P037015", "P037032"]
AGE_VARS_2000 = list(map(lambda x: "P008" + str(x).zfill(3), range(28, 35))) + \
                list(map(lambda x: "P008" + str(x).zfill(3), range(67, 74)))

INDEX_VARS = ['LICENSE ID', 'LICENSE NUMBER', 'ACCOUNT NUMBER', 'SITE NUMBER', 'DATE ISSUED']
FIPS_VARS = ["GEOID_2000", "GEOID_2010"]


def compute_population_density():
    '''Compute population per square mile '''

    pass


def extract_geoid(s):
    ''' Takes index from ACS API data frame and creates FIPS unique ID '''
    # "Census Tract 6611, Cook County, Illinois: Summary level: 140, state:17> county:031> tract:661100"

    stfips = re.search('state:[0-9]*', s).group(0).replace('state:', '')
    cofips = re.search('county:[0-9]*', s).group(0).replace('county:', '')
    trfips = re.search('tract:[0-9]*', s).group(0).replace('tract:', '')

    return stfips + cofips + trfips


def move_last_col_to_front(df):
    ''' Takes the rightmost column in a data frame and makes it the left-most '''

    cols = list(df.columns)
    return df[[cols[-1]] + cols[:-1]]


def fix_fips(df):
    ''' Turns the index column into a column with the 11 digit tract FIPS '''

    df['GEOID'] = df['index'].apply(lambda x: extract_geoid(x), 1)
    df.drop(columns = "index", inplace=True)

    return move_last_col_to_front(df)


def insert_NA_values(df):
    ''' Converts various ACS indications that a value isn't available into a NaN'''

    NULLS = ["-999999999", "-888888888", "-666666666", "-555555555", "-333333333",
            "-222222222", "*", "-"]
    df.replace(NULLS, np.nan, inplace = True)

    return df


def concatenate_years(file_pattern, years):
    ''' Takes: filename pattern for csv files generated from API call + a list of years
        Returns: concatenated data frame with a year column
    '''

    l = []
    for y in years:
        if os.path.exists(file_pattern + y + '.csv'):

            df = pd.read_csv(file_pattern + y + '.csv')
            df['year'] = y
            df = fix_fips(df.drop(columns = "Unnamed: 0"))
            df = insert_NA_values(df)
            if y == "2013-17":
                df.rename(columns = S0101_CONVERSION, inplace = True)

            l.append(df)

    return pd.concat(l)


def expand_across_years(df, year_range):
    ''' Takes: data frame with ACS-style year ranges, list of years
        Returns: data frame with duplicated row, one for each year in the range
    '''

    years = list(range(int(year_range.split("-")[0]),
                int(year_range.split("-")[1]) + 2001))

    year_df = pd.DataFrame(data = [years, [year_range] * len(years)]).T
    year_df.columns = ["year", "year_range"]
    cp = pd.merge(df, year_df, left_on = "year", right_on = "year_range")
    cp = cp.drop(columns = ["year_x", "year_range"]).rename(columns = {'year_y':'year'})
    cp['year'] = cp['year'].astype(int) 

    return cp


def compute_share(df, varname, total_var, numerator_vars, denom = False):
    ''' Compute share of population within given age range'''

    if denom:
        df[varname] = (df[numerator_vars].apply(sum, 1) / df[total_var]) * 100
    else:
        df[varname] = df[numerator_vars].apply(sum, 1)

    return df


def process_2010s():
    ''' Process 2010-14 and 2013-17 ACS API data '''

    # concatenate data pulled off the API
    detail_vars = concatenate_years("../../data/Cook_tract_detail_", ["2010-14", "2013-17"])
    subj_vars = concatenate_years("../../data/Cook_tract_subj_", ["2010-14", "2013-17"])

    # rename variables
    subj_vars = compute_share(subj_vars, 'a35to64_share', 'S0101_C01_001E', AGE_VARS)
    subj_vars = subj_vars.drop(columns = S0101_VARS).rename(columns = {BA_SHARE_VAR: "share_BA+"})
    detail_vars.rename(columns = {TOTAL_POP_VAR: 'total_pop',
                                  MEDINC_VAR: 'medhhinc'}, inplace = True)
    acs_2010s = pd.merge(detail_vars, subj_vars, on=['GEOID', 'year'], how = 'outer')

    return acs_2010s


def process_2005_09():
    ''' Process 2005-09 ACS data from API and FactFinder '''

        # bring in 2005-09 API data and manually assembled data from Factfinder
    detail09 = pd.read_csv("../../data/Cook_tract_detail_2005-09.csv")
    detail09.rename(columns = {TOTAL_POP_VAR: 'total_pop',
                                MEDINC_VAR: 'medhhinc'}, inplace = True)
    detail09 = fix_fips(detail09)
    detail09.drop(columns = "Unnamed: 0", inplace = True)
    subj09 = pd.read_csv("Cook_tract_subject_vars_2005-09.csv").rename(columns = {'GEO.id2': "GEOID"})
    subj09 = subj09.filter(items = ['GEOID', 'share_BA+', 'a35to64_share'])
    subj09['GEOID'] = subj09['GEOID'].astype(str)
    acs_2000s = pd.merge(detail09, subj09, on = ['GEOID'], how = "outer")

    return acs_2000s


def process_2000():

    cdf = pd.read_csv("census2000.csv")
    cdf.rename(columns = {"P053001": "medhhinc",
                         "P008001": "total_pop"}, inplace = True)
    cdf = fix_fips(cdf)

    cdf = compute_share(cdf, 'share_BA+', "total_pop", BA_VARS_2000, denom = True)
    cdf = compute_share(cdf, 'a35to64_share', "P037001", AGE_VARS_2000, denom = True)
    cdf.drop(columns = list(cdf.filter(regex = "P0").columns) + ["Unnamed: 0"], inplace = True)

    return cdf


def prep_license_data(bl):
    
    # for geoid in ["GEOID_2000", "GEOID_2010"]:
    for geoid in ["GEOID_2010"]:
        bl[geoid] = bl[geoid].astype(str)

    bl['year'] = pd.to_datetime(bl['DATE ISSUED']).dt.year.astype(int)

    return bl    



def merge_with_licenses(bl, data, year_ranges, bl_geoid_col, data_geoid_col):
    ''' steps to merge data df into business license data '''

    data[data_geoid_col] = data[data_geoid_col].astype(str)

    for y in year_ranges:
        data = expand_across_years(data, y)
    
    merged = pd.merge(bl, data, left_on = [bl_geoid_col, "year"], right_on = ["GEOID", "year"])

    return merged



if __name__ == "__main__":

    # acs_2010s = process_2010s()
    # acs_2000s = process_2005_09()
    # census_2000 = process_2000()

    # # write to file
    # acs_2000s.to_csv("../../data/0509_clean_ACS.csv")
    # acs_2010s.to_csv("../../data/2010s_clean_ACS.csv")
    # census_2000.to_csv("../../data/2000_clean_Census.csv")

    acs_2000s = pd.read_csv("../../data/0509_clean_ACS.csv")
    acs_2010s = pd.read_csv("../../data/2010s_clean_ACS.csv")
    census_2000 = pd.read_csv("../../data/2000_clean_Census.csv")

    bls = pd.read_csv("../../data/Cook_annual_unemployment.csv")
    bea = pd.read_csv("../../data/chicago_rgdp_2001-17.csv")


    bl = pd.read_csv("../../data/business_licenses_with_tracts.csv")[INDEX_VARS + ["GEOID_2010"]]
    bl = prep_license_data(bl)

    test = merge_with_licenses(bl, acs_2010s, ["2010-14", "2013-17"], "GEOID_2010", "GEOID")
    test.drop(columns = ["Unnamed: 0"], inplace = True)
    # test.to_csv("../../data/merged_business_acs_test_file.csv")
