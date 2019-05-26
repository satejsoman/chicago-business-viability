#==============================================================================#
# CONCATENATE GOVT DATA
#
# Cecile Murray
#==============================================================================#

import re
import os
import argparse
import pandas as pd
from functools import reduce



def compute_share(df, varname, total_var, numerator_vars):
    ''' Compute share of population within given age range'''

    return df[varname] = df[numerator_vars].apply(sum, 1) / df[total_var]



def compute_population_density():
    '''Compute population per square mile '''

    pass


def concatenate_years(file_pattern, years = ["2005-09", "2010-14", "2013-17"]):

    l = []
    for y in years:
        if os.path.exists(file_pattern + y + '.csv'):
            print("found file")
            l.append(pd.read_csv(file_pattern + y + '.csv'))

    return pd.concat(l)


def extract_geoid(s):
    ''' Takes index from ACS API data frame and creates FIPS unique ID '''

    # "Census Tract 6611, Cook County, Illinois: Summary level: 140, state:17> county:031> tract:661100"

    stfips = re.search('state:[0-9]*', s).group(0).replace('state:', '')
    cofips = re.search('county:[0-9]*', s).group(0).replace('county:', '')
    trfips = re.search('tract:[0-9]*', s).group(0).replace('tract:', '')

    return stfips + cofips + trfips


if __name__ == "__main__":
    
    detail_vars = concatenate_years("../../data/Cook_tract_detail_")
    age_vars = concatenate_years("../../data/Cook_tract_subj_")

    acs = pd.merge(detail_vars, age_vars, on='index')
    acs['GEOID'] = acs['index'].apply(lambda x: extract_geoid(x), 1)
    acs.drop(columns = ["Unnamed: 0_x", "Unnamed: 0_y"], inplace=True)