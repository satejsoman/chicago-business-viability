#==============================================================================#
# COLLECT AMERICAN COMMUNITY SURVEY DATA
#
# Cecile Murray
#==============================================================================#

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import censusdata as census
import fiona.crs
from functools import reduce



def make_api_varlist(table_id, var_start, var_stop):
    ''' Takes a table id string and start/stop indices for variable selection
        Returns: list of variables for census API call
    '''
    
    return list(map(lambda x: table_id + '_' + str(x).zfill(3) + 'E', range(var_start, var_stop)))


def construct_geolist(state = "", county = "", tract = ""):
    ''' Takes FIPS codes for geographies 
        Returns list formatted for querying API
    '''

    rv = []

    if state:
        rv.append('state', state)
    if county:
        rv.append('county', county)
    if tract:
        rv.append('tract', tract)
    
    if not rv:
        print("List is empty - specify some geographies?")
    
    return rv


def make_api_call(varlist, survey, year, geo):
    ''' Takes list of variables, survey to query, year to query, 
        and list of tuples specifying geography
        Returns: data frame with resulting data
    '''

    dfs = []

    for i in range(0, len(varlist), 50):
        
        try:
            dfs.append(census.download(survey, year, census.censusgeo(geo)))

        except Exception as e:
            print("API call failed")
            print(e)

    return reduce(lambda x, y: pd.merge(x, y, on="index"), dfs)
    
# hhold_vars = make_api_varlist('B11001', 1, 10)
# race_vars = make_api_varlist('B03002', 1, 21)
# edu_vars = make_api_varlist('B15003', 1, 26)



#     census.censusgeo([('state', '17'), ('county', '031'), ('block group', '*')]),
#     hhold_vars + race_vars).reset_index()
# edu = census.download('acs5', 2017,
#     census.censusgeo([('state', '17'), ('county', '031'), ('block group', '*')]),
#     edu_vars + ['B19013_001E']).reset_index()

# # create unique FIPS ID
# data['GEOID'] = data['index'].apply(lambda x: '17031' + x.geo[TRACTCODE][1] + x.geo[BLKGRPCODE][1])

# # compute variables of interest
# data['pct_1parent'] = data['B11001_004E'] / data['B11001_001E']
# data['pct_alone'] = data['B11001_008E'] / data['B11001_001E']
# data['pct_white'] = data['B03002_003E'] / data['B03002_001E']
# data['pct_black'] = data['B03002_004E'] / data['B03002_001E']
# data['pct_hisp'] = data['B03002_012E'] / data['B03002_001E']
# data['medinc'] = data['B19013_001E'].replace(-666666666, np.nan)
# vars = ['pct_1parent', 'pct_alone', 'pct_white', 'pct_black', 'pct_hisp', 'pct_nohs', 'pct_BA', 'medinc']

