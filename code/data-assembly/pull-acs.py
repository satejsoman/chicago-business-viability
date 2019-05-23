#==============================================================================#
# COLLECT AMERICAN COMMUNITY SURVEY DATA
#
# Cecile Murray
#==============================================================================#

import argparse
import json
import pandas as pd
import censusdata as census
from functools import reduce
from itertools import islice


def get_user_input():
    '''Prompts user to enter table ID and variable numbers, dumps to file '''

    dictlist = []
    done_entering = False

    while not done_entering:

        table_id  = input("Please input the table number.\n")
        start = int(input("Please input the first number of the variables to query.\n"))
        stop = int(input("Please input the last number of the variables to query.\n"))
        
        dictlist.append({'table_id': table_id.upper(), 
                         'var_range':  list(range(start, stop))})
        
        more = input("Type 'done' if you have no more tables to query.\n")
        if more == 'done':
            done_entering = True
    
    # with open("query.json", 'w') as f:
    #     json.dump(dictlist)
    
    return dictlist


def make_api_varlist(table_id, var_range):
    ''' Takes a table id string and indices for variable selection
        Returns: list of variables for census API call
    '''
    
    return list(map(lambda x: table_id + '_' + str(x).zfill(3) + 'E', var_range))


def construct_geolist(state = "", county = "", tract = ""):
    ''' Takes FIPS codes for geographies 
        Returns list formatted for querying API
    '''

    rv = []

    if state:
        rv.append(('state', state))
    if county:
        rv.append(('county', county))
    if tract:
        rv.append(('tract', tract))
    
    if not rv:
        print("List is empty - please specify some geographies to query.")
    
    return rv


def make_api_call(varlist, survey, year, geo):
    ''' Takes list of variables, survey to query, year to query, 
        and list of tuples specifying geography
        Returns: data frame with resulting data
    '''

    dfs = []

    # API accepts maximum 50 variable requests at a time
    i = 0
    for j in range(49, len(varlist) + 50, 50):
        
        try:
            dfs.append(census.download(survey, year, census.censusgeo(geo),
                                        list(islice(varlist, i, j))).reset_index())
            i = j

        except Exception as e:
            print("API call failed")
            print(e)

    return reduce(lambda x, y: pd.merge(x, y, on="index"), dfs)


def main(args, vardict):

    varlist = []
    for d in vardict:
        varlist = varlist + make_api_varlist(**d)
    
    geos = construct_geolist(state = args.state, county = args.county, tract = args.tract)

    data = make_api_call(varlist, args.survey, args.year, geos)

    # this is sensitive to failure, need to improve
    data['geoid'] = data['index'].apply(lambda x: args.state + args.county + x.geo[TRACTCODE][1])

    if args.outfile:
        data.to_csv(args.outfile)
        
    return data


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description='wrapper for pulling ACS data off the API')

    parser.add_argument("--varfile", help = "file containing list of dicts mapping tables to variables") # this could be better
    parser.add_argument("--survey", help = "survey string")
    parser.add_argument("--year", help = "year to query", default = 2017)
    parser.add_argument("--state", help = "state", default = "17")
    parser.add_argument("--county", help = "county", default = "031")
    parser.add_argument("--tract", help = "tract", default = "*")
    parser.add_argument("--outfile", help="file to write to")

    args = parser.parse_args()

    
    if args.varfile:
        with open(args.varfile) as f:
            varlist = json.load(f)
            print(varlist)

    else:
        varlist = get_user_input()
        print(varlist)

    data = main(args, varlist)




