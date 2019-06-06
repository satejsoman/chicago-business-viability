#==============================================================================#
# GEOCODE BUSINESSES WITHOUT COORDINATE INFORMATION
#
# Cecile Murray
#==============================================================================#

import argparse
import geocoder
import pandas as pd
import geopandas as gpd 
from itertools import islice
from functools import reduce

BUSINESS_LICENSE_DATA_LOCATION = "../../data/Business_Licenses.csv"
MISSING_LOCATION_RECORDS_FILE = "../../data/geocode/licenses_without_geocode.csv"


INDEX_VARS = ['LICENSE ID', 'LICENSE NUMBER', 'ACCOUNT NUMBER', 'SITE NUMBER', 'DATE ISSUED']
ADDRESS_VARS = ['ADDRESS', 'CITY', 'STATE', 'ZIP CODE']
GEO_VARS = ['LATITUDE', 'LONGITUDE', 'LOCATION']



def select_missing(df):
    ''' Prepare a dataframe with all the records we need to submit to geocoder '''

    missing = df.loc[df['LATITUDE'].isna()]

    missing = missing.loc[missing['YEAR'] > 2006]
    missing = missing.loc[missing['ADDRESS'] != "[REDACTED FOR PRIVACY]"]

    missing.drop(columns = ['LICENSE NUMBER', 'LICENSE NUMBER', 'ACCOUNT NUMBER',
                             'SITE NUMBER', 'DATE ISSUED', 'LATITUDE',
                            'LONGITUDE', 'LOCATION'], inplace=True)

    # missing.rename(columns = {'LICENSE NUMBER': })

    # missing.to_csv(MISSING_LOCATION_RECORDS_FILE)

    return missing



def read_keys(keyfile, key_name):
    ''' get list of keys out of key file'''

    with open(keyfile, 'rb') as k:
        keys = yaml.safe_load(k.read())
    
    return keys[key_name]


def chunk_df(df, chunk_size, base_filename = "../../data/geocode/licenses_to_geocode"):
    
    
    i = 0
    df_list = []
    for j in range(chunk_size, df.shape[0], chunk_size):
        df_list.append(df.iloc[i:j])
        df.iloc[i:j].to_csv(base_filename + "_{}-{}.csv".format(i, j))
        i = j
        
    return df_list


def prep_query(df):
    ''' Takes: dataframe of missing queries, concatenates fields
        Returns: list of address to query
    '''

    missing['full_address'] = reduce(lambda x, y: x + " " + y, )


def query_mapquest():
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Locate business records in Census block groups')

    parser.add_argument("--infile", help = "business records file to join to block groups", default = BUSINESS_LICENSE_DATA_LOCATION)
    parser.add_argument("--outfile", help = "file location for geocoded records", default = "../../data/geocoded_records.csv")
    parser.add_argument("--key", help = "API key for Google geocoder")

    args = parser.parse_args()

    df = pd.read_csv(args.infile)[INDEX_VARS + ADDRESS_VARS + GEO_VARS]
    df['DATE ISSUED'] = pd.to_datetime(df['DATE ISSUED'])
    df['YEAR'] = df['DATE ISSUED'].dt.year

    missing = select_missing(df)
    df_list = chunk_df(missing)



    
    