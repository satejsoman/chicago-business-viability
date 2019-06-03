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

BUSINESS_LICENSE_DATA_LOCATION = "../../data/Business_Licenses.csv"
MISSING_LOCATION_RECORDS_FILE = "../../data/licenses_without_geocode.csv"


INDEX_VARS = ['LICENSE ID', 'LICENSE NUMBER', 'ACCOUNT NUMBER', 'SITE NUMBER', 'DATE ISSUED']
ADDRESS_VARS = ['ADDRESS', 'CITY', 'STATE', 'ZIP CODE']
GEO_VARS = ['LATITUDE', 'LONGITUDE', 'LOCATION']



def select_missing(df):
    ''' Prepare a dataframe with all the records we need to submit to geocoder '''

    missing = df.loc[df['LATITUDE'].isna()]
    missing.to_csv(MISSING_LOCATION_RECORDS_FILE)

    return missing

    # missing['FULL_ADDRESS'] = missing[ADDRESS_VARS].apply( ???/, axis = 1)


def read_keys(keyfile):
    ''' get list of keys out of key file'''

    with open(keyfile, 'rb') as k:
        keys = yaml.safe_load(k.read())
    
    return keys["Google"]


def batch_geocode(df, google_key):
    
    df['FULL_ADDRESS'] = df['ADDRESS'] + " " + df['CITY'] + " " + df['STATE'] + " " + df['ZIP CODE'].astype(str)

    for chunk in enumerate(df):
        g = geocoder.google(df['FULL_ADDRESS'], key = google_key, method = "batch")


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

    missing = missing.loc[missing['YEAR'] > 2006]
    missing = missing.loc[missing['ADDRESS'] != "[REDACTED FOR PRIVACY]"]

    
    