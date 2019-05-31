#==============================================================================#
# ASSIGN BUSINESSES TO CENSUS TRACTS
#
# Cecile Murray
#==============================================================================#

import argparse
import fiona.crs
import pandas as pd
import geopandas as gpd 
from shapely.geometry import Point    


BUSINESS_LICENSE_DATA_LOCATION = "../../data/Business_Licenses.csv"
BLOCK_2010_SHAPEFILE_LOCATION = "../../data/Cook_tract_2010.geojson"
BLOCK_2000_SHAPEFILE_LOCATION = "../../data/Cook_tract_2000.geojson"
COLS = ['LICENSE ID', 'LICENSE NUMBER', 'LEGAL NAME', 'ACCOUNT NUMBER',
         'SITE NUMBER', 'ADDRESS', 'CITY', 'STATE', 'ZIP CODE',
         'LATITUDE', 'LONGITUDE', 'LOCATION', 'DATE ISSUED']
INDEX_VARS = ['LICENSE ID', 'LICENSE NUMBER', 'ACCOUNT NUMBER', 'SITE NUMBER', 'DATE ISSUED']
ADDRESS_VARS = ['ADDRESS', 'CITY', 'STATE', 'ZIP CODE']


def convert_to_Point(df):
    '''Takes business licenses data frame and converts lat/lon fields to Point'''

    df['coords'] = list(zip(df.LONGITUDE, df.LATITUDE))
    df['coords'] = df['coords'].apply(Point)
    return gpd.GeoDataFrame(df, geometry = 'coords', crs = fiona.crs.from_epsg(4269))


def join_to_blkgrp(gdf, block_filepath, year):
    ''' Joins business license points to block group polygons'''

    blks = gpd.read_file(block_filepath)

    if year == 2000:
        blks.rename(columns = {'TRACTCE00': "GEOID"}, inplace = True)

    gdf = gpd.sjoin(gdf, blks, op = "within", how = 'inner')
    gdf.drop(columns = list(set(blks.columns).difference(["GEOID", "geometry"])), inplace=True)

    return gdf


def join_records_to_blocks(df, block_filepath, year):
    ''' perform basic inner join on non-null observations within data frame'''

    gdf = convert_to_Point(df)
    gdf = gdf[gdf.geometry.notnull()]
    gdf = join_to_blkgrp(gdf, block_filepath, year)
    gdf.drop(columns = "index_right", inplace = True)
    gdf.rename(columns = {"GEOID": "GEOID_" + str(year)}, inplace = True)
    print(gdf.columns)
    print(gdf.head())
    
    return gdf


def perform_basic_join(df, outfile):

    gdf = join_records_to_blocks(df, BLOCK_2010_SHAPEFILE_LOCATION, 2010)
    gdf = join_records_to_blocks(gdf, BLOCK_2000_SHAPEFILE_LOCATION, 2000)
    drop_cols = list(set(gdf.columns).difference(set(INDEX_VARS + ['GEOID_2000', 'GEOID_2010'])))
    gdf.drop(columns = drop_cols, inplace = True) 

    gdf.to_csv(outfile)
    return gdf



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Locate business records in Census block groups')

    parser.add_argument("--infile", help = "business records file to join to block groups", default = BUSINESS_LICENSE_DATA_LOCATION)
    parser.add_argument("--outfile", help = "file location for basic join", default = "../../data/business_licenses_with_tracts.csv")

    args = parser.parse_args()

    df = pd.read_csv(args.infile)[COLS]
    gdf = perform_basic_join(df, args.outfile)
    
    
    





    # # there are 1,830 unique account numbers where the address is redacted
    # df[['LEGAL NAME', 'ADDRESS', 'LOCATION']].loc[df['ADDRESS'] == "[REDACTED FOR PRIVACY]"].shape

    # # there are 74,352 observations that don't spatial join
    # # of these, 66905 are not listed in Chicago
    # # none of the ones that don't join have lat/long available, 
    # #   but 435 of those that do aren't in IL and 3794 have a city other than chicago
    # df[~df['LICENSE ID'].isin(gdf['LICENSE ID'])]
    # df[~df['LICENSE ID'].isin(jdf['LICENSE ID'])].loc[df['CITY'] != "CHICAGO"]
    # df[df['LICENSE ID'].isin(jdf['LICENSE ID'])].loc[df['STATE'] != "IL"]
    # df[df['LICENSE ID'].isin(jdf['LICENSE ID'])].loc[df['CITY'] != "CHICAGO"]