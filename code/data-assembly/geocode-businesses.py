#==============================================================================#
# ASSIGN BUSINESSES TO CENSUS TRACTS
#
# Cecile Murray
#==============================================================================#

import pandas as pd
import geopandas as gpd 
import fiona.crs
from shapely.geometry import Point    

BUSINESS_LICENSE_DATA_LOCATION = "../../data/Business_Licenses.csv"
BLOCK_SHAPEFILE_LOCATION = "../../data/Cook_bg.geojson"
COLS = ['LICENSE ID', 'LEGAL NAME', 'ACCOUNT NUMBER', 'SITE NUMBER', 'ADDRESS',
         'CITY', 'STATE', 'ZIP', 'LATITUDE', 'LONGITUDE', 'LOCATION']


def convert_to_Point(df):
    '''Takes business licenses data frame and converts lat/lon fields to Point'''

    df['coords'] = list(zip(df.LONGITUDE, df.LATITUDE))
    df['coords'] = df['coords'].apply(Point)
    return gpd.GeoDataFrame(df, geometry = 'coords', crs = fiona.crs.from_epsg(4269))


def join_to_blkgrp(df):
    ''' Joins business license points to block group polygons'''

    blks = gpd.read_file(BLOCK_SHAPEFILE_LOCATION)
    return gpd.sjoin(df, blks, op = "within", how = 'inner')


def main():

    df = pd.read_csv(BUSINESS_LICENSE_DATA_LOCATION)[COLS]
    
    df = convert_to_Point(df)
    df = join_to_blkgrp(df)
    
    return df

if __name__ == "__main__":
    
    df = main()