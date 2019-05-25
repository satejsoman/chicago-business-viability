#==============================================================================#
# PULL UNEMPLOYMENT DATA ON COOK COUNTY
#
# Cecile Murray
#==============================================================================#

import bls
import json
import argparse
import pandas as pd


COOK_AREA_CODE = "LAUCN170310000000003"


def load_key(args, API = "BLS"):
    ''' Returns specified API key '''

    with open(args.keyfile) as f:
        keys = json.load(f)
    
    return keys[API]


def pull_unemployment(args, key, annual_avg = True):

    try:
        data = pd.DataFrame(bls.get_series(COOK_AREA_CODE,
                                            startyear = args.startyear,
                                            endyear = args.endyear,
                                            key = key)).reset_index()
    
    except Exception as e:
        print("API call failed")
        print(e)
    
    if annual_avg:
        data['year'] = data['date'].dt.year
        return data.groupby('year').mean().rename(columns = {COOK_AREA_CODE: "Cook_U3_ann_avg"})

    else:
        return data.rename(columns = {COOK_AREA_CODE: "Cook_U3"})




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='wrapper for pulling BLS data off the API')
    parser.add_argument("--keyfile", help="file containing API keys", default = 'keys.json')
    parser.add_argument("--startyear", default = 2000)
    parser.add_argument("--endyear", default = 2018)
    parser.add_argument("--outfile", help = "where to write the csv out")
    args = parser.parse_args()

    key = load_key(args, "BLS")
    data = pull_unemployment(args, key, annual_avg=True)
    data.to_csv(args.outfile)