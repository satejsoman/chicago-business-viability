#==============================================================================#
# ASSEMBLE BEA DATA ON OUTPUT
#
# Cecile Murray
#==============================================================================#

import json
import pybea
import argparse


def make_api_call(dataset, component, industry_id, year, fips = "MSA", result_format = 'JSON'):

    data = pybea.get_data(USER_ID,
                        DataSetName = dataset,
                        Component = component,
                        IndustryId = industry_id,
                        GeoFips = fips,
                        Year = year,
                        ResultFormat = result_format)
    return None
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='wrapper for pulling BEA data off the API')
    parser.add_argument("--keyfile", help="file containing API keys", default = 'keys.json')
    args = parser.parse_args()

    with open(args.keyfile) as f:
        keys = json.load(f)
    
    USER_ID = keys['BEA']

    try:
        data = pybea.get_data(USER_ID,
                            DataSetName = 'Regional',
                            Component = "RGDP_MAN",
                            IndustryId = 1,
                            GeoFips = "MSA",
                            Year = "ALL",
                            ResultFormat = "JSON")
    except Exception as e:
        print("API call failed")
        print(e)