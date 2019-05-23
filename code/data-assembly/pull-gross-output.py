#==============================================================================#
# ASSEMBLE BEA DATA ON OUTPUT
#
# Cecile Murray
#==============================================================================#

import pandas as pd
import pybea
import argparse

USER_ID = '5BACE2AA-6D06-4D89-997C-539ED5463C16'


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
    
    data = pybea.get_data(USER_ID,
                        DataSetName = 'Regional',
                        Component = "RGDP_MAN",
                        IndustryId = 1,
                        GeoFips = "MSA",
                        Year = "ALL",
                        ResultFormat = "JSON")