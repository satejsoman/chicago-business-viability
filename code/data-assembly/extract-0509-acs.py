#==============================================================================#
# EXTRACT 2005-09 ACS DATA
#
# Cecile Murray
#==============================================================================#

import numpy as np
import pandas as pd


AGE_CSV = "ACS_09_5YR_S0101_with_ann.csv"
EDU_CSV = "ACS_09_5YR_S1501_with_ann.csv"

EDU_VAR = "HC01_EST_VC12"



def generate_age_vars(var_index_list):
    '''Takes list of tuples of columns/variable numbers
        Returns list of variables in Factfinder spreadsheet format
    '''

    return list(map(lambda x: "HC" + str(x[0]).zfill(2) + "_EST_VC" + str(x[1]).zfill(2),
                 var_index_list))


if __name__ == "__main__":
    
    age_vars = generate_age_vars([(1, 1)] + [(1, x) for x in range(10, 16)])
    age_vars.insert(0, "GEO.id2")

    # drop first row containing only descriptive names
    agedf = pd.read_csv(AGE_CSV).iloc[1:]
    edudf = pd.read_csv(EDU_CSV).iloc[1:]

    # cut age dataframe down to size
    agedf = agedf[age_vars]

    # cut edu data frame to only relevant variable and tract ID
    edudf = edudf[['GEO.id2', EDU_VAR]].rename(columns = {EDU_VAR: "share_BA+"})

    # merge these, replace - with NA
    df = pd.merge(edudf, agedf, on = "GEO.id2")    
    df.replace("-", np.nan, inplace = True)


    df['a35to64'] = df[age_vars[1:]].astype(float).apply(sum, 1)

    df.to_csv("Cook_tract_subject_vars_2005-09.csv")
