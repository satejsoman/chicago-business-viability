import numpy as np
import pandas as pd
import unittest
from pathlib import Path

from feature_generation import (make_dummy_vars, reshape_and_create_label,
                                get_locations, count_by_zip_year,
                                count_by_dist_radius, balance_features)

# Methods to test:
# 4. count_by_zip_year()
# 5. count_by_dist_radius()
# 6. balance_features()

class TestFeatureGeneration(unittest.TestCase):


    def test_make_dummy_vars(self):
        '''Tests make_dummy_vars() in feature_generation.py'''

        input = pd.DataFrame(data={
            'ID': [1, 2, 3],
            'CITY': ['New York', 'New York', 'Chicago'],
            'STATE': ['NY', 'NY', 'IL'],
            'APPLICATION TYPE': ['renew', 'cancel', 'expand']
        })
        output = pd.DataFrame(data={
            'ID': [1, 2, 3],
            'CITY_Chicago': [0, 0, 1],
            'CITY_New York': [1, 1, 0],
            'STATE_IL': [0, 0, 1],
            'STATE_NY': [1, 1, 0],
            'APPLICATION TYPE_cancel': [0, 1, 0],
            'APPLICATION TYPE_expand': [0, 0, 1],
            'APPLICATION TYPE_renew': [1, 0, 0],
        }, dtype=np.int64)
        self.assertTrue(all(make_dummy_vars(input) == output))


    def test_reshape_and_create_label(self):
        '''Input df must have cols: ACCOUNT NUMBER, SITE NUMBER, DATE ISSUED,
        LICENSE TERM EXPIRATION DATE
        '''

        DATE_COLS = ['DATE ISSUED', 'LICENSE TERM EXPIRATION DATE']
        input = pd.read_csv(Path(__file__).parent/'test_feature_data.csv',
                            low_memory=False,
                            parse_dates=DATE_COLS)

        output = pd.DataFrame(data={
            'ACCOUNT NUMBER': [
                1, 1, 1, 1, 1,
                1,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                4, 4,
                6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6
            ],
            'SITE NUMBER': [
                1, 1, 1, 1, 1,
                2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
            ],
            'YEAR': [
                2002, 2003, 2004, 2005, 2006,
                2016,
                2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                    2012, 2013, 2014, 2015, 2016,
                2002, 2003,
                2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                    2012, 2013, 2014, 2015, 2016
            ],
            'not_renewed_2yrs': [
                0, 0, 0, 0, 1,
                0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
            ]
        })
        self.assertTrue(all(reshape_and_create_label(input) == output))


    def test_get_locations(self):
        '''Transforms license data to unique locations for account-site data.
        Output cols: 'ACCOUNT NUMBER', 'SITE NUMBER', 'ADDRESS', 'CITY',
                     'STATE', 'ZIP CODE', 'WARD', 'POLICE DISTRICT',
                     'LATITUDE', 'LONGITUDE', 'LOCATION'
        Drops rows if these cols have NA: 'LATITUDE', 'LONGITUDE', 'LOCATION'
        '''

        DATE_COLS = ['DATE ISSUED', 'LICENSE TERM EXPIRATION DATE']
        input = pd.read_csv(Path(__file__).parent/'test_feature_data.csv',
                            low_memory=False,
                            parse_dates=DATE_COLS)

        output = pd.DataFrame(data={
            'ACCOUNT NUMBER': [1, 1, 2, 4, 6],
            'SITE NUMBER': [1, 2, 2, 1, 1,],
            'ADDRESS': ['17 W ADAMS ST # 1ST',
                        '17 W ADAMS ST BSMT & 1ST',
                        '11601 W TOUHY AVE  T1 CO',
                        '1028 W DIVERSEY PKWY',
                        '3714 S HALSTED ST 1ST #'],
            'CITY': ['Chicago', 'Chicago', 'Chicago', 'Chicago', 'Chicago'],
            'STATE': ['IL', 'IL', 'IL', 'IL', 'IL'],
            'ZIP CODE': [60603, 60663, 60666, 60614, 60609],
            'LATITUDE': [41.87934194,
                         41.87934194,
                         42.0085364,
                         41.93272677,
                         41.82718502],
            'LONGITUDE': [-87.62841189,
                          -87.62841189,
                          -87.91442844,
                          -87.65504178,
                          -87.64617046]
        })
        self.assertTrue(all(get_locations(input) == output))


    def test_count_by_zip_year(self):

        DATE_COLS = ['DATE ISSUED', 'LICENSE TERM EXPIRATION DATE']
        lic_data = pd.read_csv(Path(__file__).parent/'test_feature_data.csv',
                            low_memory=False,
                            parse_dates=DATE_COLS)

        input = reshape_and_create_label(lic_data)

        output = pd.DataFrame(data={
            'ACCOUNT NUMBER': [
                1, 1, 1, 1, 1,
                1,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                4, 4,
                6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6
            ],
            'SITE NUMBER': [
                1, 1, 1, 1, 1,
                2,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
            ],
            'YEAR': [
                2002, 2003, 2004, 2005, 2006,
                2016,
                2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                    2012, 2013, 2014, 2015, 2016,
                2002, 2003,
                2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                    2012, 2013, 2014, 2015, 2016
            ],
            'num_not_renewed_zip': [
                0, 0, 0, 0, 1,
                0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 1,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
            ]
        })
        self.assertTrue(all(count_by_zip_year(input, lic_data) == output))




if __name__ == '__main__':
    unittest.main()
