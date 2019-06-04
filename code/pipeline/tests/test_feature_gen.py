import numpy as np
import pandas as pd
import unittest

from feature_generation import (make_dummy_vars, reshape_and_create_label,
                                get_locations, count_by_zip_year,
                                count_by_dist_radius, balance_features)

# Methods to test:
# 1. make_dummy_vars()
# 2. reshape_and_create_label()
# 3. get_locations()
# 4. count_by_zip_year()
# 5. count_by_dist_radius()
# 6. balance_features()

class TestFeatureGeneration(unittest.TestCase):


    def test_make_dummy_vars(self):
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
        input = pd.DataFrame(data={
            'ACCOUNT NUMBER': [
                1, 1, 1, 1,
                2, 2, 2, 2,
                3, 3, 3, 3
            ],
            'SITE NUMBER': [
                1, 1, 1, 1,
                1, 1, 1, 1,
                1, 1, 1, 1
            ],
            'DATE ISSUED': [pd.to_datetime(x) for x in (
                '1/1/2012', '1/1/2013', '1/1/2014', '1/1/2015',
                '1/1/2004', '1/1/2005', '1/1/2006', '1/1/2007',
                '1/1/2009', '1/1/2010', '1/1/2011', '1/1/2012',
            )],
            'LICENSE TERM EXPIRATION DATE': [pd.to_datetime(x) for x in (
                '1/1/2014', '1/1/2015', '1/1/2016', '1/1/2017',
                '1/1/2006', '1/1/2007', '1/1/2008', '1/1/2009',
                '1/1/2011', '1/1/2012', '1/1/2013', '1/1/2014',
            )]
        })
        output = pd.DataFrame(data={
            'ACCOUNT NUMBER': [
                1, 1, 1, 1, 1,
                2, 2, 2, 2, 2,
                3, 3, 3, 3, 3
            ],
            'SITE NUMBER': [
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
                1, 1, 1, 1, 1
            ],
            'YEAR': [
                2012, 2013, 2014, 2015, 2016,
                2004, 2005, 2006, 2007, 2008,
                2009, 2010, 2011, 2012, 2013
            ],
            'not_renewed_2yrs': [
                0, 0, 0, 0, 0, # expiry after buffer
                0, 0, 0, 0, 1, # expiry inside training set
                0, 0, 0, 0, 1  # expiry within buffer
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
        input = pd.DataFrame(data={
            'ACCOUNT NUMBER': [
                1, 1, 1,
                2, 2, 2,
                3, 3, 3,
            ],
            'SITE NUMBER': [
                1, 1, 1,
                1, 1, 1,
                1, 1, 1,
            ],
            'ADDRESS': [
                '200 1ST AVE', '200 1ST AVE', '200 1ST AVE',
                '1600 PENN AVE', '1600 PENN AVE', '1600 PENN AVE',
                '1307 E 60TH ST', '1307 E 60TH ST', '1307 E 60TH ST'
            ],
            'EXTRANEOUS COLUMN 1': [
                1, 1, 1,
                2, 2, 2,
                3, 3, 3
            ],
            'CITY': [
                'New York', 'New York', 'New York',
                'Washington', 'Washington', 'Washington',
                'Chicago', 'Chicago', 'Chicago'
            ],
            'STATE': [
                'NY', 'NY', 'NY',
                'DC', 'DC', 'DC',
                'IL', 'IL', 'IL'
            ],
            'ZIP CODE': [
                10009, 10009, 10009,
                20500, 20500, 20500,
                60637, 60637, 60637
            ],
            'EXTRANEOUS COLUMN 2': [
                1, 1, 1,
                2, 2, 2,
                3, 3, 3
            ],
            'LATITUDE': [
                100000, 100000, 100000,
                200000, 200000, 200000,
                300000, 300000, 300000
            ],
            'LONGITUDE': [
                100000, 100000, 100000,
                200000, np.nan, 200000, # should be robust to NaNs if not all NaN
                300000, 300000, 300000
            ]
        })
        output = pd.DataFrame(data={
            'ACCOUNT NUMBER': [1, 2, 3],
            'SITE NUMBER': [1, 1, 1],
            'ADDRESS': ['200 1ST AVE', '1600 PENN AVE', '1307 E 60TH ST'],
            'CITY': ['New York', 'Washington', 'Chicago'],
            'STATE': ['NY', 'DC', 'IL'],
            'ZIP CODE': [10009, 20500, 60637],
            'LATITUDE': [100000, 200000, 300000],
            'LONGITUDE': [100000, 200000, 300000]
        })
        self.assertTrue(all(get_locations(input) == output))




if __name__ == '__main__':
    unittest.main()
