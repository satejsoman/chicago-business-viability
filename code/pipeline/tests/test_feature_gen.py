import numpy as np
import pandas as pd
import unittest

from feature_generation import (reshape_and_create_label, count_by_zip_year,
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
            'CITY_New York': [1, 1, 0],
            'CITY_Chicago': [0, 0, 1],
            'STATE_NY': [1, 1, 0],
            'STATE_IL': [0, 0, 1],
            'APPLICATION TYPE_renew': [1, 0, 0],
            'APPLICATION TYPE_cancel': [0, 1, 0],
            'APPLICATION TYPE_expand': [0, 0, 1]
        })

        pass




if __name__ == '__main__':
    unittest.main()
