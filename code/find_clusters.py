import yaml
import pandas as pd 
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from types import MethodType
from pipeline.core import Pipeline
from pipeline.transformation import (Transformation, binarize, categorize,
                                     hash_string, replace_missing_with_value,
                                     scale_by_max, to_datetime)
from pipeline.grid import Grid
from pipeline.splitter import Splitter
from data_cleaning import (clean_data, filter_out_2019_data)
from feature_generation import (make_features, reshape_and_create_label,
                                count_by_zip_year, count_by_dist_radius,
                                balance_features, make_dummy_vars)

from pipeline.evaluation import *

TRACT_SHAPEFILE = "../data/Cook_tract_2010.geojson"
BUSINESS_LOCATION_DF = "../data/business_licenses_with_tracts.csv"
INDEX_VARS = ['ACCOUNT NUMBER', 'SITE NUMBER']


def clean_chicago_business_data(self):
    self.logger.info("    Running cleaning steps on raw data")
    self.dataframe = clean_data(self.dataframe, self.data_cleaning)
    return self


def make_chicago_business_features(self):

    original_features = self.features[:]
    old_cols = set(self.test_sets[0].columns)

    # Make features for each dataset in train_sets, test_sets
    n = len(self.feature_generators)
    for df_list in (self.test_sets, self.train_sets):
        for i in range(len(df_list)):
            self.logger.info("    Creating %s features on test-train set %s", n, i+1)

            # Filter out licenses for Jan 1, 2019 onwards
            df_list[i] = df_list[i].loc[df_list[i]['DATE ISSUED'] <= pd.to_datetime('12/31/2018')]
            df_list[i] = make_features(df_list[i], self.feature_generators, original_features)

    # Check for feature balance on each test-train pair
    for i in range(len(self.test_sets)):
        self.logger.info("    Balancing features for test-train set %s", i+1)
        self.train_sets[i], self.test_sets[i] = balance_features(self.train_sets[i], self.test_sets[i])

        # Filter only for business-years in 1 year test window
        self.test_sets[i] = self.test_sets[i].loc[
            self.test_sets[i]['YEAR'] == self.test_sets[i]['YEAR'].max()
        ]

    # Add newly-generated features to self.features
    new_cols = set(self.test_sets[0].columns)
    self.features += list(new_cols - old_cols) # set difference

    self.features = list(set(self.features) - set([self.target]))

    return self


def create_classifer(config):

    pipeline = Pipeline(
        Path(config["data"]["input_path"]),
        config["pipeline"]["target"],
        data_cleaning=[
            to_datetime("LICENSE TERM EXPIRATION DATE"),
            to_datetime("DATE ISSUED")
        ],
        feature_generators=[
            count_by_zip_year,
            make_dummy_vars
        ],
        summarize=False,
        model_grid=Grid.from_config(config["models"]),
        splitter=Splitter.from_config(config["pipeline"]["test_train"]),
        name="best-model-results",
        output_root_dir=Path("output/"))
    pipeline.generate_features = MethodType(make_chicago_business_features, pipeline)
    pipeline.load_data() \
            .clean_data() \
            .summarize_data() \
            .generate_test_train() \
            .preprocess_data() \
            .generate_features() \
            .run_model_grid() \

    return pipeline


def predict_failures(pipeline, k):

    X, y = pipeline.features, pipeline.target

    pipeline.logger.info("    Making predictions")
    n = len(pipeline.test_sets)

    for (description, models) in pipeline.trained_models.items():    
        for (index, (model, split_name, test_set)) in enumerate(zip(models, pipeline.split_names, pipeline.test_sets)):
            pipeline.logger.info("        Predicting on set \"%s\" (%s/%s):", split_name, index + 1, n)
            if type(model) in score_function_overrides.keys():
                score_function = score_function_overrides[type(model)]
                y_score = score_function(self=model, X=test_set[X])
            else:
                y_score = np.array([_[pipeline.positive_label] for _ in model.predict_proba(test_set[X])])

    rv = pipeline.test_sets[0][INDEX_VARS]
    rv['score'] = y_score

    preds_at_k, _ =  generate_binary_at_k(y_score, k)
    rv['label'] = preds_at_k
    
    return rv


def join_to_tracts(results, config):
    ''' takes results, joins to dataframe with tract location
        Returns: count of business failures by tract
    '''

    tracts = pd.read_csv(BUSINESS_LOCATION_DF).drop(columns = "Unnamed: 0")
    tracts['DATE ISSUED'] = pd.to_datetime(tracts['DATE ISSUED'], format="%m/%d/%Y")
    tracts["GEOID"] = tracts["GEOID_2010"].astype(str)
    predict_start_date = pd.to_datetime(config["pipeline"]["test_train"]["splits"][0]["test"]["start"])

    tracts = tracts.loc[tracts['DATE ISSUED'] > predict_start_date]
    tracts = tracts.drop(columns = list(set(tracts.columns).difference(set(INDEX_VARS + ["GEOID"])))) 
    
    df = pd.merge(results, tracts, on=INDEX_VARS, how = "left")

    return df.groupby("GEOID").sum()


def get_top_tracts(df, num_tracts):

    return df.sort_values("label", ascending = False).iloc[0:num_tracts]


def map_top_tracts(geo_results, shp):

    tract_map = shp.merge(geo_results, how="left")
    
    # drop tract that is all water
    tract_map = tract_map.loc[tract_map["GEOID"] != "17031990000"]

    # Plot the map
    ax3 = tract_map.plot(column = "label",
                        edgecolor='white',
                        figsize=(12, 10),
                        legend=True)

    # Some visual tweaks
    plt.title('Predicted business failures by tract')
    plt.axis('off')
    plt.show()

    return 



if __name__ == "__main__":
    
    with open("analysis.yml", 'rb') as config_file:
        config = yaml.safe_load(config_file.read())

    pipeline = create_classifer(config)

    results = predict_failures(pipeline, config['k'])
    geo_results = join_to_tracts(results, config).reset_index()

    results.to_csv("results.csv")
    geo_results.to_csv("geo_results.csv")

    top_k_tracts = get_top_tracts(geo_results, config['num_tracts'])

    Cook_tracts_shp = gpd.read_file(TRACT_SHAPEFILE)

    # gdf = map_top_tracts(Cook_tracts_shp, geo_results)