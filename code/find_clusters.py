import yaml
import pandas as pd 
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from chicago_business import get_config, get_pipeline

TRACT_SHAPEFILE = "../data/Cook_tract_2010.geojson"
BUSINESS_LOCATION_DF = "../data/business_licenses_with_tracts.csv"
INDEX_VARS = ['ACCOUNT NUMBER', 'SITE NUMBER']


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

    geo_results["GEOID"] = geo_results["GEOID"].astype(str)
    tract_map = shp.merge(geo_results, how="left")

    # NA's here are actually zeroes since we left joined;
    gdf = gdf.fillna(0)
    
    # drop tract that is all water
    tract_map = tract_map.loc[tract_map["GEOID"] != "17031990000"]

    # transform to log
    gdf['log-label'] = np.log(gdf['label'] + 1)

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
    
    config, script_dir = get_config("config.yml")

    pipeline = get_pipeline(config, script_dir)
    pipeline.run()

    results = predict_failures(pipeline, config['k'])
    geo_results = join_to_tracts(results, config).reset_index()

    results.to_csv("results.csv")
    geo_results.to_csv("geo_results.csv")

    top_k_tracts = get_top_tracts(geo_results, config['num_tracts'])

    Cook_tracts_shp = gpd.read_file(TRACT_SHAPEFILE)

    # gdf = map_top_tracts(Cook_tracts_shp, geo_results)