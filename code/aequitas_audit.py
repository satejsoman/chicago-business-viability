import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.group import Group
from aequitas.plotting import Plot
from aequitas.preprocessing import preprocess_input_df
from sklearn.tree.tree import DecisionTreeClassifier

def load_model(path: str) -> DecisionTreeClassifier:
    return joblib.load(path)[0]

def load_features(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

if __name__ == "__main__":
    model_path   = "best_models_DecisionTreeClassifier-max_depth1.joblib"
    feature_path = "best_model_test.csv"
    model = load_model(model_path)

    df = load_features(feature_path)
    df = (df
        .rename(columns = {"not_renewed_2yrs" : "label_value"})
        .drop(columns = ['Unnamed: 0', 'ACCOUNT NUMBER', 'SITE NUMBER', 'YEAR'])
        .fillna(0))

    fixed = df[df.columns[df.columns != "label_value"]]
    scores = model.predict_proba(fixed)
    df["score"] = [_[1] for _ in scores]

    df.to_csv("aequitas_scored.csv")

    df, cols = preprocess_input_df(df)

    g = Group()
    xtab, _ = g.get_crosstabs(df, attr_cols=["num_renewals"])
    print(xtab)
    absolute_metrics = g.list_absolute_metrics(xtab)
    print(xtab[[col for col in xtab.columns if col not in absolute_metrics]])