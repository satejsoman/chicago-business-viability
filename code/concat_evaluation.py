import os
import argparse
import pandas as pd 
from pipeline.evaluation import find_best_model


def get_csvs(pattern):
    ''' walk through specified directories and read in csvs
        return concatenated file
    '''

    dirs = ["output/" + pattern + str(x) + "-LATEST" for x in range(1, 8)]

    files = []
    for d in dirs:
        print(d + "/" + "evaluations.csv")
        if os.path.isfile(d + "/" + "evaluations.csv"):
            
            files.append(pd.read_csv(d + "/" + "evaluations.csv"))

    return pd.concat(files).drop(columns = "Unnamed: 0")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", help = "directory path pattern")
    args = parser.parse_args()

    if not os.path.exists("output/" + args.pattern + "LATEST"):
        os.mkdir("output/" + args.pattern + "LATEST")

    final = get_csvs(args.pattern)
    final.to_csv("output/" + args.pattern + "LATEST/evaluations.csv")

    print(find_best_model(final, "recall", 0.05))
