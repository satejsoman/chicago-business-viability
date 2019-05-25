# script for gluing together features

from pathlib import Path
import pandas as pd

INDEX = [
    "ACCOUNT NUMBER", 
    "SITE NUMBER",
    "YEAR"
]

def join_together(dataframes):
    merged, *dfs = dataframes
    for df in dfs:
        merged = merged.merge(df, on=INDEX)
    return merged

def test():
    df1 = pd.DataFrame({
        "ACCOUNT NUMBER": ["acc" + str(n+1) for n in [1, 1, 1, 2, 3, 4, 5, 6, 6, 7]],
        "SITE NUMBER"   : [1, 2, 3, 1, 1, 1, 1, 1, 2, 1],
        "YEAR" : ([2002] * 8) + [2003, 2004]
    })

    df2 = df1.copy(deep=True)
    df2["wubwub"] = pd.Series(["womp", "w", "o", "m", "p", "womp", "w", "o", "m", "p"])

    df3 = df1.copy(deep=True)
    df3["letters"] = pd.Series(list('dkdjahebne'))

    print(join_together([df1, df2, df3]))

if __name__ == "__main__":
    data = Path("./data")
    input_filenames = []
    output_filename = "joined_table.csv"
    
    input_paths = [data/filename for filename in input_filenames]
    dataframes = [pd.read_csv(path) for path in input_paths]

    join_together(dataframes).to_csv(output_filename)