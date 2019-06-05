# script for gluing together features

from pathlib import Path
import pandas as pd
from sklearn.utils import shuffle

INDEX = [
    "ACCOUNT NUMBER",
    "SITE NUMBER",
    "YEAR"
]

def join_together(dataframes, cols_to_drop):
    ''' Takes: list of dataframes, list of columns to drop from any input dataframe

        Returns: one merged dataframe
    '''

    merged, *dfs = dataframes

    for df in dfs:

        df.drop(columns = cols_to_drop, inplace=True, errors='ignore')
        # print("df shape before dropping dups", df.shape)
        df.drop_duplicates(INDEX, inplace=True)
        # print("df shape after dropping dups", df.shape)

        merged = merged.merge(shuffle(df), on=INDEX, how = "left", validate = 'many_to_one')

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
    
    data = Path("../data")
    
    orig_business_df = pd.read_csv(data/"Business_Licenses.csv")
    orig_business_df['YEAR'] = pd.to_datetime(orig_business_df['DATE ISSUED']).dt.year

    input_filenames = ["merged_business_govtdata.csv", "licenses_joined.csv"]
    output_filename = data/"joined_table.csv"

    input_paths = [data/filename for filename in input_filenames]
    dataframes = [orig_business_df] + [pd.read_csv(path) for path in input_paths]

    cols_to_drop = ['Unnamed: 0', 'LICENSE NUMBER']

    merged = join_together(dataframes, cols_to_drop)
    merged.to_csv(output_filename)
