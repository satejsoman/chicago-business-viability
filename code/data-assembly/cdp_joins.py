# script to join together various business attributes from CDP datasets
# [X] number of previous license renewals 
# [X] number of unique sites 
# [X] in SSA (binary)
# [X] which SSA (categorical)
# [X] Number of days between LICENSE APPROVED FOR ISSUANCE and DATE ISSUED
# [X] Number of business owned by owner
# [ ] received TIF
# [ ] Received Small Business Improvement Fund (SBIF) grant
# [d] is national chain

from pathlib import Path

import pandas as pd
from tqdm import tqdm

tqdm.pandas()

def process_datetimes(df):
    dt_cols = ['LICENSE APPROVED FOR ISSUANCE', 'DATE ISSUED']
    for col in dt_cols:
        df[col] = df[col].progress_apply(lambda row: pd.to_datetime(row, errors="coerce", format="%m/%d/%Y"))
    df["YEAR"] = df["DATE ISSUED"].dt.year
    df["days_between_approval_issuance"] = (df['DATE ISSUED'] - df['LICENSE APPROVED FOR ISSUANCE']).dt.days.fillna(0).astype(int)
    return df

def process_business_owners(df):
    df[["Owner First Name", "Owner Middle Initial", "Owner Last Name", "Legal Entity Owner"]] = df[["Owner First Name", "Owner Middle Initial", "Owner Last Name", "Legal Entity Owner"]].fillna("")
    df["full_name"] = (df["Owner First Name"] + " " + df["Owner Middle Initial"] + " " + df["Owner Last Name"] + " " + df["Legal Entity Owner"]).replace('\s+', ' ', regex=True)
    df_filtered = df[~(df["full_name"].isin(["Business Owner", ""]))]
    df_filtered["num_businesses_on_account"] = df_filtered["full_name"].map(df["full_name"].value_counts())
    return df_filtered[["Account Number", "num_businesses_on_account"]]
 
def get_num_renewals_map(df):
    # subsequent renewal and license granted
    renewals = df[(df["APPLICATION TYPE"] == "RENEW") & (df["LICENSE STATUS"] == "AAI")]
    return renewals.groupby("license_site")["LICENSE TERM EXPIRATION DATE"].apply(len)

def non_join_transformations(df):
    df["which_ssa"] = df.SSA.fillna(0)
    df["in_ssa"] = (df["which_ssa"] > 0).astype(int)
    df["num_sites"] = df.groupby("ACCOUNT NUMBER")["SITE NUMBER"].transform('count')
    df["license_site"] = df["LICENSE NUMBER"].astype(str) + "-" + df["SITE NUMBER"].astype(str)
    return df 

def main(data_path, licenses, owners):
    # licenses = process_datetimes(licenses)
    licenses = non_join_transformations(process_datetimes(licenses))
    renewal_counts = get_num_renewals_map(licenses)
    licenses["num_renewals"] = licenses["license_site"].map(renewal_counts).fillna(0)
    
    licenses = licenses.merge(process_business_owners(owners), left_on="ACCOUNT NUMBER", right_on="Account Number")

    # tif_relevant_columns = tif[['APPROVED AMOUNT', 'TOTAL PROJECT COST', 'LOCATION']].fillna(-1)
    # lmerge = licenses.merge(tif_relevant_columns, on="LOCATION")
    # licenses[['tif_approved_amount', 'tif_total_cost']] = tif[['APPROVED AMOUNT', 'TOTAL PROJECT COST']].fillna(0)
    
    licenses[["ACCOUNT NUMBER", "SITE NUMBER", "YEAR",'which_ssa', 'in_ssa', 'num_sites', 'num_renewals']].to_csv(data_path/'licenses_joined.csv')

if __name__ == "__main__":
    data = Path("./data")
    licenses_path   = data/"Business_Licenses.csv"
    owners_path     = data/"Business_Owners.csv"
    tif_path        = data/"TIF.csv"
    fortune1k_path  = data/"fortune1000.csv"
    
    main(
        data_path = data,
        licenses  = pd.read_csv(licenses_path, low_memory=False), 
        owners    = pd.read_csv(owners_path,   low_memory=False))
