# script to join together various business attributes from CDP datasets
# [x] number of previous license renewals 
# [X] number of unique sites 
# [X] in SSA (binary)
# [X] which SSA (categorical)
# [ ] Number of days between LICENSE APPROVED FOR ISSUANCE and DATE ISSUED
# [x] Number of business owned by owner
# [ ] received TIF
# [ ] Received Small Business Improvement Fund (SBIF) grant
# [ ] is national chain

from pathlib import Path
import pandas as pd

def process_datetimes(df):
    # dt_cols = [col for col in df.columns if 'DATE' in col]
    dt_cols = ['LICENSE APPROVED FOR ISSUANCE', 'DATE ISSUED']
    df[dt_cols] = pd.to_datetime(df[dt_cols])
    return df

def process_business_owners(df):
    df[["Owner First Name", "Owner Middle Initial", "Owner Last Name"]] = df[["Owner First Name", "Owner Middle Initial", "Owner Last Name"]].fillna(" ")
    df["full_name"] = df["Owner First Name"] + " " + df["Owner Middle Initial"] + " " + df["Owner Last Name"]
    df["num_accounts"] = df["full_name"].map(df["full_name"].value_counts())
    return df 
 
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

def main(licenses, owners):
    # licenses = process_datetimes(licenses)
    licenses = non_join_transformations(licenses)
    owners = process_business_owners(owners)
    owners.rename(str.upper, axis="columns", inplace=True)
    # licenses = licenses.

if __name__ == "__main__":
    data = Path("./data")
    licenses_path   = data/"Business_Licenses.csv"
    owners_path     = data/"Business_Owners.csv"
    fortune1k_path  = data/"fortune1000.csv"
    
    main(
        licenses = pd.read_csv(licenses_path), 
        owners   = pd.read_csv(owners_path))