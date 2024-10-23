import pandas as pd
import os

def clean_and_build_dataset(year: int) -> pd.DataFrame:
    # Clean dataset from images that could not be downloaded
    df = pd.read_csv(f"photos_{year}.csv", sep="ඞ", index_col=0, engine="python")
    if os.path.exists(f"errors_{year}.txt"):
        with open(f"errors_{year}.txt", "r") as f:
            error_lines = f.read().splitlines()

        df = df[~df["photo_url"].isin(error_lines)]

    # Build a dataset in a format suitable for the PU library
    pu_df = pd.DataFrame()
    pu_df["collection_name"] = df["collection_name"]
    pu_df["author_and_country"] = df["name_and_country"]
    pu_df["ranking"] = df["place"].apply(lambda val: int(val[0]))
    pu_df["year"] = df["year"]
    pu_df["main_category"] = df["main_category"]
    pu_df["secondary_category"] = df["secondary_category"]
    pu_df["path"] = df.apply(lambda row: f"{year}/{row['main_category'].replace('/', '_')}/{row['secondary_category'].replace('/', '_')}/{os.path.basename(row['photo_url'])}", axis=1)

    return pu_df

def main():
    years = list(range(2004, 2025))[::-1]
    datasets = []
    for year in years:
        datasets.append(clean_and_build_dataset(year))

    df = pd.concat(datasets)
    df.to_csv("cima.csv", sep="ඞ")

if __name__ == "__main__":
    main()