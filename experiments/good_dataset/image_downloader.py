import pandas as pd
import requests
import os

from concurrent.futures import ThreadPoolExecutor

def build_directories(year: int) -> None:
    df_year = pd.read_csv(f"photos_{year}.csv", sep="ඞ", index_col=0, engine="python")
    for idx, row in df_year.iterrows():
        clean_main_category = row["main_category"].replace("/", "_")
        clean_secondary_category = row["secondary_category"].replace("/", "_")
        os.makedirs(f"{year}/{clean_main_category}/{clean_secondary_category}", exist_ok=True)

def download_imgs(url: str, year: int, main_cat: str, second_cat: str) -> str | None:
    try:
        data = requests.get(url)
        if (data.status_code == 200):
            path = f"{year}/{main_cat}/{second_cat}/{os.path.basename(url)}"
            with open(path, "wb") as f:
                f.write(data.content)
            return None
        
        else:
            return url

    except:
        return url


def download_year_images(year: int) -> None:
    df_year = pd.read_csv(f"photos_{year}.csv", sep="ඞ", index_col=0, engine="python")
    urls = df_year["photo_url"].to_list()
    years = [year] * len(urls)
    main_cats = df_year["main_category"].str.replace("/", "_").to_list()
    second_cats = df_year["secondary_category"].str.replace("/", "_").to_list()

    if (len({len(i) for i in [urls, years, main_cats, second_cats]}) != 1):
        print(f"Error in year {year}: data lists are not same length!")
        return -1
    
    with ThreadPoolExecutor() as executor:
        print(f"Downloading images from {year}. This might take some time...")
        result = executor.map(download_imgs, urls, years, main_cats, second_cats)

    errors = [url for url in result if url is not None]
    if errors:
        print(f"Warning: some photos were not downloaded correctly in year {year}. Check errors_{year}.txt for more info.")
        with open(f"errors_{year}.txt", "w") as f:
            for error in errors:
                f.write(f"{error}\n")

    else:
        print(f"All photos in year {year} downloaded successfully!")

def main() -> None:
    years = list(range(2004, 2025))[::-1]
    for year in years:
        build_directories(year)

    # Download them images!
    for year in years:
        download_year_images(year)

if __name__ == "__main__":
    main()