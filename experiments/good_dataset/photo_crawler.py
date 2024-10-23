import requests
import pandas as pd

from bs4 import BeautifulSoup
from tqdm import tqdm

def process_new_collection(link: str) -> list[str]:
    data = requests.get(link)
    parsed_html = BeautifulSoup(data.content, features="html.parser")

    html_img_tags = parsed_html.find_all("img", recursive=True)
    images = [link["src"] for link in html_img_tags if "amazonaws" in link["src"]]

    return images

def process_old_collection(link: str) -> list[str]:
    data = requests.get(link)
    parsed_html = BeautifulSoup(data.content, features="html.parser")

    link_template = "https://photoawards.com/winner/{link}"

    images = []
    photo_links = parsed_html.find_all("a", {"class": "opacityit"}, recursive=True)
    for photo_link in photo_links:
        data = requests.get(link_template.format(link=photo_link["href"]))
        parsed_html = BeautifulSoup(data.content, features="html.parser")
        html_img_tags = parsed_html.find_all("img", recursive=True)
        for link in html_img_tags:
            if "amazonaws" in link["src"]:
                images.append(link["src"])

    return images

def process_collection(link: str) -> list[str]:
    if ("zoom.php" in link):
        return process_new_collection(link)
    elif ("zoomOLD.php" in link):
        return process_old_collection(link)
    
    raise ValueError(f"Unrecognized link format! {link}")

def process_collections_in_year(df: pd.DataFrame) -> pd.DataFrame:
    df["photo_url"] = [[] for _ in range(len(df))]
    print(f"Now processing year: {df.at[0, 'year']}")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        df.at[idx, "photo_url"] = process_collection(row["collection_url"])

    return df.explode("photo_url").dropna(subset="photo_url")

def main() -> None:
    years = list(range(2004, 2025))[::-1]
    for year in years:
        df_year = pd.read_csv(f"collection_{year}.csv", sep="ඞ", index_col=0, engine="python")
        df_photos_year = process_collections_in_year(df_year)
        df_photos_year.to_csv(f"photos_{year}.csv", sep="ඞ")
        

if __name__ == "__main__":
    main()