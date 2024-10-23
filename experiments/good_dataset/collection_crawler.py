import requests
import pandas as pd

from bs4 import BeautifulSoup

request_data_formatter = "limit=1&start={page}&compName={year}&level=pro"
request_headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
}

def map_year_to_request(year: int) -> str:
    return f"IPA+{str(year)}" if year >= 2009 else str(year)[-1]

def request_page_year(year: int, page: int) -> requests.Response:
    request_data = request_data_formatter.format(page=page, year=map_year_to_request(year))
    data = requests.post("https://photoawards.com/winner/fetch_winners.php", 
                         data=request_data,
                         headers=request_headers,
                         cookies={},
                         auth=()
    )

    return data

def parse_contest_year(year: int) -> pd.DataFrame:
    photo_collections_dict = {
        "collection_url": [],
        "name_and_country": [],
        "collection_name": [],
        "place": [],
        "main_category": [],
        "secondary_category": [],
        "year": []
    }

    page = 0
    data = request_page_year(year, page)

    print(f"Now parsing: {year}")

    while(data.content != b'\n'):
        parsed_html = BeautifulSoup(data.content)
        gallery_titles   = parsed_html.find_all("div", {"class": "gallery-title"})
        gallery_wrappers = parsed_html.find_all("div", {"class": "gallery-wrapper"})
        if (len(gallery_titles) != len(gallery_wrappers)):
            print(f"Error in year {year}, page {page}: titles and wrappers do not match!")
            continue

        for title, wrapper in zip(gallery_titles, gallery_wrappers):
            try:
                categories = title.text.split(",")
                main_category = ' '.join(categories[0].split()[1:])
                secondary_category = categories[1]

                for winner in wrapper.find_all("div", {"class": "winner-item"}):
                    place = winner.find("p", {"class": "label"}).text
                    url_name_country_title = winner.find("a", {"class": "copy"})
                    url = f"https://photoawards.com/winner/{url_name_country_title['href']}"
                    name_country = url_name_country_title.find("h3").text
                    title = url_name_country_title.find("p").text
                    
                    photo_collections_dict["collection_url"].append(url)
                    photo_collections_dict["name_and_country"].append(name_country)
                    photo_collections_dict["collection_name"].append(title)
                    photo_collections_dict["place"].append(place)
                    photo_collections_dict["main_category"].append(main_category)
                    photo_collections_dict["secondary_category"].append(secondary_category)
                    photo_collections_dict["year"].append(year)

            except Exception as e:
                print(f"Error while parsing category {title} in year {year}, page {page}: {e}")
        
        page += 1
        data = request_page_year(year, page)

    df = pd.DataFrame.from_dict(photo_collections_dict)
    return df
        

def main() -> None:
    years = list(range(2004, 2025))[::-1]
    for year in years:
        df_year = parse_contest_year(year)
        df_year.to_csv(f"collection_{year}.csv", sep="ඞ")


if __name__ == "__main__":
    main()