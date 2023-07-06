import requests
import pandas as pd
from stqdm import stqdm
from tqdm.auto import tqdm
from urllib.parse import urlparse
from datetime import datetime
from bs4 import BeautifulSoup

BUSINESS_URL = 'https://nypost.com/business'
TECH_URL = 'https://nypost.com/tech'
SPORTS_URL = 'https://nypost.com/sports'
METRO_URL = 'https://nypost.com/metro'
ENTERTAINMENT_URL = 'https://nypost.com/entertainment'
OPINION_URL = 'https://nypost.com/opinion'
URLS = {
    'business': BUSINESS_URL, 
    'tech': TECH_URL, 
    'sports': SPORTS_URL, 
    'metro': METRO_URL, 
    'entertainment': ENTERTAINMENT_URL, 
    'opinion': OPINION_URL
}


def convert_datetime(datetime_str):
    # Parse the input datetime string
    datetime_obj = datetime.strptime(datetime_str, "%B %d, %Y | %I:%M%p")
    
    # Convert datetime to the desired format
    formatted_datetime = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
    date, time = formatted_datetime.split()
    
    return date, time


def get_one_page(page, base_url):
    res = requests.get(f'{base_url}/page/{page}')
    soup = BeautifulSoup(res.text, 'html.parser')
    stories = soup.find_all('div', class_='story__text')
    data_dict = []
    for story in stories:
        try:
            title = story.find('a').text.strip()
            link = story.find('a').get('href')
            timestamp = story.find('span')
            if timestamp is not None:
                timestamp = timestamp.text.strip()
                date, time = convert_datetime(timestamp)
            else:
                parsed_url = urlparse(link)
                path = parsed_url.path
                date_components = path.split("/")[1:4]
                year, month, day = map(int, date_components)
                date = datetime(year, month, day).strftime("%Y-%m-%d")
                time = '00:00:00'
            data_dict.append({
                'title': title, 
                'date': date, 
                'time': time, 
                'link': link, 
                'source': 'nypost'
            })
        except:
            continue
    return data_dict


def get_multi_pages(pages, base_url):
    data_dict = []
    pbar = stqdm(range(pages))
    for page in pbar:
        pbar.set_description(f'PAGE {page}')
        data_dict.extend(get_one_page(page, base_url))
    return data_dict


def main():
    data_dict = get_multi_pages(5, BUSINESS_URL)
    df = pd.DataFrame(data_dict)
    print(df)


if __name__ == '__main__':
    main()