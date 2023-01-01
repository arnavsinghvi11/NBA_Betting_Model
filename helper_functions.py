from bs4 import BeautifulSoup
import datetime
from datetime import datetime
from datetime import timedelta
import pytz
from pytz import timezone
import requests

def abbrv(col):
    if ' ' in col:
        col = col.split(' ')[0][0] + '.' + col.split(' ')[1]
    return col

def site_scrape(url):
    headers = requests.utils.default_headers()
    headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
    })
    page = requests.get(url, headers = headers)
    soup = BeautifulSoup(page.content, "html.parser")
    return soup