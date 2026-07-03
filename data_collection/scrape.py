import sys
sys.stdout.reconfigure(encoding="utf-8")

from bs4 import BeautifulSoup
import requests
import time
import numpy as np
import json

headers = {
    "User-Agent": "ALD Database Creation Bot (Academic)"
}

url = "https://www.atomiclimits.com/alddatabase/api/processes.php"
html = requests.get(url, headers=headers).text
soup = BeautifulSoup(html, "lxml")


def get_responses():
    '''
    func, get_responses: no input -> returns json of references,processes.
    '''

    url = "https://www.atomiclimits.com/alddatabase/api/processes.php"
    html = requests.get(url, headers=headers).text
    soup = BeautifulSoup(html, "lxml")

    data = json.loads(html)
    references = data["references"]
    processes = data["processes"]

    return references,processes

