import requests
from bs4 import BeautifulSoup
import os
import re
from multiprocessing import Pool
import csv
import glob

os.chdir('.')


def minarchive(year):
    url_main = 'https://www.federalreserve.gov'
    year = int(year)

    if year > 2015:  # Post 2015 has a different URL
        url = url_main + '/monetarypolicy/fomccalendars.htm'
    else:
        url = url_main + '/monetarypolicy/fomchistorical' + str(year) + '.htm'

    r = requests.get(url)
    soup = BeautifulSoup(r.text, "lxml")
    if year > 1996:
        min_link = soup.findAll('a', href=re.compile('(?=.*minutes)(?=.*' + str(year) + ')'),
                                text=lambda text: text and 'PDF' not in text)
    else:
        min_link = soup.findAll('a', href=re.compile('(?=.*MINUTES)(?=.*' + str(year) + ')'),
                                text=lambda text: text and 'PDF' not in text)

    # Methods
    m0 = {'name': 'div', 'attrs': {'id': 'article'}}
    m1 = {'name': 'div', 'attrs': {'id': 'leftText'}}
    m2 = {'name': 'p'}

    for m in min_link:
        url = str(m['href']) if 'federalreserve.gov' in str(m['href']) else url_main + str(m['href'])
        filename = re.search("(\d{4})(\d{2})(\d{2})", url)[0]
        soup = BeautifulSoup(requests.get(url).content, 'lxml')

        i = 0

        main_text = []
        while not main_text and i < 5:
            main_text = soup.findAll(**eval('m' + str(i)))
            i += 1
        saveFile(filename, year, main_text)


def saveFile(fname, year, text):
    file = open(fname + '.txt', 'w', errors='ignore')

    text_clean = '\n'.join(t.text for t in text)
    text_clean = re.sub(r'\r\n', ' ', text_clean)
    text_clean = re.sub(r'(?<!\n)\n(?!\n)', r'\n\n', text_clean)

    file.write(text_clean)
    file.close()


if __name__ == '__main__':
    start_year, end_year = 2020, 1993
    p = Pool(processes=(start_year - end_year))
    p.map(minarchive, range(start_year, end_year - 1, -1))
    p.close()
