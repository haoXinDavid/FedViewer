import requests
from bs4 import BeautifulSoup
import os
import re
from multiprocessing import Pool
import csv
import glob

os.chdir('.')

url_main = 'https://www.federalreserve.gov'
newurl = url_main + '/monetarypolicy/fomccalendars.htm'
r = requests.get(newurl)
soup = BeautifulSoup(r.text, "lxml")
stat_link = soup.findAll('a', href=re.compile('^/newsevents/pressreleases/monetary\d{8}a.htm'))


def saveFile(fname, year, text):
    file = open(fname + '.txt', 'w', errors='ignore')

    text_clean = '\n'.join(t.text for t in text)
    text_clean = re.sub(r'\r\n', ' ', text_clean)
    text_clean = re.sub(r'(?<!\n)\n(?!\n)', r'\n\n', text_clean)

    file.write(text_clean)
    file.close()


for m in stat_link:
    nurl = str(m['href']) if 'federalreserve.gov' in str(m['href']) else url_main + str(m['href'])
    filename = re.search("(\d{4})(\d{2})(\d{2})", nurl)[0]
    soup = BeautifulSoup(requests.get(nurl).content, 'lxml')
    main_text = []
    i = 0
    year = 0
    while not main_text and i < 3:
        m0 = {'name': 'div', 'attrs': {'id': 'article'}}
        m1 = {'name': 'div', 'attrs': {'id': 'leftText'}}
        m2 = {'name': 'p'}
        main_text = soup.findAll(**eval('m' + str(i)))
        i += 1
    saveFile(filename, year, main_text)

for year in list(range(1994, 2015)):
    url_main = 'https://www.federalreserve.gov'
    ourl = url_main + '/monetarypolicy/fomchistorical' + str(year) + '.htm'
    o = requests.get(ourl)
    soup = BeautifulSoup(o.text, "lxml")
    ostat_link = soup.findAll('a', href=re.compile(str(year)), string=re.compile('Statement.*'))

    for m in ostat_link:
        nurl = str(m['href']) if 'federalreserve.gov' in str(m['href']) else url_main + str(m['href'])
        filename = re.search("(\d{4})(\d{2})(\d{2})", nurl)[0]
        soup = BeautifulSoup(requests.get(nurl).content, 'lxml')
        main_text = []
        i = 0
        while not main_text and i < 3:
            m0 = {'name': 'div', 'attrs': {'id': 'article'}}
            m1 = {'name': 'div', 'attrs': {'id': 'leftText'}}
            m2 = {'name': 'p'}
            main_text = soup.findAll(**eval('m' + str(i)))
            i += 1
        saveFile(filename, year, main_text)