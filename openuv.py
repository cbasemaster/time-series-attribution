# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 09:56:42 2020

@author: yudis
"""

import pandas as pd
from bs4 import BeautifulSoup, SoupStrainer
import requests
import pickle

url = "http://www.temis.nl/uvradiation/UVarchive/stations_uv.html"

page = requests.get(url)    
data = page.text
soup = BeautifulSoup(data)
myKeys = []
myValues = []


f = open("dfuv.pkl","wb")
pickle.dump(finaluv,f)
f.close()
"""
for no,link in enumerate(soup.find_all('a')):
    #print(link.get('href'))
    url = 'http://www.temis.nl/uvradiation/'+str(link.get('href'))[3:]
    print (url)
    if ('.dat' in url):
        dfuv = pd.read_table(url,skiprows=29+6474,header=None,delimiter=",")
        dfuv = dfuv[dfuv.columns[0]].str.split(expand=True)
        dfuv.columns = ['YYYYMMDD','UVIEF','UVIEFerr','UVDEF','UVDEFerr','UVDEC','UVDECerr','UVDVF','UVDVFerr','UVDVC','UVDVCerr','UVDDF','UVDDFerr','UVDDC','UVDDCerr','CMF','ozone']
        name=url.split("_")[-1].split(".")[0]
        myKeys.append(name)
        myValues.append(dfuv)
finaluv=dict(zip(myKeys, myValues))
"""