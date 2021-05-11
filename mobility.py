# -*- coding: utf-8 -*-
"""
Created on Sun May 17 22:25:00 2020

@author: yudis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize
from numpy.random import rand
#matplotlib inline
import csv
import scipy.stats as stats


dki=[183,215,267,307,351,424,463,515,589,627,675,698,747,808,897,971,1028,1124,1131,1369,1470,1706,1753,1948,2044,2186,2335,2474,2670,2815,2924,3032,3097,3260,3383,3517,3599,3684,3798,3869,4022,4092,4175,4317,4397,4463,4539,4687,4770,4855,4955,5056,5190,5276,5375,5554,5688,5774,5881,6010,6059,6155,6236,6301,6400,6515,6636,6709,6798,6895,7001,7128]
bali=[27,32,35,36,43,49,63,75,79,81,86,92,98,113]

dki=pd.DataFrame(dki)
bali=pd.DataFrame(bali)

df = pd.read_csv('data/waqi-covid19-airqualitydata-jakarta.csv', skiprows=1,error_bad_lines=False)
df.head()
df.info()
df.columns = ['Date','Country','City','Specie','count','min','max','median','variance']
df.head
#print df.day
df.fillna
df.Date = pd.to_datetime(df.Date)
df.set_index('Date', inplace=True)
######################################################################################
dfm = pd.read_csv('data/Global_Mobility_Report-1.csv', skiprows=1)
dfm.head()
dfm.info()

dfm.columns = ["country_region_code","country_region","sub_region_1","sub_region_2","date","retail_and_recreation_percent_change_from_baseline","grocery_and_pharmacy_percent_change_from_baseline","parks_percent_change_from_baseline","transit_stations_percent_change_from_baseline","workplaces_percent_change_from_baseline","residential_percent_change_from_baseline"
]
dfm.head
#print df.day
dfm.fillna(0)
dfm.date = pd.to_datetime(dfm.date)
dfm.set_index('date', inplace=True)

dkidiff=dki.diff().copy().fillna(0)
print (dkidiff[0][1])
dki=dkidiff.apply(lambda x: (x.subtract(dkidiff[0][1]).div(dkidiff[0][1])).mul(100))
dki=dki.fillna(0)
#######################################################################################
is_jkt =  (dfm['country_region']=='Indonesia') & (dfm['sub_region_1']=='Jakarta')
dfmjkt_a= dfm[(is_jkt)][['retail_and_recreation_percent_change_from_baseline']]
dfmjkt_b= dfm[(is_jkt)][['grocery_and_pharmacy_percent_change_from_baseline']]
dfmjkt_c= dfm[(is_jkt)][['parks_percent_change_from_baseline']]
dfmjkt_d= dfm[(is_jkt)][['transit_stations_percent_change_from_baseline']]
dfmjkt_e= dfm[(is_jkt)][['workplaces_percent_change_from_baseline']]
dfmjkt_f= dfm[(is_jkt)][['residential_percent_change_from_baseline']]

#######################################################################################
is_jkt25 =  (df['City']=='Jakarta') & (df['Specie']=='pm25')
is_jkt10 =  (df['City']=='Jakarta') & (df['Specie']=='pm10')
is_jkthum =  (df['City']=='Jakarta') & (df['Specie']=='humidity')
is_jktwind =  (df['City']=='Jakarta') & (df['Specie']=='wind speed')
is_jktwind_ =  (df['City']=='Jakarta') & (df['Specie']=='wind-speed')
is_jkttemp =  (df['City']=='Jakarta') & (df['Specie']=='temperature')

dfjkt25= df[(is_jkt25)][['median']]
#print (dfjkt25.iloc[0])
dfjkt25=((dfjkt25.sort_values('Date')-137)/137)*100.0
#print (dfjkt25)
dfjkt10= df[(is_jkt10)][['median']]
#print (dfjkt10)
dfjkt10=((dfjkt10.sort_values('Date')-34)/34)*100.0

dfjkthum= df[(is_jkthum)][['median']]
#print (dfjkthum)
dfjkthum=((dfjkthum.sort_values('Date')-83.8)/83.8)*100.0

dfjktwind= df[(is_jktwind)][['median']]
dfjktwind_= df[(is_jktwind_)][['median']]

dfjktwind=dfjktwind_.append(dfjktwind)
#print (dfjkthum)
dfjktwind=((dfjktwind.sort_values('Date')-1.5)/1.5)*100.0

dfjkttemp= df[(is_jkttemp)][['median']]
#print (dfjkt10)
dfjkttemp=((dfjkttemp.sort_values('Date')-27.1)/27.1)*100.0
print (dfjktwind)
print (len(dfmjkt_a[34:].rolling(10).mean()),len(dfjkt25[81:-6].rolling(10).mean()),len(dfjkt10[69:].rolling(10).mean()),len(dki[:-1].rolling(10).mean()))
#trends######################################################################################


#######################################################################################
fig, axl = plt.subplots(figsize=(15,10))
axl.plot(dfmjkt_a.index[34:],dki[:-5].rolling(5).mean(),color='black', label='Jakarta confirmed cases')
axl.plot(dfmjkt_a[34:].rolling(5).mean(),color='orange', label='retail_and_recreation_percent_change_from_baseline')
axl.plot(dfmjkt_b[34:].rolling(5).mean(),color='green', label='grocery_and_pharmacy_percent_change_from_baseline')
axl.plot(dfmjkt_c[34:].rolling(5).mean(),color='purple', label='parks_percent_change_from_baseline')
axl.plot(dfmjkt_d[34:].rolling(5).mean(),color='cyan', label='transit_stations_percent_change_from_baseline')
axl.plot(dfmjkt_e[34:].rolling(5).mean(),color='red', label='workplaces_percent_change_from_baseline')
axl.plot(dfmjkt_f[34:].rolling(5).mean(),color='yellow', label='residential_percent_change_from_baseline')
axl.plot(dfjkthum[63:-6].rolling(5).mean(),color='purple',linestyle='--', label='Jakarta mobility percent change from baseline')
axl.plot(dfjkt25[81:-6].rolling(5).mean(),color='blue',linestyle='--',  label='Jakarta mobility percent change from baseline')
axl.plot(dfjktwind[66:-6].rolling(5).mean(),color='green',linestyle='--',  label='Jakarta mobility percent change from baseline')
axl.plot(dfjkttemp[63:-6].rolling(5).mean(),color='red',linestyle='--',  label='Jakarta mobility percent change from baseline')
#axl.plot(dfjkthum,color='yellow',linestyle='--', label='Jakarta mobility percent change from baseline')
#axl.plot(dfmtyo,color='green', label='Tokyo mobility percent change from baseline')
#axl.plot(dfmsin,color='red', label='Singapore mobility percent change from baseline')
#axl.plot(dfmny[0:83],color='orange', label='New York mobility percent change from baseline')
#axl.axhline(y=0, color='r', linestyle='--', label='baseline')
#plt.fill_between(dfmjkt.index,0, dfmjkt.retail_and_recreation_percent_change_from_baseline,color='b',alpha=.3,label='change from baseline')
plt.legend(loc=2)
plt.xlabel('Day', fontsize=20);
plt.ylabel('retail_and_recreation_percent_change_from_baseline', fontsize=20);
#plt.savefig('data/jkt_mobility.png')
#######################################################################################
is_jb =  (dfm['country_region']=='Indonesia') & (dfm['sub_region_1']=='West Java')
dfmjb= dfm[(is_jb)][['retail_and_recreation_percent_change_from_baseline']]
#######################################################################################
#fig, axl = plt.subplots(figsize=(15,10))
#axl.plot(dfmjb,color='green', label='West Java mobility percent change from baseline')
#axl.axhline(y=0, color='r', linestyle='--', label='baseline')
#plt.fill_between(dfmjb.index,0, dfmjb.retail_and_recreation_percent_change_from_baseline,color='g',alpha=.3,label='change from baseline')

#plt.legend(loc=2)
#plt.xlabel('Day', fontsize=20);
#plt.ylabel('retail_and_recreation_percent_change_from_baseline', fontsize=20);

#plt.savefig('data/jb_mobility.png')
#######################################################################################
is_jt =  (dfm['country_region']=='Indonesia') & (dfm['sub_region_1']=='East Java')
dfmjt= dfm[(is_jt)][['retail_and_recreation_percent_change_from_baseline']]
#######################################################################################
#fig, axl = plt.subplots(figsize=(15,10))
#axl.plot(dfmjt,color='red', label='East Java mobility percent change from baseline')
#axl.axhline(y=0, color='r', linestyle='--', label='baseline')
#plt.fill_between(dfmjt.index,0, dfmjt.retail_and_recreation_percent_change_from_baseline,color='r',alpha=.3,label='change from baseline')

#plt.legend(loc=2)
#plt.xlabel('Day', fontsize=20);
#plt.ylabel('retail_and_recreation_percent_change_from_baseline', fontsize=20);

#plt.savefig('data/jt_mobility.png')
#######################################################################################
is_jt =  (dfm['country_region']=='Indonesia') & (dfm['sub_region_1']=='East Java')
dfmjt= dfm[(is_jt)][['transit_stations_percent_change_from_baseline']]
#######################################################################################
#fig, axl = plt.subplots(figsize=(15,10))
#axl.plot(dfmjt,color='red', label='East Java mobility percent change from baseline')
#axl.axhline(y=0, color='r', linestyle='--', label='baseline')
#plt.fill_between(dfmjt.index,0, dfmjt.transit_stations_percent_change_from_baseline,color='r',alpha=.3,label='change from baseline')

#plt.legend(loc=2)
#plt.xlabel('Day', fontsize=20);
#plt.ylabel('transit_stations_percent_change_from_baseline', fontsize=20);
#plt.savefig('data/jt_transport_mobility.png')

######################################################################################
print (dfjkt25[80:-6])
r, p = stats.pearsonr(dki[:-5],dfmjkt_d[34:])
print(r,p)


df = pd.read_csv('data/synchrony_sample.csv')

def crosscorr(datax, datay, lag=0, wrap=False):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))
new_arr=[]
for arr in [dfmjkt_a[34:],dfmjkt_b[34:],dfmjkt_c[34:],dfmjkt_d[34:],dfmjkt_e[34:],dfmjkt_f[34:],dfjkthum[63:-6],dfjkt25[81:-6],dfjktwind[68:-6],dfjkttemp[63:-6]]:
    yes=arr
    yes['no']=np.arange(0,67,1)
    yes.set_index('no', inplace=True)
    new_arr.append(yes)
seconds = 1
fps = 5
no_splits = 1
samples_per_split = dki.shape[0]/no_splits
rss=[]
for t in range(0, no_splits):
    for dki2 in new_arr:
        d1 = dki[:-5][0].loc[(t)*samples_per_split:(t+1)*samples_per_split]
        d2 = dki2.iloc[:, 0].loc[(t)*samples_per_split:(t+1)*samples_per_split]
        print (d1.shape,d2.shape)
        rs = [crosscorr(d1,d2, lag) for lag in range(-int(seconds*fps),int(seconds*fps+1))]
        rss.append(rs)
rss = pd.DataFrame(rss)
f,ax = plt.subplots(figsize=(10,5))
sns.heatmap(rss,cmap='RdBu_r',ax=ax)

ax.set(title='Time Lagged Cross Correlation', xlabel='Offset',ylabel='Window epochs')
#ax.set_xticks([0, 50, 100, 151])
#ax.set_xticklabels([ -40, -20, 0, 20, 40])

