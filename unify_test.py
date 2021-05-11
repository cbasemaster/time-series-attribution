# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 23:16:56 2020

@author: yudis
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 19:03:14 2020

@author: yudis
"""

import pickle
import pandas as pd


uv  = pd.read_pickle('dfuv.pkl')
covid  = pd.read_pickle('dfcovid.pkl')
mobile  = pd.read_pickle('dfmobile.pkl')

def containsAll(set, str):
    """ Check whether sequence str contains ALL of the items in set. """
    return 0 not in [c in str for c in set]


def merging(set1, set2):
    mask = (set1['date'] >= '2020-03-22') & (set1['date'] <= '2020-09-11')
    mask2 = (set2['YYYYMMDD'] >= '20200322') & (set2['YYYYMMDD'] <= '20200911')
    
    #print (reg, en_pd.values.shape)
    set1 = set1.loc[mask]
    set2 = set2.loc[mask2]
    print (set1.shape, set2.shape)
    new = [set1[0:174].reset_index(drop=True),set2.reset_index(drop=True)]
    return pd.concat(new,axis=1)

count=0
myKeys=[]
myValues=[]

for key, value in covid.items(): 
      found=0
      #print (key)
      for key2, value2 in uv.items():
            if (key=="Indonesia" and key2=="Thailand") or key==key2=="Sweden" or  key==key2=="Norway" or key==key2=='Italy':
              #print (key,key2)
              count+=1
              found=1
              
              new_pd=merging(value, value2)
              myKeys.append(key)
              myValues.append(new_pd)
              
merged_dict=dict(zip(myKeys, myValues))

myKeys=[]
myValues=[]
count=0
for key, value in mobile.items(): 
      found=0
      #print (key)
      for key2, value2 in merged_dict.items():
            if key==key2:
              print (key,key2)
              count+=1
              found=1
              new_pd=merging(value, value2)
              myKeys.append(key)
              myValues.append(new_pd)
merged_dict=dict(zip(myKeys, myValues))
print (count)
#f = open("dfunity_test.pkl","wb")
#pickle.dump(merged_dict,f)
#f.close()