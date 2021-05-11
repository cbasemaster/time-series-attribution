# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 00:09:12 2020

@author: yudis
"""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from matplotlib.pyplot import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

nba=np.load('cam_array.npy')


print (nba.shape)

nba=np.sum(nba,axis=2)/3.0

nba2=np.sum(nba,axis=1)



# remove index title
nba=pd.DataFrame(nba)
# normalize data columns
print (nba)

# relabel columns

labels = ['retail, recreation mobility',
       'grocery, pharmacy mobility',
       'parks mobility',
       'transit stations mobility',
       'workplaces mobility',
       'residential mobility', 'total cases', 'new cases',
       'new cases smoothed', 'total deaths', 'new deaths',
       'new deaths smoothed', 'total cases per million',
       'new cases per million', 'new cases smoothed per million',
       'total deaths per million', 'new deaths per million',
       'new deaths smoothed per million', 'new tests', 'total tests',
       'total tests per thousand', 'new tests per thousand',
       'new tests smoothed', 'new tests smoothed per thousand',
       'tests per case', 'positive rate', 'tests units', 'stringency index',
       'population', 'population density', 'median age', 'aged 65 older',
       'aged 70 older', 'gdp per capita', 'extreme poverty',
       'cardiovasc death rate', 'diabetes prevalence', 'female smokers',
       'male smokers', 'handwashing facilities', 'hospital beds per thousand',
       'life expectancy', 'human development index', 'UVIEF', 'UVIEFerr',
       'UVDEF', 'UVDEFerr', 'UVDEC', 'UVDECerr', 'UVDVF', 'UVDVFerr', 'UVDVC',
       'UVDVCerr', 'UVDDF', 'UVDDFerr', 'UVDDC', 'UVDDCerr', 'CMF', 'ozone']

#labels.reverse()

nba.columns = pd.date_range(start="2020-03-22",end="2020-09-11").date

# set appropriate font and dpi
sns.set(font_scale=4.0)
sns.set_style({"savefig.dpi": 100})
# plot it out
ax = sns.heatmap(nba,  vmin=0.0, vmax=1.0, yticklabels=labels,cmap='jet', linewidths=.1)
# set the x-axis labels on the top
ax.xaxis.tick_top()
# rotate the x-axis labels
plt.xticks(rotation=30)
plt.xticks(fontsize=70)
plt.yticks(fontsize=70)
# get figure (usually obtained via "fig,ax=plt.subplots()" with matplotlib)
fig = ax.get_figure()
# specify dimensions and save
fig.set_size_inches(125, 75)

fig.savefig("norway_att.tiff", format='tiff')

plt.figure(figsize=(16,9))

# Create horizontal bar plot
ax2=sns.barplot(x=np.array(labels)[np.argsort(-nba2)[0:10]], y=-np.sort(-nba2)[0:10], palette='Greens_r')

fig2 = ax2.get_figure()
#plt.bar(np.array(labels)[np.argsort(-nba2)[0:10]],-np.sort(-nba2)[0:10]) 
plt.xticks(rotation=30)
plt.xticks(fontsize=70)
plt.yticks(fontsize=70)

# specify dimensions and save
fig2.set_size_inches(100, 50)

fig2.savefig("histo_norway.tiff", format='tiff')