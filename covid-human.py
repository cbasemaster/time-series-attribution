# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 08:54:34 2020

@author: yudis
"""

# importing pandas package 
import pandas as pd 

# making data frame from csv file 
data = pd.read_csv("./data/time-series-19-covid-combined-6.csv") 

# replacing blank spaces with '_' 
data.columns =[column.replace("/", "_") for column in data.columns] 

# filtering with query method 
indon=data.query('Country_Region == "Indonesia"', inplace = False) 


# display 
 
