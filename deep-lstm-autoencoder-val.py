# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:19:54 2020

@author: yudis
"""

import numpy as np
import torch
import pandas as pd
#from numpy import array
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
#from sklearn.metrics import mean_squared_error
#from math import sqrt
import matplotlib.pyplot as plt

class MV_LSTM(torch.nn.Module):
    def __init__(self,n_features,seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 59 # number of hidden states
        self.n_hidden2 = 59
        self.n_hidden3 = 59
        self.n_hidden4 = 59
        self.n_layers = 1 # number of LSTM layers (stacked)
        #self.dropout = nn.Dropout(0.1) 
        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                             batch_first = True
                                 )
        
        self.conv1 = torch.nn.Conv1d(59,59,kernel_size=7,stride=3)
        
        self.l_lstm2 = torch.nn.LSTM(input_size =59, 
                                 hidden_size = self.n_hidden2,
                                 num_layers = self.n_layers, 
                                 batch_first = True
                                 )

        self.l_linear = torch.nn.Linear(self.n_hidden*(56), self.seq_len*self.n_features)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
     
        


    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers,batch_size,self.n_hidden).cuda()
        cell_state = torch.zeros(self.n_layers,batch_size,self.n_hidden).cuda()
        
        hidden_state2 = torch.zeros(self.n_layers,batch_size,self.n_hidden2).cuda()
        cell_state2 = torch.zeros(self.n_layers,batch_size,self.n_hidden2).cuda()
        
        hidden_state3 = torch.zeros(self.n_layers,batch_size,self.n_hidden3).cuda()
        cell_state3 = torch.zeros(self.n_layers,batch_size,self.n_hidden3).cuda()
        
        self.hidden = (hidden_state, cell_state)
        self.hidden2 = (hidden_state2, cell_state2)
        self.hidden3 = (hidden_state3, cell_state3)
        print (self.hidden)


    def forward(self, x):        
        batch_size, seq_len, _ = x.size()


        conv_out = self.conv1(x.permute(0,2,1))

       
        lstm_out, self.hidden = self.l_lstm(conv_out.permute(0,2,1),self.hidden)
        lstm_out, self.hidden2 = self.l_lstm2(lstm_out,self.hidden2)

        
        x = lstm_out.contiguous().view(batch_size,-1)

        return self.sigmoid(self.l_linear(x))
    
def merge(sequences):
    X, y = list(), list()
   
    for reg in sequences:
        en_pd=sequences.get(reg)
        print (en_pd)
        
        en_pd.drop(['country_region_code', 'country_region', 'sub_region_1', 'sub_region_2',
       'metro_area', 'iso_3166_2_code', 'census_fips_code', 'date', 'iso_code', 'continent',
       'location','YYYYMMDD'#,'UVIEF', 'UVIEFerr',
       #'UVDEF', 'UVDEFerr', 'UVDEC', 'UVDECerr', 'UVDVF', 'UVDVFerr', 'UVDVC',
       #'UVDVCerr', 'UVDDF', 'UVDDFerr', 'UVDDC', 'UVDDCerr', 'CMF', 'ozone'#,'retail_and_recreation_percent_change_from_baseline',
       #'grocery_and_pharmacy_percent_change_from_baseline',
       #'parks_percent_change_from_baseline',
       #'transit_stations_percent_change_from_baseline',
       #'workplaces_percent_change_from_baseline',
       #'residential_percent_change_from_baseline',#,'UVIEFerr',
       #'UVDEFerr', 'UVDEC', 'UVDECerr', 'UVDVFerr', 'UVDVC',
       #'UVDVCerr', 'UVDDFerr', 'UVDDC', 'UVDDCerr', 'CMF']
        ], axis=1,inplace=True)
        print (en_pd.columns)
        #en_pd.iloc[:,[7,8,10,11,13,14,16,17]] = en_pd.iloc[:,[7,8,10,11,13,14,16,17]].fillna(0)
        en_pd.iloc[:,[6,9,12,15,19,20]] = en_pd.iloc[:,[6,9,12,15,19,20]].fillna(method='ffill')
        en_pd = en_pd.fillna(0)
        print (en_pd.isnull().values.any())
        #en_pd['YYYYMMDD'] = pd.to_datetime(en_pd['YYYYMMDD'])
        en_pd['tests_units']=en_pd['tests_units'].astype('category')
        en_pd['tests_units'] = en_pd['tests_units'].cat.codes
        #mask = (en_pd['YYYYMMDD'] > '20200322') & (en_pd['YYYYMMDD'] <= '20200901')
        #print (reg, en_pd.values.shape)
        #en_pd = en_pd.loc[mask]
        #if (en_pd.shape[0]<163):
        #    continue
        #print (reg,en_pd.values[:,8])
        #quit()
        X.append(en_pd.values)
        print (en_pd.values.shape)
        
    return np.array(X)


dfs = pd.read_pickle("dfunity.pkl")
dfs_test = pd.read_pickle("dfunity_test.pkl")


new_dataset=merge(dfs)
new_dataset_test=merge(dfs_test)

alld=new_dataset
alld=alld.reshape(alld.shape[0]*alld.shape[1],alld.shape[2])
alld_test=new_dataset_test
alld_test=alld_test.reshape(alld_test.shape[0]*alld_test.shape[1],alld_test.shape[2])

scaler = MinMaxScaler()
scaler.fit(alld)
X=[scaler.transform(x) for x in new_dataset_test]



X=np.array(X)

n_features = 59

mv_net = MV_LSTM(n_features,174).cuda()



X_test=X[3:4,:,:]


mv_net.load_state_dict(torch.load("deep_lstm_gradcam_59_total.pt"))
mv_net.init_hidden(1)    
lstm_out = mv_net(torch.tensor(X_test,dtype=torch.float32).cuda())
lstm_out=lstm_out.cpu().data.numpy().reshape(1,174,59)



plt.plot(np.array(X_test[0,:,8]),label='Prediction')
plt.plot(np.array(lstm_out[0,:,8]),label='Actual')
#plt.xticks(rotation=90)

plt.xlabel("Days")
plt.ylabel("New Cases Smoothed (Normalized)")
#fig.autofmt_xdate()
plt.legend(loc=2)
plt.show()
