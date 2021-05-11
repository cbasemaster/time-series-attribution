import torch

#import os
import numpy as np
import pandas as pd
#from tqdm import tqdm
#import seaborn as sns
#from pylab import rcParams
import matplotlib.pyplot as plt
#from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
#from pandas.plotting import register_matplotlib_converters
#from torch import nn, optim
#from torch.utils.data import DataLoader, Dataset
#import random
#from scipy.interpolate import UnivariateSpline  
import pickle
#from sklearn.decomposition import PCA
#import time
#import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt

#import pickle



class MV_LSTM(torch.nn.Module):
    def __init__(self,n_features,seq_length):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_length
        self.n_hidden = 43 # number of hidden states
        self.n_hidden2 = 43
        self.n_hidden3 = 43
        self.n_hidden4 = 43
        self.n_layers = 1 # number of LSTM layers (stacked)
        #self.dropout = nn.Dropout(0.1) 
        
        self.l_lstm = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                             batch_first = True
                                 )
        
        self.conv1 = torch.nn.Conv1d(43,43,kernel_size=7, stride=3)
        
        self.l_lstm2 = torch.nn.LSTM(input_size =43, 
                                 hidden_size = self.n_hidden2,
                                 num_layers = self.n_layers, 
                                 batch_first = True
                                 )
        

        self.l_linear = torch.nn.Linear(self.n_hidden*(56), self.seq_len*self.n_features)
        #self.sigmoid = nn.Sigmoid()
        #self.relu = nn.ReLU()
     
        


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



    def forward(self, x):        
        batch_size, seq_len, _ = x.size()

        conv_out = self.conv1(x.permute(0,2,1))
       
        lstm_out, self.hidden = self.l_lstm(conv_out.permute(0,2,1),self.hidden)
        lstm_out, self.hidden2 = self.l_lstm2(lstm_out,self.hidden2)
        
        x = lstm_out.contiguous().view(batch_size,-1)
        return self.sigmoid(self.l_linear(x))
        

    

# split a multivariate sequence into samples
def merge(sequences):   
    for reg in sequences:
        en_pd=sequences.get(reg)
        print (en_pd)
        
        en_pd.drop(['country_region_code', 'country_region', 'sub_region_1', 'sub_region_2',
       'metro_area', 'iso_3166_2_code', 'census_fips_code', 'date', 'iso_code', 'continent',
       'location','YYYYMMDD','UVIEF', 'UVIEFerr',
       'UVDEF', 'UVDEFerr', 'UVDEC', 'UVDECerr', 'UVDVF', 'UVDVFerr', 'UVDVC',
       'UVDVCerr', 'UVDDF', 'UVDDFerr', 'UVDDC', 'UVDDCerr', 'CMF', 'ozone'#,'retail_and_recreation_percent_change_from_baseline',
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
"""
df = pd.read_csv('data/owid-covid-data.csv', skiprows=1)
df.head()
df.info()

#s = 'iso_code,continent,location,date,total_cases,new_cases,new_cases_smoothed,total_deaths,new_deaths,new_deaths_smoothed,total_cases_per_million,new_cases_per_million,new_cases_smoothed_per_million,total_deaths_per_million,new_deaths_per_million,new_deaths_smoothed_per_million,new_tests,total_tests,total_tests_per_thousand,new_tests_per_thousand,new_tests_smoothed,new_tests_smoothed_per_thousand,tests_per_case,positive_rate,tests_units,stringency_index,population,population_density,median_age,aged_65_older,aged_70_older,gdp_per_capita,extreme_poverty,cardiovasc_death_rate,diabetes_prevalence,female_smokers,male_smokers,handwashing_facilities,hospital_beds_per_thousand,life_expectancy,human_development_index'
#s=s.split(',')
#print (s)
df.columns = ['iso_code', 'continent', 'location', 'date', 'total_cases', 'new_cases', 'new_cases_smoothed', 'total_deaths', 'new_deaths', 'new_deaths_smoothed', 'total_cases_per_million', 'new_cases_per_million', 'new_cases_smoothed_per_million', 'total_deaths_per_million', 'new_deaths_per_million', 'new_deaths_smoothed_per_million', 'new_tests', 'total_tests', 'total_tests_per_thousand', 'new_tests_per_thousand', 'new_tests_smoothed', 'new_tests_smoothed_per_thousand', 'tests_per_case', 'positive_rate', 'tests_units', 'stringency_index', 'population', 'population_density', 'median_age', 'aged_65_older', 'aged_70_older', 'gdp_per_capita', 'extreme_poverty', 'cardiovasc_death_rate', 'diabetes_prevalence', 'female_smokers', 'male_smokers', 'handwashing_facilities', 'hospital_beds_per_thousand', 'life_expectancy', 'human_development_index']
df.set_index('iso_code', inplace=True)

dfs = dict(tuple(df.groupby('iso_code')))
"""

dfs = pd.read_pickle("dfunity.pkl")


new_dataset=merge(dfs)
alld=new_dataset
alld=alld.reshape(alld.shape[0]*alld.shape[1],alld.shape[2])

scaler = MinMaxScaler()
scaler.fit(alld)
X=[scaler.transform(x) for x in new_dataset]


X=np.array(X)

n_features = 43
print (X.shape)

mv_net = MV_LSTM(n_features,174).cuda()
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(mv_net.parameters(), lr=1e-3)

train_episodes = 3000

batch_size = 16


mv_net.train()

for t in range(train_episodes):
    
    #np.random.shuffle(X)
    
    for b in range(0,len(X),batch_size):

        p = np.random.permutation(len(X))
        
        inpt = X[p][b:b+batch_size,:,:]

        
        x_batch = torch.tensor(inpt,dtype=torch.float32).cuda()    

        mv_net.init_hidden(x_batch.size(0))

        output = mv_net(x_batch) 
        

        loss = 1000*criterion(output.reshape(x_batch.size(0),174,43), x_batch)  

        loss.backward()
        optimizer.step()        
        optimizer.zero_grad() 
    print('step : ' , t , 'loss : ' , loss.item())


torch.save(mv_net.state_dict(),"deep_lstm_gradcam_43_total.pt") 

quit()    
#data2x=data2[~(data2.confirmed==0)]
data2x=data2
truth = data2





data2x.values[0:len(data2x),0]=np.insert(np.diff(data2x.values[0:len(data2x),0]),0,0)

data2x=scaler.transform(data2x) 


X_test = np.expand_dims(data2x, axis=0)

mv_net.init_hidden(1)

lstm_out = mv_net(torch.tensor(X_test[:,-76:,:],dtype=torch.float32).cuda())
lstm_out=lstm_out.reshape(1,109,1).cpu().data.numpy()


print (data2x[-76:,0],lstm_out)
actual_predictions = scaler.inverse_transform(np.tile(lstm_out, (1, 1,5))[0])[:,0]



x = np.arange(0, 54, 1)
x2 = np.arange(0, 67, 1)
x3 = np.arange(0, 100, 10)
x4 = np.arange(0, 50, 1)



with open('./lstmdata/predict_indo3.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(pd.Series(actual_predictions), f,protocol=2)


fig, ax = plt.subplots() 
plt.title('Days vs Confirmed Cases Accumulation')
plt.ylabel('Confirmed')

left, width = .25, .5
bottom, height = .25, .5
right = left + width
top = bottom + height

print (date.index)
date_list=pd.date_range(start=date.index[0],end=date.index[-1])
print (date_list)

plt.axvline(x=np.array(date_list)[66], color='r', linestyle='--')

ax.text(0.2*(left+right), 0.8*(bottom+top), 'input sequence',
        horizontalalignment='left',
        verticalalignment='center',
        fontsize=10, color='red',
        transform=ax.transAxes)
ax.text(0.0125*(left+right), 0.77*(bottom+top), '______________________',
        horizontalalignment='left',
        verticalalignment='center',
        fontsize=20, color='red',
        transform=ax.transAxes)



sumpred=np.cumsum(np.absolute(actual_predictions))

#print (date.values.shape) 
#print (sqrt(mean_squared_error(date.confirmed,sumpred)))          
#plt.plot(date.values[-67:],np.cumsum(data2.confirmed.values[-67:]))
plt.plot(np.array(date_list),sumpred,label='Prediction')
#plt.plot(np.array(date_list),date.confirmed,label='Actual')
plt.xticks(rotation=90)
fig.autofmt_xdate()
plt.legend(loc=2)
plt.show() 

