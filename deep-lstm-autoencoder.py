import torch


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
import pickle


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
        
        self.conv1 = torch.nn.Conv1d(59,59,kernel_size=7, stride=3)
        
        self.l_lstm2 = torch.nn.LSTM(input_size =59, 
                                 hidden_size = self.n_hidden2,
                                 num_layers = self.n_layers, 
                                 batch_first = True
                                 )
        

        self.l_linear = torch.nn.Linear(self.n_hidden*(56), self.seq_len*self.n_features)
        self.sigmoid = nn.Sigmoid()
     
        


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
        en_pd.iloc[:,[6,9,12,15,19,20]] = en_pd.iloc[:,[6,9,12,15,19,20]].fillna(method='ffill')
        en_pd = en_pd.fillna(0)
        print (en_pd.isnull().values.any())
        en_pd['tests_units']=en_pd['tests_units'].astype('category')
        en_pd['tests_units'] = en_pd['tests_units'].cat.codes
        X=[]
        X.append(en_pd.values)
        print (en_pd.values.shape)
        
    return np.array(X)


dfs = pd.read_pickle("dfunity.pkl")
new_dataset=merge(dfs)
alld=new_dataset
alld=alld.reshape(alld.shape[0]*alld.shape[1],alld.shape[2])

scaler = MinMaxScaler()
scaler.fit(alld)
X=[scaler.transform(x) for x in new_dataset]


X=np.array(X)
n_features = 59
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
        

        loss = 1000*criterion(output.reshape(x_batch.size(0),174,59), x_batch)  

        loss.backward()
        optimizer.step()        
        optimizer.zero_grad() 
    print('step : ' , t , 'loss : ' , loss.item())


torch.save(mv_net.state_dict(),"deep_lstm_gradcam_59_total.pt") 


