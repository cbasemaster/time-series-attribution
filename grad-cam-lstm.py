# -*- coding: utf-8 -*-
"""
Created on Fri May  1 16:59:12 2020

@author: yudis
"""
import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
from numpy import array
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim

def split_sequences(sequences, n_steps):
    X, y = list(), list()
    
    #data2_scale=data2_scale[~(data2_scale[:,0:3]==0).all(1)]
    #df_china=scaler.fit_transform(df[(is_china)][['confirmed','recovered','deaths']])
    
    #scaler = MinMaxScaler()
    for i in range(0,len(sequences),109):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if i!=0 and end_ix > len(sequences):
            break
        
        sequences[i:end_ix,0]=np.insert(np.diff(sequences[i:end_ix,0]),0,0)
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix-33], sequences[end_ix-33:end_ix]
        
        #scaler.fit(seq_x)
        
        #print (i,len(sequences),seq_x.shape,seq_y.shape)
        
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def show_cam_on_image(img, mask,i,j):
   
    print (mask.shape)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    print (heatmap.shape)
    #print (heatmap.shape,np.rollaxis(heatmap,0,2).shape)
    cam = heatmap #*(np.float32(img)).transpose(1, 2, 0)
    cam = cam / np.max(cam)
    
    #print ( cam[0],cam[0].shape, type(cam))
    #print ( img.transpose(1, 2, 0)[0],img.transpose(1, 2, 0)[0].shape, type(img))
    #print (img.transpose(1, 2, 0).transpose(1,0,2)[0],img.transpose(1, 2, 0).transpose(1,0,2).shape, cam.transpose(1,0,2).shape)
    
    np.save('cam_array', cam.transpose(1,0,2))
    
    cv2.imwrite("./brain_unet/gradcam/att_dot/lstm_unity_norway_gdp%s_%s.jpg" %(i, j), np.uint8(255*cam))
    

def preprocess_image(img):
    #means = [0.595, 0.456, 0.406]
    #stds = [0.229, 0.224, 0.225]
    img=np.transpose(img, (1, 2, 0))
    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    print (preprocessed_img.shape)
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input
    
class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        
        #for name, module in self.model._modules.items():
        #    x = module(x)
        #    print (name,self.target_layers)
            #if name in self.target_layers:
        print ('xshape',x.shape)
        
        x.register_hook(self.save_gradient)
        outputs += [x]
        return outputs, x

    
class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        out,inp = self.model(x)
        h_x = F.softmax(out, dim=1).data.squeeze() 
        #probs, idx = h_x.sort(0, True)
        print (out.shape,inp.shape)
        target_activations, x = self.feature_extractor(inp)    
        print (target_activations[0].shape)
        
        return target_activations, out

class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.train()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        
        
        
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)
        
        output=1-output.resize(1,174,59)
        
        print (features[-1].shape, output.shape)
        
        #if index == None:
        #    index = np.argmax(output[0].cpu().data.numpy())
            
        #print (index)
        #quit()

        one_hot = np.zeros((1, output.size()[-2], output.size()[-1]), dtype=np.float32)
        
        one_hot[0][:][8] = 1
        output_diff = output[0][1:] - output[0][:-1]
        output_diff = nn.functional.pad(output_diff, (0,0,1,0))
        input_diff = input[0][1:] - input[0][:-1]
        input_diff = nn.functional.pad(input_diff, (0,0,1,0))
        m=nn.ReLU()
        output_diff = m(output_diff)
        input_diff = m(input_diff)
       
        
        #one_hot=one_hot.reshape(1,174,59)
        #mse=torch.nn.MSELoss()
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        
        
        

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        
        #print (self.extractor.get_gradients()[-1].shape)
        #quit()
        weights = self.extractor.get_gradients()[-1].cpu().data.numpy()
        
        
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]
        print (weights.shape)
        weights = np.mean(weights, axis=(0))
       
        #cam = np.zeros(target.shape[1:], dtype=np.float32)
        #print (weights)
        #for i, w in enumerate(weights):
        #    cam += w * target[:,i]
        
        
        cam = weights*(target)#+(input).cpu().data.numpy()[0, :])

        #
        cam = np.maximum(cam, 0)
        
        increase=input_diff.cpu().data.numpy()
        increase[increase>0]=1
        
        print (input.shape[1:])
        cam = cv2.resize(cam, torch.Size([59, 174]))#*increase
        print (cam.shape)
        
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

#from model2 import UNet


"""
#==============================================================================
# Transform
#==============================================================================
normalize = transforms.Normalize(mean=[0.595, 0.456, 0.406], std=[0.229, 0.224, 0.225])


transformer = transforms.Compose([
    transforms.Resize((256,256)),
    #transforms.RandomAffine((-180,180)),
    #transforms.RandomResizedCrop(224),
#    transforms.Grayscale(),
#    transforms.Resize((h,w)),
    transforms.ToTensor(),
    #normalize,
    ])

transformer2 = transforms.Compose([
    transforms.Resize((256,256)),
    #transforms.RandomAffine(degrees=(-180,180)),
    #transforms.RandomResizedCrop(256),
    #transforms.Grayscale(),
#    transforms.Resize((h,w)),
    transforms.ToTensor(),
    normalize,
    ])

# Tentukan 1 dataset
dataset = datasets.ImageFolder(root='.\COVID-19-master\COVID-19-master\X-Ray Image DataSet', transform=transformer)
"""
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
        self.dropout = nn.Dropout(0.1) 
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
        
        #self.conv2 = torch.nn.Conv1d(59,59,kernel_size=3)
        """
        self.l_lstm3 = torch.nn.LSTM(input_size = n_features, 
                                 hidden_size = self.n_hidden3,
                                 num_layers = self.n_layers, 
                                 batch_first = True
                                 )
        """
        #self.conv3 = torch.nn.Conv1d(59,59,kernel_size=5)
        #self.l_lstm4 = torch.nn.LSTM(input_size = n_features, 
        #                         hidden_size = self.n_hidden4,
        #                         num_layers = self.n_layers, 
        #                         batch_first = True
        #                         )
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
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

        #lstm_out, self.hidden = self.l_lstm(x,self.hidden)
        #lstm_out, self.hidden = self.l_lstm(x)
        conv_out = self.conv1(x.permute(0,2,1))
        #print (conv_out.shape)
        #lstm_out, self.hidden2 = self.l_lstm2(conv_out.permute(0,2,1),self.hidden2)
        
        #conv_out = self.conv2(conv_out)
        #conv_out = self.conv3(conv_out)
        #print (conv_out2.shape)
       
        lstm_out, self.hidden = self.l_lstm(conv_out.permute(0,2,1),self.hidden)
        lstm_out, self.hidden2 = self.l_lstm2(lstm_out,self.hidden2)
        # lstm_out(with batch_first = True) is 
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest       
        # .contiguous() -> solves tensor compatibility error
        
        x = lstm_out.contiguous().view(batch_size,-1)
        return self.sigmoid(self.l_linear(x)),lstm_out

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
       #'UVDVCerr', 'UVDDFerr', 'UVDDC', 'UVDDCerr', 'CMF'
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
#X=[x for x in new_dataset_test]


X=np.array(X)

n_features = 59

print (X.shape)

X_test=X[0:1,:,:]




model = MV_LSTM(n_features,174)
model.load_state_dict(torch.load("deep_lstm_gradcam_total.pt"))

model.init_hidden(1)

print (model)
use_gpu = torch.cuda.is_available()  # Running GPU
if use_gpu:
    model = model.cuda()
    
model.train()

grad_cam = GradCam(model=model, feature_module=model.l_lstm , \
                       target_layer_names=["0"], use_cuda=True)



target_index = None
mask = grad_cam(torch.tensor(X_test,dtype=torch.float32), target_index)

no=0  

show_cam_on_image(X_test, mask,no,None)
quit()
    