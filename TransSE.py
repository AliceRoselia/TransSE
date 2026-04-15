# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 20:42:47 2026

@author: User
"""

import torch
import torch.nn as nn
import torch.nn.functional as func

from medmnist import RetinaMNIST
import torchvision.transforms as transforms
import torch.utils.data as data

torch.manual_seed(42)

torch.set_float32_matmul_precision("high")


#TransSEnet for medical imaging and similar tasks.



class residualBlock(nn.Module):
    
    def __init__(self,submodule):
        super(residualBlock,self).__init__()
        self.submodule = submodule
    
    def forward(self,x):
        #Simple residual block to be fixed later if we want a more complex one.
        return x + self.submodule(x)
    

class convBlock(nn.Module):
    def __init__(self,n):
        super(convBlock,self).__init__()
        self.submodule = nn.Conv2d(n, n, (3,3),padding="same")
        self.batchnorm = nn.BatchNorm2d(n)
    def forward(self,x):
        return self.batchnorm(func.relu(self.submodule(x)))
        # Main convnet block.
        #I am not sure whether to put batchnorm before or after relu. The original literature ix mixed.
        #It's said that batchnorm after relu performs a bit better, which makes sense given that is the "bias" is literally the point before doing relu.
        #Regardless, there is little difference because batchnorm cares about covariate shift which only accumulates after several layers.
        
        
#https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/ in case you need 
#a flexible attention.
class attentionBlock(nn.Module):
    def __init__(self,n,attention_vector_size = 8,n_hidden_multiplier=8,nhead = 8,nlayer=8):
        
        super(attentionBlock,self).__init__()
        
        assert n % attention_vector_size == 0
        
        #We need
        
        self.n = n//attention_vector_size
        self.attention_vector_size = attention_vector_size
        self.n_total = n
        #Also try using norm_first = True. Let's try later. However, given that batchnorm was already used before during the conv layer, I'm not sure.
        attention = nn.TransformerEncoderLayer(d_model=self.attention_vector_size, nhead=nhead,dim_feedforward = self.attention_vector_size*n_hidden_multiplier, batch_first=True)
        self.attention = nn.TransformerEncoder(attention,num_layers = nlayer)
        
    def forward(self,x):
        
        attention_inputs = x.mean((2,3)).view(-1,self.n,self.attention_vector_size)
        
        attention_outputs = self.attention(attention_inputs)
        
        result = func.sigmoid(attention_outputs.view(-1,self.n_total))
        
        return x*result[:,:,None,None]
        
    
class TransSEBlock(nn.Module):
    def __init__(self,n):
        super(TransSEBlock,self).__init__()
        self.mainConvBlock = convBlock(n)
        self.attentionBlock = attentionBlock(n)
    def forward(self,x):
        return self.attentionBlock(self.mainConvBlock(x))
    
channel_count = 256
batch_size = 32



train_data = RetinaMNIST(split="train",transform = transforms.ToTensor(),download=True,size = 224)
train_data_loader = data.DataLoader(dataset = train_data, batch_size = batch_size,shuffle = True)

test_data = RetinaMNIST(split="test",transform = transforms.ToTensor(),download=True,size = 224)
test_data_loader = data.DataLoader(dataset = test_data, batch_size = batch_size,shuffle = False)

class TransSENet(nn.Module):
    def __init__(self, in_channels,classes):
        super(TransSENet,self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels,64,kernel_size=7),
            nn.ReLU(),
            residualBlock(TransSEBlock(64)),
            residualBlock(TransSEBlock(64)),
            residualBlock(TransSEBlock(64)),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64,128,kernel_size=3),
            nn.ReLU(),
            residualBlock(TransSEBlock(128)),
            residualBlock(TransSEBlock(128)),
            residualBlock(TransSEBlock(128)),
            residualBlock(TransSEBlock(128)),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128,256,kernel_size=3),
            nn.ReLU(),
            residualBlock(TransSEBlock(256)),
            residualBlock(TransSEBlock(256)),
            residualBlock(TransSEBlock(256)),
            residualBlock(TransSEBlock(256)),
            residualBlock(TransSEBlock(256)),
            residualBlock(TransSEBlock(256)),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256,512,kernel_size=3),
            nn.ReLU(),
            residualBlock(TransSEBlock(512)),
            residualBlock(TransSEBlock(512)),
            residualBlock(TransSEBlock(512)),
            )
        self.results = nn.Linear(512, classes)
    def forward(self,x):
        features = self.layers(x)
        return self.results(features.mean((2,3)))

net = TransSENet(3, 5).to("cuda")
optimizer = torch.optim.Adam(net.parameters(),lr = 1.5e-4)
loss = nn.CrossEntropyLoss()


current = 0
for data_input, result in train_data_loader:
    print(current)
    current += batch_size
    result = result.to("cuda")
    prediction = net(data_input.to("cuda"))
    result_loss = loss(prediction,result.view(-1))
    result_loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
    
    
    
    print(result_loss)
    
correct = 0
total = 400
    
with torch.no_grad():
    
    for data_input, result in test_data_loader:
        result = result.to("cuda")
        prediction = net(data_input.to("cuda"))
        correct += (prediction.argmax(dim=1) == result.view(-1)).sum()

print("accuracy: ",correct / total)


#1st epoch: 0.4350 (3,4,6,3 layers)

#1st epoch: 0.4500 (3,4,23,8 layers)

#1st epoch: 0.4775 (hidden multiplier doubled and reduced the attention vector to 8)

#1st epoch: 0.5200 (increased the transformer network to 8 layers).

#1st epoch: 0.5425 (Learning rate = 1.5e-4).



#Grid search hyperparameters:
    #Batch norms:
        
        #Batch norm before or after relu.
        #Batch norm before or after attention.
        
#This architecture is chosen first because it's easier to implement.
#The one I'd like to also try would be a "squeeze-and-attend" mechanism 
#where the query and key are from the squeeze but the attention values are 
#from image themselves (Don't throw away spatial information).
#This means squeezing the query and key to form "channel group query"
# and "channel group key" while the value remains unsqueezed into channels.
#It represents how channels attend to other channels.