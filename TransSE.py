# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 20:42:47 2026

@author: User
"""

import torch
import torch.nn as nn
import torch.nn.functional as func

torch.manual_seed(42)

torch.set_float32_matmul_precision("high")

from time import time

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
    def forward(self,x):
        #Grok suggested adding batchnorm here. Maybe try later.
        return func.relu(self.submodule(x))
        # Main convnet block.
        #I am not sure
        
        
#https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/ in case you need 
#a flexible attention.
class attentionBlock(nn.Module):
    def __init__(self,n,attention_vector_size = 16,n_hidden_multiplier=4,nhead = 8):
        
        super(attentionBlock,self).__init__()
        
        assert n % attention_vector_size == 0
        
        #We need
        
        self.n = n//attention_vector_size
        self.attention_vector_size = attention_vector_size
        self.n_total = n
        #Grok suggested using norm_first = True. Let's try later. However, given that batchnorm was already used before during the conv layer, I'm not sure.
        self.attention = nn.TransformerEncoderLayer(d_model=self.attention_vector_size, nhead=nhead,dim_feedforward = self.attention_vector_size*n_hidden_multiplier, batch_first=True)
        
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

Test_block = residualBlock(TransSEBlock(channel_count)).to("cuda")

Test_data = torch.randn(32,channel_count,224,224).to("cuda")

print("Try running the model.")

start = time()

result = Test_block(Test_data)

print(time()-start)