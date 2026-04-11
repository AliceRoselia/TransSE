# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 20:42:47 2026

@author: User
"""

import torch
import torch.nn as nn
import torch.nn.functional as func

torch.manual_seed(42)

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
        self.submodule = nn.Conv2d(n, n, (3,3))
    def forward(self,x):
        return self.submodule(x)
        # Main convnet block.
        #I am not sure
        
        
#https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/ in case you need 
#a flexible attention.
class attentionBlock(nn.Module):
    def __init__(self,n,attention_vector_size = 8,n_hidden_multiplier=4,nhead = 8):
        
        super(attentionBlock,self).__init__()
        self.proj_kernel = nn.Conv2d(n, n*attention_vector_size*3, (1,1))
        
        #We need
        
        self.n = n
        self.attention_vector_size = attention_vector_size
        
    def forward(self,x):
        x = self.proj_kernel(x)
        
        attention_inputs = x.mean((1,2)).view(-1,self.n*3,self.attention_vector_size)
        q,k,v = attention_inputs[:,:self.n,:], attention_inputs[:,self.n:self.n*2,:], attention_inputs[:,self.n*2:,:]
        
        #Working in progress. Raise the error for now.
        
        result = None
        
        return x*result[:,None,None,:]
        
    
class TransSEBlock(nn.Module):
    def __init__(self,n):
        super(TransSEBlock,self).__init__()
        self.mainConvBlock = convBlock(n)
        self.attentionBlock = attentionBlock(n)
    def forward(self,x):
        return self.attentionBlock(self.mainConvBlock(x))