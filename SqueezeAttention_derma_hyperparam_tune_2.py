# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 20:42:47 2026

@author: User
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func

from medmnist import DermaMNIST
import torchvision.transforms as transforms
import torch.utils.data as data
from muon import SingleDeviceMuonWithAuxAdam
#We still can't use Pytorch native implementation because it lacks suppport for 4d conv params, so we will use Keller's version.
torch._dynamo.config.recompile_limit = 128
torch._dynamo.config.cache_size_limit = 128 
torch._dynamo.config.accumulated_cache_size_limit = 128

#from torch.nn.attention import SDPBackend, sdpa_kernel

torch.manual_seed(71909099)

torch.set_float32_matmul_precision("high")


#TransSEnet for medical imaging and similar tasks.
import math

def schedule_LR(optimizer, epoch, max_epoch, muon_max, muon_min, adam_max, adam_min):
    #Cosine annealing. Self-implemented because the Pytorch version can't accept when there are 2 versions.
    for group in optimizer.param_groups:
        if group["use_muon"]:
            group["lr"] = muon_min + (1+math.cos(math.pi*epoch/max_epoch))*(muon_max-muon_min)/2
        else:
            group["lr"] = adam_min + (1+math.cos(math.pi*epoch/max_epoch))*(adam_max-adam_min)/2
    

        
#https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/ in case you need 
#a flexible attention.

    
batch_size = 2
num_workers = 4
prefetch_factor = 4

train_data = DermaMNIST(split="train",transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    # Optional: transforms.RandomResizedCrop(224, scale=(0.9,1.0))
]),download=True,size = 224)
train_data_loader = data.DataLoader(dataset = train_data, batch_size = batch_size,shuffle = True,
pin_memory=True,num_workers=num_workers,prefetch_factor=prefetch_factor,persistent_workers=True)

val_data = DermaMNIST(split="val",transform = transforms.ToTensor(),download=True,size = 224)
val_data_loader = data.DataLoader(dataset = val_data, batch_size = batch_size,shuffle = True,
pin_memory=True,num_workers=num_workers,prefetch_factor=prefetch_factor,persistent_workers=True)



#Use torch.nn.functional.scaled_dot_product_attention

class SqueezeAttentionBlock(nn.Module):
    
    def __init__(self,m,n, head = 4):
        super(SqueezeAttentionBlock,self).__init__()
        assert n%head == 0
        self.channel_group_count = m
        self.qk = nn.Linear(n,2*n)
        self.heads = head
        self.value_conv = nn.Conv2d(n, n, 1)
        self.bn1 = nn.BatchNorm2d(m*n)
        self.depthwise_conv = nn.Conv2d(n, n, 5, padding = "same",groups=n)
        self.pointwise_conv = nn.Conv2d(n, n, 1)
        #self.pw_conv = nn.Conv2d(n,n,1,padding = "same")
        self.bn2 = nn.BatchNorm2d(m*n)
        self.head_size = n//head
        scale = torch.full((head,m,m),self.head_size ** -0.5)
        self.register_buffer("scale", scale) #Pre-broadcast
    
    #@torch.compile()
    def fused_conv_activation(self,x):
        return x + func.silu(self.pointwise_conv(func.silu(self.depthwise_conv(x))))
    
    #def convs(self,x):
        #return self.pw_conv(self.dw_conv(x))
    
    
    #@torch.compile()
    def forward(self,x):
        # X is of shape [B,M,N,H,W]
        B,M,N,H,W = x.shape
        channel_reps = x.mean((3,4)) #dimension: B,M,N
        
        
        query, key = self.qk(channel_reps).view(B,M,self.heads,N*2//self.heads).transpose(1,2).chunk(2,dim=3) #Dimensions B,Head,M,N/Head 
        value = self.value_conv(x.view(B*M,N,H,W)).view(B,M,self.heads,self.head_size,H,W).transpose(1,2) #Dimensions: B,Head,M,N/Head,H,W
        
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale #Dimensions B, Head, M, M
        
        attn = func.softmax(scores,dim=-1)
        
        attention_result = torch.einsum('baij,bajchw -> baichw', attn, value).transpose(1,2)
        
        #TODO: add local gate.
        #print(self.scale.device)
        
        
        #attention_result = self.attention_kernel(query,key,value,self.scale)
    
        
        
        #Manual attention result because optimized kernels weren't optimized for this.
        
       #with sdpa_kernel(backends=[SDPBackend.MATH]):
            #attention_result = func.scaled_dot_product_attention(query,key,value).view(B,M,N,H,W)
        attention_result = attention_result.reshape(B,M,N,H,W) + x
        attention_result = self.bn1(attention_result.view(B,M*N,H,W)).view(B*M,N,H,W) #Dimensions: B*M,N,H,W
        
        attention_result = self.bn2(self.fused_conv_activation(attention_result).view(B,M*N,H,W))
        
        
        return attention_result.view(B,M,N,H,W)

class UpProjection(nn.Module):
    def __init__(self,n,n2):
        super(UpProjection,self).__init__()
        self.conv = nn.Conv2d(n, n2, 1, padding = "same")
        assert n2 == 2*n
    #@torch.compile()
    def forward(self,x):
        B,M,N,H,W = x.shape
        return self.conv(x.view(B*M,N,H,W)).view(B,M,N*2,H,W)

        
        
class SqueezeAttention(nn.Module):
    #@torch.compile()
    def squeeze_to_pool(self,x):
        B,M,N,H,W = x.shape
        return func.max_pool2d(x.view(B*M,N,H,W), 2).view(B,M,N,H//2,W//2)
    #USE max pooling. Strided convolution was tried and did NOT help.

    
    def __init__(self,in_channels,classes):
        super(SqueezeAttention,self).__init__()
        self.intro = nn.Conv2d(in_channels,256,kernel_size=7,padding="same")
        self.SAB1 = SqueezeAttentionBlock(8, 32)
        self.SAB2 = SqueezeAttentionBlock(8, 32)
        self.SAB3 = SqueezeAttentionBlock(8, 32)
        
        self.SAB4 = SqueezeAttentionBlock(8, 64)
        self.SAB5 = SqueezeAttentionBlock(8, 64)
        self.SAB6 = SqueezeAttentionBlock(8, 64)
        
        self.SAB7 = SqueezeAttentionBlock(8, 128)
        self.SAB8 = SqueezeAttentionBlock(8, 128)
        self.SAB9 = SqueezeAttentionBlock(8, 128)
        
        self.SAB10 = SqueezeAttentionBlock(8, 256)
        self.SAB11 = SqueezeAttentionBlock(8, 256)
        self.SAB12 = SqueezeAttentionBlock(8, 256)
        
        self.SAB13 = SqueezeAttentionBlock(8, 512)
        self.SAB14 = SqueezeAttentionBlock(8, 512)
        self.SAB15 = SqueezeAttentionBlock(8, 512)
        self.SAB16 = SqueezeAttentionBlock(8, 512)
        #self.SAB17 = SqueezeAttentionBlock(8, 512)
        #self.SAB18 = SqueezeAttentionBlock(8, 512)
        
        
        self.UP1 = UpProjection(32, 64)
        self.UP2 = UpProjection(64, 128)
        self.UP3 = UpProjection(128, 256)
        self.UP4 = UpProjection(256, 512)
        
        self.dropout = nn.Dropout(0.25)
        
        self.results = nn.Linear(4096, classes)
    @torch.compile()
    def forward(self,x):
        B,C,H,W = x.shape
        x = self.intro(x).view(B,8,32,H,W)
        x = self.SAB1(x)
        x = self.SAB2(x)
        x = self.SAB3(x)
        x = self.squeeze_to_pool(x) #112
        x = self.UP1(x)
        x = self.SAB4(x)
        x = self.SAB5(x)
        x = self.SAB6(x)
        x = self.squeeze_to_pool(x) #56
        x = self.UP2(x)
        x = self.SAB7(x)
        x = self.SAB8(x)
        x = self.SAB9(x)
        x = self.squeeze_to_pool(x) #28
        x = self.UP3(x)
        x = self.SAB10(x)
        x = self.SAB11(x)
        x = self.SAB12(x)
        x = self.squeeze_to_pool(x) #14
        x = self.UP4(x)
        x = self.SAB13(x)
        x = self.SAB14(x)
        x = self.SAB15(x)
        x = self.SAB16(x)
        #x = self.SAB17(x)
        #x = self.SAB18(x)
        
        
        
        x = self.dropout(x.mean((3,4)).view(-1,4096))
        
        return self.results(x)
        
        
        




hyperparams = [0.0005811201660180042, 0.010186128250544161, 7.224494817828703e-05, 0.09919650786263239, 0.013120900307549924, 0.004193814188090797, 2.927567762318058e-05, 0.20923637804401124, 0.005184498596714692, 0.045646415508275744]
loss = nn.CrossEntropyLoss()


#Checkpoint 1: hyperparams = [0.0006379996061473911, 0.009580295107975279, 7.347227445833253e-05, 0.06591971996486302, 0.018547741205381177, 0.0038566069717178137, 2.9036672273392558e-05, 0.2171617601498036, 0.004479376782513118, 0.03863763382148639]

#Checkpoint 2: hyperparams = [0.000623740135416117, 0.009240442999425318, 7.991989483253994e-05, 0.05852123150581308, 0.01672334524750428, 0.0036799015355077898, 2.8656037898397567e-05, 0.21603787466354835, 0.003956042023594486, 0.04504845601420199]

#Checkpoint 3: hyperparams = [0.0006319528801574653, 0.009829275381726732, 8.040890517636147e-05, 0.06050047793171963, 0.017052279500885514, 0.0036593223685018185, 2.8766774660612213e-05, 0.21750707077785142, 0.0038972799607081814, 0.045958200326276066]

#Checkpoint 4: hyperparams = [0.0006729867061284829, 0.009413706933048823, 8.121991512547211e-05, 0.058126115950135414, 0.015485414945547402, 0.0036363227090873843, 2.984296557241922e-05, 0.19917532865830914, 0.003696616602346983, 0.04534038294546077]
best = 0

#pretrained = torch.load("Retina_SqueezeAttention6_1.pt") #Let's get up to 10 epochs?
#net.load_state_dict(pretrained)

max_tries = 40
#muon_max = 0.02
#muon_min = 0.005
#adam_max = 3e-4
#adam_min = 7.5e-5

if __name__ == "__main__":
    for epoch in range(max_tries):
        
        hyperparam_update = hyperparams*np.clip(0.15*np.exp((-epoch*1.6)/(max_tries+1))*np.random.randn(10),-1.0,1.0)
        
        
        #net = torch.compile(net) #Counterproductive. Only compile the bottleneck.

        #Muon with new adjustment algorithm. No weight decay because only 3m parameters.
        
        print("Current try:",epoch+1)
        
        new_hyperparams = hyperparams + hyperparam_update
        net = SqueezeAttention(3, 7).to("cuda")
        hidden_weights = [p for p in net.parameters() if p.ndim >= 2][:-1]
        hidden_gains_biases = [p for p in net.parameters() if p.ndim < 2]
        nonhidden_params = [net.results.weight]
        param_groups = [
            dict(params=hidden_weights, use_muon=True,
                 lr=new_hyperparams[0], weight_decay=new_hyperparams[1]),
            dict(params=hidden_gains_biases, use_muon=False,
                 lr=new_hyperparams[2], betas=(1-new_hyperparams[3], 1-new_hyperparams[4]), weight_decay=new_hyperparams[5]),
            dict(params=nonhidden_params, use_muon=False,
                 lr=new_hyperparams[6], betas=(1-new_hyperparams[7], 1-new_hyperparams[8]), weight_decay=new_hyperparams[9])
        ]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

        net.train()
        correct_positive = 0
        #batch = 0
        for sub_epoch in range(1):
            
            for data_input, result in train_data_loader:
                
                #batch += 1
                #if batch % 100 == 0:
                    #print("batch:",batch, "reached")
                result = result.to("cuda",non_blocking = True)
                prediction = net(data_input.to("cuda",non_blocking = True))
                result_loss = loss(prediction,result.view(-1))
                result_loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
            
            
            
            #print(result_loss)
        
            net.eval()
            with torch.no_grad():
                for data_input, result in val_data_loader:
                    result = result.to("cuda",non_blocking = True)
                    prediction = net(data_input.to("cuda",non_blocking = True))
                    correct_positive += (prediction.argmax(dim=1) == result.view(-1)).sum().item()
        
        
        
        new_hyperparams = hyperparams - hyperparam_update
        net = SqueezeAttention(3, 7).to("cuda")
        hidden_weights = [p for p in net.parameters() if p.ndim >= 2][:-1]
        hidden_gains_biases = [p for p in net.parameters() if p.ndim < 2]
        nonhidden_params = [net.results.weight]
        param_groups = [
            dict(params=hidden_weights, use_muon=True,
                 lr=new_hyperparams[0], weight_decay=new_hyperparams[1]),
            dict(params=hidden_gains_biases, use_muon=False,
                 lr=new_hyperparams[2], betas=(1-new_hyperparams[3], 1-new_hyperparams[4]), weight_decay=new_hyperparams[5]),
            dict(params=nonhidden_params, use_muon=False,
                 lr=new_hyperparams[6], betas=(1-new_hyperparams[7], 1-new_hyperparams[8]), weight_decay=new_hyperparams[9])
        ]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
        
        net.train()
        correct_negative = 0
        #batch = 0
        for sub_epoch in range(1):
            
            for data_input, result in train_data_loader:
                
                #batch += 1
                #if batch % 100 == 0:
                    #print("batch:",batch, "reached")
                result = result.to("cuda",non_blocking = True)
                prediction = net(data_input.to("cuda",non_blocking = True))
                result_loss = loss(prediction,result.view(-1))
                result_loss.backward()
                
                optimizer.step()
                optimizer.zero_grad()
            
            
            
            #print(result_loss)
        
            net.eval()
            with torch.no_grad():
                for data_input, result in val_data_loader:
                    result = result.to("cuda",non_blocking = True)
                    prediction = net(data_input.to("cuda",non_blocking = True))
                    correct_negative += (prediction.argmax(dim=1) == result.view(-1)).sum().item()
        
        
        hyperparams += hyperparam_update*(0.2*np.clip((correct_positive-correct_negative)/15,-5.0,5.0))
        
        print("positive:", correct_positive)
        print("negative:", correct_negative)
        
        print([float(i) for i in hyperparams])

#torch.save(net.state_dict(),"SEnet_breast.pt")

#Baseline: 
#Breast mnist: 0.896

#This one:
#Breast mnist: 0.7179
#Smaller version: 0.7308
# 90k parameters only! 
#Change to (8,64)
# about 1m parameters.
#0.7692

#Add up_projection. 
#0.8333
# only about 2m parameters.
#Larger model doesn't seem to help. (About 3m params.)
#0.8333
#On the second thought... After some more training, the 3m params beat the 2m params version.
#0.8590

#On the other hand, it is very compute-heavy. This one took 102.3(G) MACS compared to 4.14(G) in resnet50.

#V3 doesn't look so good on the first attempt. 0.7949
#Try bringing back the max pool. 
#0.8077

#Breast mnist with more params:
#0.8205

#8 heads.
#0.7436 Horrible result.

#Reverting again. Now, try betas = (0.8,0.96)
#0.7692 Not working.

#The previous evaluations were noisy because I forgot to set the net in eval mode.

#The best model so far got this. 
#0.8654 (Breast_SqueezeAttention4_2)

#With the adjusted betas... 0.7756 Not working.

#Try the version with the correct evaluation (20 epochs): 0.8526
#(Breast_SqueezeAttention10_1)

#Try a second round to push the number higher. (Not trained to completion.)

# 0.8718

#Breast mnist: 0.814

#With jitter = 0.3: 0.8526

#With rotation = 30: 0.859

#With more layers: 0.8718 (version 18.)

#Version 19 got only like 0.78. Maybe the training got interrupted. Trying again.

#Real version 19: 
# 0.859

#With even more layer (version 20): 0.782

#Version 23: With adam for the first layer.
#0.8077

#Version 24: good validation, but only 0.8462 test.

#Trying again, 30 epochs. 0.827

#With dropout = 0.5 (4 blocks at the end): 0.878 (version 26)

#With dropout = 0.5, 10 epochs, 3 blocks at the end: 0.82

#With dropout = 0.5, 30 epochs, 6 blocks at the end: 0.859

#With label smoothing: (from version 26): 0.8654 (This is version 29.)

#Label smoothing = 0.05: 0.8654

#With norm: 0.8269

#With norm in a different way: 0.8526

#Weight decay = 0.01: 0.8462

#Lower data augmentation: 0.8782

#With weight decay = 0.03: 0.8782

#The best one is weight decay = (0.02, 0.002) (version 38.)

#With weight decay = (0.02,0.005) and adjusted betas... not working. 0.833

#With the adjusted betas alone... 0.8653. Not working.

#With weight decay = (0.02, 0.002): 0.8717

#With weight decay = (0.01, 0.001): 0.8910

#New one: 0.8974 (version 45)

#New one: 0.8526 (version 46)

#Version 47: 0.8846153846153846

#Version 48: 0.8397435897435898

#Version 49: 0.8782051282051282

#Version 50: 0.8846153846153846

#Version 51: 0.8782051282051282

#Version 52: 0.8589743589743589

#Version 53: 0.8846153846153846

#Version 54: 0.8910256410256411

#Version 55: (5x5 depthwise separable): 0.9166666666666666

#Version 56: (7x7): 0.8717948717948718

#Version 57: (Use adam on depthwise): 0.8717948717948718 Rejected.