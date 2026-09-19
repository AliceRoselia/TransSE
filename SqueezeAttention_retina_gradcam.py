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
#from torch_optimizer import Lookahead
#from muon import SingleDeviceMuonWithAuxAdam

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
#We still can't use Pytorch native implementation because it lacks suppport for 4d conv params, so we will use Keller's version.
torch._dynamo.config.recompile_limit = 128
torch._dynamo.config.cache_size_limit = 128 
torch._dynamo.config.accumulated_cache_size_limit = 128

import matplotlib.pyplot as plt

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

    
batch_size = 1
num_workers = 4
prefetch_factor = 4

train_data = RetinaMNIST(split="train",transform = transforms.ToTensor(),
    download=True,size = 224)
train_data_loader = data.DataLoader(dataset = train_data, batch_size = batch_size,shuffle = True,
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
        self.depthwise_conv = nn.Conv2d(n, n, 3, padding = "same",groups=n)
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

class ResultShape(nn.Module):
    def __init__(self):
        super(ResultShape,self).__init__()
    def forward(self,x):
        B,M,N,H,W = x.shape
        return x.reshape(B,M*N,H,W)
        
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
        
        
        
        self.UP1 = UpProjection(32, 64)
        self.UP2 = UpProjection(64, 128)
        self.UP3 = UpProjection(128, 256)
        
        self.OUTPUT_SHAPE = ResultShape()
        
        self.dropout = nn.Dropout(0.25)
        
        self.results = nn.Linear(2048, classes)
    #@torch.compile() We will not need to compile for small-scale test.
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
        #x = self.squeeze_to_pool(x) #14
        x = self.OUTPUT_SHAPE(x)
        
        
        
        x = self.dropout(x.mean((2,3)).view(-1,2048))
        
        return self.results(x)
        
        
        



net = SqueezeAttention(3, 5).to("cuda")

def gradcam_plot(model, data_loader, limit = 10):
    print("Okay")
    count = 0
    #targets = [ClassifierOutputTarget(2)]
    with GradCAM(model = model, target_layers = [model.OUTPUT_SHAPE]) as cam:
        for data_input, result in data_loader:
            #print(data_input.shape)
            cam_result = cam(input_tensor=data_input,targets = None)
            cam_result = cam_result[0,:]
            cam_image = show_cam_on_image(data_input[0].cpu().numpy().transpose(1,2,0), cam_result, use_rgb=True)
            
            plt.imshow(cam_image)
            plt.show()
            count += 1
            if count >= limit:
                break
            
            
            
            
    



if __name__ == "__main__":
    pretrained = torch.load("Retina_SqueezeAttention53_1.pt") #Let's get up to 10 epochs?
    net.load_state_dict(pretrained)


    correct = 0
    total = 400 
        
    net.eval()
        
    gradcam_plot(net, train_data_loader)

#Try a second round to push the number higher. (Not trained to completion.)

# 0.8718
#Retina mnist:
#0.5375
#Retina_SqueezeAttention2_1

#Now, version 3. Let's change to Gelu.
#0.5700

#Let's try breastMNIST with it.
#0.8205 (Not so good yet.)
#Final result: 0.8526

#Version 12. Let's see if it was a fluke.
#0.8205
#With actual training, silu: 
#0.8718
#Retina mnist:
#0.565'

#With Muon optimizer: (stopped before 10 epochs during the 8th epochs using the result from the 5th epoch.)
# 0.6175. SOTA!
# With 5 epochs: 0.595
# With another 5 epochs: 0.6075
# Trying 10 epochs with a different seed: 
# 0.625 (Yay!) (Retina_SqueezeAttention7_1)

# Without color jitter: 0.5325

# With dropout reduced to 0.25: 0.6275 (Yay!)

#Oops, accidentally overwrote squeezeattention9_1: With only 0.1 jitter: 0.62

# With jitter = 0.3, horrible. (0.525)

#Delete squeezeattention_9_1 and try 0.15 color jitter.

#0.6425! Yay! *(Squeezeattention9_1) Also checkpoint 5

#With head = 8: 0.6175 (v10)

#Now, update to v11 and try head=8 for the last 2 resolutions only: 0.5825

#Try another seed with 20 epochs: 0.5975
#So, if not that lucky, could be worse.

#With another seed, 25 epochs: 
# 0.615

#Version 15: More layers!
# 0.6075

#Version 17: 0.6025

#Try again (20 epochs): 0.545


#Heard you like param tunes? 0.5975 with weight decay = 0.02. Wait... wrong... that was lr=0.02
# The best one is with lr = 0.01 and 1.5e-4. (Version 19), 0.655

#With lr = 0.02 on both: 0.6425

#Version 22: with changed betas to (0.9, 0.95): 0.655

#Version 23: 0.62

#Version 24: 0.6025

#Version 25: 0.595

#Version 33: 0.585

#Version 35: 0.575

#Version 37: 0.585

#Version 38: 0.5975

#Version 39: 0.58

#Version 41: 0.585

#Version 42: 0.6125 (Add extra block)

#Version 43: 0.6525

#Version 44: 0.6275

#Version 45: 0.6475

#Version 46: 0.6

#Version 47: 0.62

#Version 48: 0.6225

#Version 49: 0.6425

#Version 50: 0.63

#Version 51 (25 epochs): 0.62

#Version 52: With the same tuned parameters: 0.6425

#Version 53: With depthwise-separable convs: 0.65

#Version 54: 5x5: 0.6