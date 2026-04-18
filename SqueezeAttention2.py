# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 20:42:47 2026

@author: User
"""

import torch
import torch.nn as nn
import torch.nn.functional as func

from medmnist import BreastMNIST
import torchvision.transforms as transforms
import torch.utils.data as data

#from torch.nn.attention import SDPBackend, sdpa_kernel

torch.manual_seed(44542)

torch.set_float32_matmul_precision("high")


#TransSEnet for medical imaging and similar tasks.



class residualBlock(nn.Module):
    
    def __init__(self,submodule):
        super(residualBlock,self).__init__()
        self.submodule = submodule
    
    def forward(self,x):
        #Simple residual block to be fixed later if we want a more complex one.
        return x + self.submodule(x)
    

        
#https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/ in case you need 
#a flexible attention.

    
batch_size = 16


train_data = BreastMNIST(split="train",transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    # Optional: transforms.RandomResizedCrop(224, scale=(0.9,1.0))
]),download=True,size = 224)
train_data_loader = data.DataLoader(dataset = train_data, batch_size = batch_size,shuffle = True)

val_data = BreastMNIST(split="val",transform = transforms.ToTensor(),download=True,size = 224)
val_data_loader = data.DataLoader(dataset = val_data, batch_size = batch_size,shuffle = True)

test_data = BreastMNIST(split="test",transform = transforms.ToTensor(),download=True,size = 224)
test_data_loader = data.DataLoader(dataset = test_data, batch_size = batch_size,shuffle = False)


#Use torch.nn.functional.scaled_dot_product_attention

class SqueezeAttentionBlock(nn.Module):
    
    def __init__(self,m,n, head = 4):
        super(SqueezeAttentionBlock,self).__init__()
        assert m%head == 0
        self.channel_group_count = m
        self.qk = nn.Linear(n,2*n)
        self.heads = head
        self.value_conv = nn.Conv2d(n, n, 1)
        self.bn1 = nn.BatchNorm2d(m*n)
        self.conv = nn.Conv2d(n, n, 3, padding = "same")
        #self.pw_conv = nn.Conv2d(n,n,1,padding = "same")
        self.bn2 = nn.BatchNorm2d(m*n)
    
    #def convs(self,x):
        #return self.pw_conv(self.dw_conv(x))
    
    def forward(self,x):
        # X is of shape [B,M,N,H,W]
        B,M,N,H,W = x.shape
        channel_reps = x.mean((3,4)) #dimension: B,M,N
        
        
        query, key = self.qk(channel_reps).view(B,M,self.heads,N*2//self.heads).transpose(1,2).chunk(2,dim=3) #Dimensions B,Head,M,N/Head 
        value = self.value_conv(x.view(B*M,N,H,W)).view(B,M,self.heads,N//self.heads,H,W).transpose(1,2) #Dimensions: B,Head,M,N/Head,H,W
        scale = (N//self.heads) ** -0.5
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale #Dimensions B, Head, M, M
        
        attn = func.softmax(scores,dim=-1)
        
        attention_result = attention_result = torch.einsum('baij,bajchw -> baichw', attn, value).transpose(1,2)
        
        
        #Manual attention result because optimized kernels weren't optimized for this.
        
       #with sdpa_kernel(backends=[SDPBackend.MATH]):
            #attention_result = func.scaled_dot_product_attention(query,key,value).view(B,M,N,H,W)
        attention_result = attention_result.reshape(B,M,N,H,W) + x
        attention_result = self.bn1(attention_result.view(B,M*N,H,W)).view(B*M,N,H,W) #Dimensions: B*M,N,H,W
        
        attention_result = self.bn2((attention_result + func.relu(self.conv(attention_result))).view(B,M*N,H,W))
        
        
        return attention_result.view(B,M,N,H,W)

class UpProjection(nn.Module):
    def __init__(self,n,n2):
        super(UpProjection,self).__init__()
        self.conv = nn.Conv2d(n, n2, 1, padding = "same")
        self.n2 = n2
    def forward(self,x):
        B,M,N,H,W = x.shape
        return self.conv(x.view(B*M,N,H,W)).view(B,M,self.n2,H,W)

        
        
class SqueezeAttention(nn.Module):
    def squeeze_to_pool(self,x):
        B,M,N,H,W = x.shape
        return func.max_pool2d(x.view(B*M,N,H,W), 2).view(B,M,N,H//2,W//2)

    
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
        """
        self.SAB9 = SqueezeAttentionBlock(8, 64)
        self.SAB10 = SqueezeAttentionBlock(8, 64)
        self.SAB11 = SqueezeAttentionBlock(8, 64)
        self.SAB12 = SqueezeAttentionBlock(8, 64)
        self.SAB13 = SqueezeAttentionBlock(8, 64)
        self.SAB14 = SqueezeAttentionBlock(8, 64)
        self.SAB15 = SqueezeAttentionBlock(8, 64)
        self.SAB16 = SqueezeAttentionBlock(8, 64)
        self.SAB17 = SqueezeAttentionBlock(8, 64)
        self.SAB18 = SqueezeAttentionBlock(8, 64)
        self.SAB19 = SqueezeAttentionBlock(8, 64)
        self.SAB20 = SqueezeAttentionBlock(8, 64)
        self.SAB21 = SqueezeAttentionBlock(8, 64)
        """
        
        self.dropout = nn.Dropout(0.5)
        
        self.results = nn.Linear(2048, classes)
    
    def forward(self,x):
        B,C,H,W = x.shape
        x = self.intro(x).view(B,8,32,H,W)
        x = self.SAB1(x)
        x = self.SAB2(x)
        x = self.SAB3(x)
        x = self.squeeze_to_pool(x)
        x = self.UP1(x)
        x = self.SAB4(x)
        x = self.SAB5(x)
        x = self.SAB6(x)
        x = self.squeeze_to_pool(x)
        x = self.UP2(x)
        x = self.SAB7(x)
        x = self.SAB8(x)
        x = self.SAB9(x)
        x = self.squeeze_to_pool(x)
        x = self.UP3(x)
        x = self.SAB10(x)
        x = self.SAB11(x)
        x = self.SAB12(x)
        
        x = self.dropout(x.mean((3,4)).view(-1,2048))
        
        return self.results(x)
        
        
        



net = SqueezeAttention(1, 2).to("cuda")
#net = torch.compile(net)
optimizer = torch.optim.Adam(net.parameters(),lr = 1.5e-4)
loss = nn.CrossEntropyLoss()

#pretrained = torch.load("SEnet.pt")

#del pretrained["layers.0.weight"]
#del pretrained["layers.0.bias"]
#del pretrained["results.weight"]
#del pretrained["results.bias"]

#net.load_state_dict(pretrained,strict=False)

best = 0

for epoch in range(10):
    
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
    with torch.no_grad():
        for data_input, result in val_data_loader:
            result = result.to("cuda")
            prediction = net(data_input.to("cuda"))
            correct += (prediction.argmax(dim=1) == result.view(-1)).sum()
    
    print("correct:",correct)
    #Out of 120 for retina.
    #78 for breast.
    if correct > best:
        best = correct
        print("New frontier reached.")
        torch.save(net.state_dict(),"Breast_SqueezeAttention3_1.pt")
    
pretrained = torch.load("Breast_SqueezeAttention3_1.pt") #Let's get up to 10 epochs?
net.load_state_dict(pretrained)


correct = 0
total = 156 
    
with torch.no_grad():
    
    for data_input, result in test_data_loader:
        result = result.to("cuda")
        prediction = net(data_input.to("cuda"))
        correct += (prediction.argmax(dim=1) == result.view(-1)).sum()

print("accuracy: ",correct / total)

#torch.save(net.state_dict(),"SEnet_breast.pt")

#Baseline: 
#Breast mnist: 0.896
#Retina mnist: 0.561 

#This one:
#Breast mnist: 0.7179
#Smaller version: 0.7308
# 90k parameters only! 
#Change to (8,64)
# about 1m parameters.
#0.7692

#Add up_projection. 
#0.8333
# 2m parameters.


