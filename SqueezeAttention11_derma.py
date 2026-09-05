# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 20:42:47 2026

@author: User
"""

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

torch.manual_seed(71203674)

torch.set_float32_matmul_precision("high")


#TransSEnet for medical imaging and similar tasks.


    

        
#https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/ in case you need 
#a flexible attention.

    
batch_size = 2
num_workers = 4
prefetch_factor = 40 

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

test_data = DermaMNIST(split="test",transform = transforms.ToTensor(),download=True,size = 224)
test_data_loader = data.DataLoader(dataset = test_data, batch_size = batch_size,shuffle = False,
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
        self.conv = nn.Conv2d(n, n, 3, padding = "same")
        #self.pw_conv = nn.Conv2d(n,n,1,padding = "same")
        self.bn2 = nn.BatchNorm2d(m*n)
        self.head_size = n//head
        scale = torch.full((head,m,m),self.head_size ** -0.5)
        self.register_buffer("scale", scale) #Pre-broadcast
    
    #@torch.compile()
    def fused_conv_activation(self,x):
        return x + func.silu(self.conv(x))
    
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
        #x = self.squeeze_to_pool(x) #14
        
        
        
        x = self.dropout(x.mean((3,4)).view(-1,4096))
        
        return self.results(x)
        
        
        



net = SqueezeAttention(3, 7).to("cuda")
#net = torch.compile(net) #Counterproductive. Only compile the bottleneck.

#Muon with new adjustment algorithm. No weight decay because only 3m parameters.

hidden_weights = [p for p in net.parameters() if p.ndim >= 2][1:-1]
hidden_gains_biases = [p for p in net.parameters() if p.ndim < 2]
nonhidden_params = [net.intro.weight, net.results.weight]
hyperparams = [0.0008602522078379203, 0.012369705667362903, 9.752763156138204e-05, 0.0655110378721131, 0.013811079331863048, 0.010340056669188179, 1.9917593041199128e-05, 0.1769209451676188, 0.007003697900328614, 0.056487999850367676]
param_groups = [
    dict(params=hidden_weights, use_muon=True,
         lr=hyperparams[0], weight_decay=hyperparams[1]),
    dict(params=hidden_gains_biases, use_muon=False,
         lr=hyperparams[2], betas=(1-hyperparams[3], 1-hyperparams[4]), weight_decay=hyperparams[5]),
    dict(params=nonhidden_params, use_muon=False,
         lr=hyperparams[6], betas=(1-hyperparams[7], 1-hyperparams[8]), weight_decay=hyperparams[9])
]
optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

#optimizer = torch.optim.Muon(net.parameters(),weight_decay = 0.0,lr = 1.5e-4,adjust_lr_fn = "match_rms_adamw")
loss = nn.CrossEntropyLoss()

#pretrained = torch.load("SEnet.pt")

#del pretrained["layers.0.weight"]
#del pretrained["layers.0.bias"]
#del pretrained["results.weight"]
#del pretrained["results.bias"]

#net.load_state_dict(pretrained,strict=False)

best = 0

max_epoch = 100



if __name__ == "__main__":
    for epoch in range(max_epoch):
        print("Current epoch:",epoch+1)
        net.train()
        for data_input, result in train_data_loader:
            result = result.to("cuda",non_blocking = True)
            prediction = net(data_input.to("cuda",non_blocking = True))
            result_loss = loss(prediction,result.view(-1))
            result_loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            
            
            #print(result_loss)
        
        net.eval()
        correct = 0
        with torch.no_grad():
            for data_input, result in val_data_loader:
                result = result.to("cuda",non_blocking = True)
                prediction = net(data_input.to("cuda",non_blocking = True))
                correct += (prediction.argmax(dim=1) == result.view(-1)).sum().item()
        
        print("correct:",correct)
        #Out of 120 for retina.
        #78 for breast.
        if correct > best:
            best = correct
            print("New frontier reached.")
            torch.save(net.state_dict(),"Derma_SqueezeAttention6_1.pt")



#This section is deliberately separate in case we want to just evaluate the model.

if __name__ == "__main__":
    pretrained = torch.load("Derma_SqueezeAttention6_1.pt") #Let's get up to 10 epochs?
    net.load_state_dict(pretrained)


    correct = 0
    total = 2005 
        
    net.eval()
    with torch.no_grad():
        
        for data_input, result in test_data_loader:
            result = result.to("cuda",non_blocking = True)
            prediction = net(data_input.to("cuda",non_blocking = True))
            correct += (prediction.argmax(dim=1) == result.view(-1)).sum().item()
    
    print("accuracy: ",correct / total)



#Derma mnist: 0.7406

#Derma mnist 2 (without max pooling before): 0.7387

#Derma mnist 3 (With a few more layers): 0.745

#Derma mnist 5 (with weight decay): 0.7406

#Derma mnist 6 (100 epochs): 0.8040 

#Operation count: 69 Gigaops. 16m params. 20 ms.