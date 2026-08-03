# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:31:58 2026

@author: User
"""

from medmnist import BreastMNIST
from itertools import islice
import torchvision.transforms as transforms
import torch.utils.data as data

batch_size = 16
num_workers = 4
prefetch_factor = 8 

train_data = BreastMNIST(split="train",size = 224,download=True)

for i in islice(train_data, 200):
    
    #print(i[1][0])
    i[0].show()
    input("Press any key for the next image.")
    
    
test_data = BreastMNIST(split="test",transform = transforms.ToTensor(),download=True,size = 224)
test_data_loader = data.DataLoader(dataset = test_data, batch_size = batch_size,shuffle = False,
pin_memory=True,num_workers=num_workers,prefetch_factor=prefetch_factor,persistent_workers=True)