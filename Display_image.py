# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:31:58 2026

@author: User
"""

from medmnist import BreastMNIST
from itertools import islice

train_data = BreastMNIST(split="train",size = 224,download=True)

for i in islice(train_data, 200):
    if (i[1][0] != 0):
        continue
    #print(i[1][0])
    i[0].show()
    input("Press any key for the next image.")
    