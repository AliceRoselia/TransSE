# -*- coding: utf-8 -*-
"""
Created on Wed May 27 17:31:58 2026

@author: User
"""

from medmnist import RetinaMNIST
from itertools import islice

train_data = RetinaMNIST(split="train",size = 224)

for i in islice(train_data, 30):
    if (i[1][0] != 0):
        continue
    i[0].show()
    #input("Press any key for the next image.")
    