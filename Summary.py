# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 22:35:21 2026

@author: User
"""

hyperparams = [0.0005811201660180042, 0.010186128250544161, 7.224494817828703e-05, 0.09919650786263239, 0.013120900307549924, 0.004193814188090797, 2.927567762318058e-05, 0.20923637804401124, 0.005184498596714692, 0.045646415508275744]
#Retina: 3x3, version 60: acc = 0.64
#Breast: 5x5, version 72: acc = 0.8846
#Derma:  5x5, version 12: dropout = 0.1, acc = 0.8204
#Blood:  5x5, version 6: acc = 0.9752
#Pneumonia: 3x3, version 13: acc = 0.9359
seed = 71909099

#Transfer learning from different medical dataset didn't work well.
