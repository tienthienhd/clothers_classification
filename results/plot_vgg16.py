# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 22:22:06 2018

@author: tienthien
"""

import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

path = 'train_output_vgg/'

optimizers = []


for i, optimizer in enumerate(glob.glob(path + '*/*.csv')):
    print(optimizer)
    opt = optimizer.split('\\')[1]
    optimizers.append(opt)
    df_result = pd.read_csv(optimizer, usecols=['baroque', 'camo', 'color block', 'leopard' , 'solid color'], nrows=51)
    array_mean = np.mean(df_result.values, axis=1)
    array_mean = array_mean.reshape([array_mean.shape[0], 1])
    print(array_mean.shape)
    if i == 0:
        accs = array_mean
    elif i == 3:
        array_mean = array_mean[:50]
        accs = np.concatenate([accs, array_mean], axis=1)
    else:
        accs = np.concatenate([accs, array_mean], axis=1)
accs = np.array(accs)
print(accs.shape)
df_acc = pd.DataFrame(accs, columns=optimizers)
df_acc.plot()
#plt.plot(array_mean, label=opt)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title("Accuracy on Test set")
plt.legend()
plt.show()