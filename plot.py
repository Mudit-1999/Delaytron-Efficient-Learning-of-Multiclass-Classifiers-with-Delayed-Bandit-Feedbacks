import os
from typing import *
import numpy as np
import random
import numpy.linalg as LA
import numba as nb
import pandas as pd
from numba import jit,float32,int32 
import types
import pickle
import numpy as np
from collections import defaultdict
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import sys
import sklearn
from sklearn.datasets import load_iris,load_wine,load_digits
import argparse




if __name__=="__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--size", type=float,default=1e6)

  args, _ = parser.parse_known_args()

  prefix = os.getcwd()
  
  algos=['ban_dbf' , 'soba']
  ealgos=['1' , '100', '1000',  '2500',  '5000']
  datasets={
      'digits':'Digits',
      'fashion':'F-MNIST',
      'synnonsep':'Syn-NonSep',
      'synsep':'Syn-Sep',
      'cifar10':'CIFAR-10',
      'abalone':'Abalone',
      'ecoli':'Ecoli',
      'usps':'USPS',
      }
  print(args)
  for key,name in datasets.items():
    data=[]
    frac_data=[]
    query_data=[]
    min_len=int(1e7)
    for i,algo in enumerate(ealgos):
      data.append(np.load(f'{prefix}/ban_dbf/{key}/error_rate_{algo}.npy'))
    data.append(np.load(f'{prefix}/soba/{key}/error_rate.npy'))
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.grid()
    ax.set_xscale('log')
    indx=np.arange(0,args.size,100,int)
    print(indx[-1])
    ax.plot(indx, data[0][0][indx],linestyle='dashed',color='g' ,label='Banditron',marker='x',markevery=1/10)
    ax.plot(indx, data[5][0][indx],linestyle='dotted',color='c' ,label='SOBA',marker='>',markevery=1/10)
    ax.plot(indx, data[1][0][indx],linestyle='dotted',color='b' ,label='Delaytron(D=100)',marker='*',markevery=1/10)
    ax.plot(indx, data[2][0][indx],linestyle='dashed',color='r' ,label='Delaytron(D=1000)',marker='o',markevery=1/10)
    ax.plot(indx, data[3][0][indx],linestyle='dotted',color='m' ,label='Delaytron(D=2500)',marker='d',markevery=1/10)
    ax.plot(indx, data[4][0][indx],linestyle='dashed',color='k' ,label='Delaytron(D=5000)',marker='P',markevery=1/10)
    ax.set_xlabel("Number of Examples",fontsize=13)
    ax.set_ylabel("Error Rate",fontsize=13)
    ax.legend(prop={'size' : 11},loc=1)
    ax.plot()

    f.suptitle(f'{name}', fontsize=13)
    f.subplots_adjust(wspace=0.3)
    f.subplots_adjust(hspace=0.5)
    f.savefig(f'{prefix}/ban_dbf/results/{key}.png')
    f.clf()

 