# Following banditron ( without pegasos like updates simple banditron)
import os
from typing import *
import numpy as np
import random
import numpy.linalg as LA
import numba as nb
import pandas as pd
from numba import jit,float32,int32 
import types
# from numba.experimental import jitclass
import pickle
import numpy as np
from collections import defaultdict
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt
import sys
import sklearn
from sklearn.datasets import load_iris,load_wine,load_digits
import time
import argparse
import multiprocessing as mp
import   concurrent.futures 

data=None

@jit(nopython=True)
def feedback(true_label:int,predicted_label:int)->int:
  fb=0
  if true_label==predicted_label:
    fb=1
  return fb

@jit(nopython=True)
def cal_prob(cal_label:int,gamma:float,k:int)->np.ndarray:
  prob= np.ones(k)
  prob = prob*gamma/k
  prob[cal_label]+= 1 -gamma
  # print("Prob",prob)
  return prob

@jit(nopython=True)
def random_sample(prob:np.ndarray)->int:
  number = float32(random.random()) * np.sum(prob)
  # print("Sum prob",sum(prob), number)
  for i in range(0,prob.shape[0]):
    if number < prob[i]:
      return i
    number -= prob[i]
  return prob.shape[0]-1



@jit(nopython=True)
def Run(
  k:int,
  d:int,
  size:int,
  gamma:float,
  time_skip:int=100,
  )->Tuple[np.ndarray]:

  weight_matrix=np.zeros((k,d))
  updates=np.zeros((size+2*time_skip,k,d))
  cnt=np.zeros(size+2*time_skip)
  incorrect_classified=0
  error_rate=0
  correct_classified=0
  m_t=0
  recieved_till_now=0
  e=0
  error_rate_list=np.zeros((size))
  for i in range(0,size):
    num=random.randint(0,data.shape[0]-1)
    entry= data[num,:]
    feature_vector=np.reshape(entry[0:d],(d,1))
    true_label=int(entry[d])
    val_f=np.reshape((np.dot(weight_matrix,feature_vector)),-1)
    y_hat=np.argmax(val_f)   # calculated label
    prob=cal_prob(y_hat,gamma,k)
    y_tilde= random_sample(prob) #predicted label 

    if true_label==y_tilde:    
      correct_classified+=1
    else:
      incorrect_classified+=1

    error_rate=incorrect_classified/(i+1)
    error_rate_list[i] = error_rate

    fb_t=i+np.random.randint(0,time_skip, size=1)[0]
    fb=feedback(true_label,y_tilde)
    cnt[fb_t]=cnt[fb_t]+1;
    recieved_till_now =recieved_till_now + cnt[i] 
    m_t= i+1 - recieved_till_now
    if m_t >= power(2.0,e):
      e=e+1

    basis_vec=np.zeros((k,1))
    if fb==1:
      basis_vec[y_tilde,0]+=1/(prob[y_tilde])
    basis_vec[y_hat,0]-=1
    updates[fb_t]+= np.kron(basis_vec,feature_vector.T)
    weight_matrix=weight_matrix +  (np.sqrt(power(2.0,-e)))*updates[i]
  return error_rate_list




if __name__=="__main__":
  parser = argparse.ArgumentParser()

  # hyperparameters sent by the client are passed as command-line arguments to the script.
  parser.add_argument("--data", type=str,default='digits')
  parser.add_argument("--dim", type=int,default=64)
  parser.add_argument("--num_class", type=int,default=10)
  parser.add_argument("--repition", type=int, default=20)
  parser.add_argument("--size", type=float,default=1e6)
  parser.add_argument("--ll_gamma", type=int,default=-10)
  parser.add_argument("--ul_gamma",type=int,default=-1)

  args, _ = parser.parse_known_args()

  print(args)
  # path to the dataset directory 
  prefix=f"/home/{os.getenv('USER')}/exp/dataset"
# Dataset 
  # Iris Data
  if args.data=='iris':
    d1 = load_iris()
    data = np.hstack( (d1.data,d1.target.reshape(-1,1)))
  # SynSep Data
  elif args.data=='synsep':
    data=np.load(f'{prefix}/syn_sep.npy','r')
  # SynNonSep Data
  elif args.data=='synnonsep':
    data=np.load(f'{prefix}/syn_nonsep.npy','r')
  # Fashion MNIST
  elif args.data=='fashion':
    data=np.load(f'{prefix}/fashion.npy','r')
  # USPS dataset 
  elif args.data=='usps':
    dataset=[]
    with open(f'{prefix}/upsp.csv','r') as f:
      lines=f.readlines()
      for i in lines:
        temp=i.strip().split(',')
        dataset.append([np.float64(i) for i in temp])
    data=np.array(dataset)
  elif args.data=='ecoli':
    data=np.load(f'{prefix}/ecoli.npy','r')
  elif args.data=='sat':
    data=np.load(f'{prefix}/sat.npy','r')
  elif args.data=='abalone':
    data=np.load(f'{prefix}/abalone.npy','r')
  elif args.data=='mnist':
    data=np.load(f'{prefix}/mnist.npy','r')
  elif args.data=='cifar10':
    data=np.load(f'{prefix}/cifar10.npy','r')
  elif args.data=='mnist100':
    data=np.load(f'{prefix}/mnist100.npy','r')
  # New 4 Group dataset 
  elif args.data=='news4':
    data=np.load(f'{prefix}/news20.npy','r')
  elif args.data=='wine':
    d1 = load_wine()
    data = np.hstack( (d1.data,d1.target.reshape(-1,1)))
  elif args.data=='digits':
    d1 = load_digits()
    data = np.hstack( (d1.data,d1.target.reshape(-1,1)))
  # print(data.shape)  
  data=np.float64(data)


  # Hyper-parameters
  k=args.num_class
  d=args.dim
  repition=args.repition
  size=int(args.size)
  # gamma
  g_val=2.0**np.arange(args.ll_gamma,args.ul_gamma)
  t_val=[1,100,1000,2500,5000]

  avg_time=0
  final_list=np.zeros([repition,size])
  avg_final_list=np.zeros([2,size])
  best_avg_final_list=None

  avg_time=0
  final_list=np.zeros([repition,size])
  avg_final_list=np.zeros([2,size])
  best_avg_final_list=None


  data_dict={
    'ts':[],
    'gamma':[],
    'error_rate':[],
    'avg_time':[],
  }
  for ts in t_val:
    best_error_rate=10
    best_gamma=1
    best_avg_time=0
    best_delta=0
    file_name=""
    data_file1=""
    data_file2=""
    max_norm=0
    avg_max_norm=0
    for exp in g_val:
      t0= time.perf_counter()
      with concurrent.futures.ProcessPoolExecutor(max_workers=20) as executor:
          output = [executor.submit(Run,k, d, size, exp,ts)for _ in range(0,repition)]
          results = [f.result() for f in concurrent.futures.as_completed(output)]
      t1 = round(time.perf_counter() - t0,2)
      for ind,mat in enumerate(results):
        final_list[ind]=mat
      t1 = round(time.perf_counter() - t0,2)
      avg_time=t1/repition
      avg_final_list[0]=final_list.mean(axis=0)
      avg_final_list[1]=final_list.std(axis=0)
      print(f"{20} Runs Completed for gamma:{exp} and time_skip:{ts} with error_rate {avg_final_list[0,-1]} in {avg_time} seconds")
      
      data_dict['ts'].append(ts)
      data_dict['gamma'].append(exp)
      data_dict['error_rate'].append(avg_final_list[0,-1])
      data_dict['avg_time'].append(avg_time)

      if avg_final_list[0,-1] < best_error_rate:
        best_gamma=exp
        best_avg_time=avg_time
        best_avg_final_list=avg_final_list.copy()
        best_error_rate=avg_final_list[0,-1]

    cwd = os.getcwd()
    cwd= cwd+f'/{args.data}'
    if not os.path.exists(cwd):
      print("making dir",cwd)
      os.mkdir(cwd)

    file_name ="param"
    data_file2="error_rate"

    np.save(f'{cwd}/{data_file2}_{ts}', best_avg_final_list)
    with open(f'{cwd}/{file_name}_{ts}.txt','w') as f:
      f.write("AvgTime: "+str(best_avg_time)+"\n")
      f.write("ErrorRate: "+str(best_error_rate)+"\n")
      f.write("BestGamma: "+str(best_gamma)+"\n")

  df=pd.DataFrame.from_dict(data_dict)
  df.to_parquet(cwd+'/data.parquet')


