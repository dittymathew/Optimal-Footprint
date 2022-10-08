import numpy as np
from datetime import datetime
from cvxpy import *
#from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.model_selection import train_test_split,KFold
from footprintCA import *
import sys
from sklearn import neighbors,datasets,preprocessing
import networkx as nx
from footprint import *
import pickle
from loadDataset import *
from optimalFootprint import *

#  return x.value
k=1
ep=5
dataset='automobile'
print dataset.upper(),ep
print 'start time ',str(datetime.now())
#dataset = datasets.load_diabetes()
#(data_X, data_Y) =hardware()# dataset.data, dataset.target
(data_X, data_Y) =automobile()# dataset.data, dataset.target
#(data_X, data_Y) =diabetes()# dataset.data, dataset.target
#(data_X, data_Y) =boston()# dataset.data, dataset.target
data_X= (data_X-data_X.min(axis=0))/(data_X.max(axis=0)-data_X.min(axis=0))
#data_X=data_X[0:200,:]
#data_Y=data_Y[0:200]


nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(data_X)
distances, indices = nbrs.kneighbors(data_X)
Gdata= {}
epsilon=25
for i in range(len(indices)):
  Gdata[i]=[]
  y_actual = int(data_Y[i])
  y_predict=0
  weights={} 
  sum_weights=0
  soln=[]
  for j in range(indices[i].shape[0]):
    nbr=indices[i][j]
    if nbr != i:
      soln.append(nbr)
      weights[nbr]= 1.0/(1+float(distances[i][j]*distances[i][j]))
      y_predict += float(weights[nbr]*data_Y[nbr])
      sum_weights +=weights[nbr]
  y_predict =round(float(y_predict)/float(sum_weights),2)
  if y_actual >y_predict:
    y_diff =y_actual-y_predict
  else:
    y_diff =y_predict-y_actual
  wght = 1.0/float(1+(y_diff*y_diff))
  error_percent=float(y_diff*100)/float(y_actual)
#  if y_diff*y_diff <=epsilon:
  if error_percent <=ep:
    Gdata[i].append((soln,wght))

print 'Created network, current time ',str(datetime.now())
#print Gdata
#(w,C)=optimalFp(Gdata,len(fp1))
CB=Gdata
#CB ={0:[([3],0.4)],1:[([0],1),([2],0.2)], 2:[([1],0.3),([3],0.5)],3:[([4],0.3),([0],0.9)],4:[([3],1)]}
#CB ={0:[([3],1)],1:[([0],1),([2],1)], 2:[([1],1),([3],1)],3:[([4],1),([0],1)],4:[([3],1)]}
cb_size=len(CB)
C=[0 for i in range(cb_size*cb_size)]
C=np.reshape(np.matrix(C),(cb_size,cb_size))
w=np.matrix(np.zeros((cb_size,cb_size)))
for i in CB.keys():
  for (soln,wgt) in CB[i]:
    for j in soln:
      w[j,i] = wgt
      C[j,i] = 1
      #print i,j,wgt
  w[i,i]=1
  C[i,i]=1
#w_max=w.max()
#w_min=w.min()
#print w_max,w_min
#for i in range(len(CB)):
 # w_max=
#  for j in range(len(CB)):
#    if i !=j:
#      w[i,j]= float(w[i,j]-w_min)/float(w_max-w_min)
#print w.max(),w.min()
idleCases=[]
coveringCases=[]
for i in range(len(CB)):
  if C[i,:].sum()>1:
    coveringCases.append(i)
  if len(CB[i])==0:
    idleCases.append(i)
fp_rc= Footprint()
fp1=fp_rc.footprintOr(CB)
fp_ca = FootprintCA()
(ret_score,reachability) =fp_ca.retentionScore(CB)
fp2 = fp_ca.footprint(ret_score,reachability)
(ret_score,reachability) =fp_ca.retentionScoreWeighted(CB)
fp3 = fp_ca.footprint(ret_score,reachability)
for c in idleCases:
  fp1.append(c)
  fp2.append(c)
  fp3.append(c)

#print len(fp1),len(coveringCases)
FP={}
FP['rc']=fp1
FP['rs']=fp2
FP['wrs']=fp3
print 'Relative Coverage', computeLoss(w,fp1,CB), len(fp1)
#optimize(CB,s_fp1)
print 'Retention score', computeLoss(w,fp2,CB),len(fp2)
#optimize(CB,s_fp2)
print 'Weighted Retention Score', computeLoss(w,fp3,CB),len(fp3)
#optimize(CB,s_fp3)
alpha=0
(W,Cov) =assignWeights(Gdata)
(constraints,expression,fp_var)=addConstraints(Gdata,W,Cov,alpha)
fp=optimize(Gdata,alpha,constraints,expression,fp_var)
FP[0]=fp
print computeLoss(w,fp,CB) 
alpha=0.1
(W,Cov) =assignWeights(Gdata)
(constraints,expression,fp_var)=addConstraints(Gdata,W,Cov,alpha)
model= ['rc','rs','wrs',0,0.2,0.4,0.5,0.6,0.8,1]
for i in range(4,len(model)):
  alpha=model[i]
  fp=optimize(Gdata,alpha,constraints,expression,fp_var)
  FP[alpha]=fp
  print computeLoss(w,fp,CB) 
pickle.dump(FP,open(dataset+'/'+dataset+'_tradeoff_ep_'+str(ep)+'.p','wb'))
  
