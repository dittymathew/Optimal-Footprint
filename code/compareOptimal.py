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
ep=10
dataset='hardware'
print 'hardware, Error percent',ep
print 'start time ',str(datetime.now())
#dataset = datasets.load_diabetes()
#(data_X, data_Y) =boston()# dataset.data, dataset.target
#(data_X, data_Y) =auto_mpg()# dataset.data, dataset.target
(data_X, data_Y) =hardware()# dataset.data, dataset.target
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
      weights[nbr]= 1.0/float(distances[i][j]*distances[i][j])
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
  fp2.append(c)
  fp3.append(c)

#print len(fp1),len(coveringCases)
FP={}
FP['rc']=fp1
FP['rs']=fp2
FP['wrs']=fp3
w_str='Model, Size, Loss\n'
loss1=computeLoss(w,fp1,CB)
loss2=computeLoss(w,fp2,CB)
loss3=computeLoss(w,fp3,CB)
print 'Relative Coverage', len(fp1), loss1
w_str+= 'FP_RC,'+str(len(fp1))+','+str(loss1)+'\n'
#optimize(CB,s_fp1)
print 'Retention score', len(fp2),loss2
w_str+= 'FP_RS,'+str(len(fp2))+','+str(loss2)+'\n'
#optimize(CB,s_fp2)
print 'Weighted Retention Score', len(fp3), loss3
w_str+= 'FP_WRS,'+str(len(fp3))+','+str(loss3)+'\n'
#optimize(CB,s_fp3)
alpha=0
threshold=[0.25,0.5,0.75,1]
for t in threshold:
  (W,Cov) =assignWeights(Gdata,t)
  (constraints,expression,fp_var)=addConstraints(Gdata,W,Cov,alpha)
  fp=optimize(Gdata,alpha,constraints,expression,fp_var)
  loss4=computeLoss(w,fp,CB)
  w_str+= 'FP_WRS,'+str(len(fp))+','+str(loss4)+'\n'
  FP[t]=fp
  
pickle.dump((FP,w_str),open(dataset+'/'+dataset+'_compare_ep_'+str(ep)+'.p','wb'))

  
