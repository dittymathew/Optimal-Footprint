import numpy as np
#from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.model_selection import train_test_split,KFold
from footprintCA import *
from footprint import *
from optimalFootprint import *
import sys
from sklearn import neighbors,datasets,preprocessing
import networkx as nx
import pickle
from loadDataset import *


k=1
ep=5
dataset='automobile'
print dataset, ep
save_file=dataset+'/fp_regression_result_'+dataset+'-ep'+str(ep)+'_beta.p'
#dataset = datasets.load_boston()
#dataset = datasets.load_diabetes()
#data_X, data_Y = dataset.data, dataset.target
(data_X,data_Y)=automobile()
#(data_X,data_Y)=diabetes()
#(data_X,data_Y)=hardware()
data_X= (data_X-data_X.min(axis=0))/(data_X.max(axis=0)-data_X.min(axis=0))
#data_X=(data_X-data_X.mean(axis=0))/(data_X.std(axis=0))
#print data_Y
#data_X=preprocessing.scale(data_X)
#data_X =(data_X-data_X.mean())/data_X.std()
#data_X =(data_X-data_X.min())/(data_X.max()-data_X.min())
#X_train, X_test, y_train, y_test = train_test_split(data_X,data_Y,test_size=0.30, random_state=0)
#X_train=preprocessing.scale(X_train)
#X_test=preprocessing.scale(X_test)
#X_train =(X_train-X_train.mean())/X_train.std()
#X_train =(X_train-X_train.min())/(X_train.max()-X_train.min())
#X_test =(X_test-X_test.min())/(X_test.max()-X_test.min())
#X_test =(X_test-X_test.mean())/X_test.std()
#nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X_train)
#distances, indices = nbrs.kneighbors(X_train)

best_rss=float('inf')
cv = KFold(n_splits=5, shuffle=True,random_state=3)#44)
model=['rc','wrs',0,0.2,0.4,0.5,0.6,0.8,1]
beta=[0.2,0.4,0.6,0.8,1]
rss={}
fps={}
cov={}
for itm in model:
  rss[itm]=[]
  fps[itm]=[]
  cov[itm]=[]
for train,test in cv.split(data_X):
  X_train=data_X[train,:]
  y_train=data_Y[train]
  X_test=data_X[test,:]
  y_test=data_Y[test]
  print train.shape, X_train.shape
  nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X_train)
  distances, indices = nbrs.kneighbors(X_train)
  Gdata= {}
  for i in range(len(indices)):
    Gdata[i]=[]
    y_actual = int(y_train[i])
    y_predict=0
    weights={} 
    sum_weights=0
    soln=[]
    for j in range(indices[i].shape[0]):
      nbr=indices[i][j]
      if nbr != i:
        soln.append(nbr)
        weights[nbr]= 1.0/(1+float(distances[i][j]*distances[i][j]))
        y_predict += float(weights[nbr]*y_train[nbr])
        sum_weights +=weights[nbr]
    y_predict =round(float(y_predict)/float(sum_weights),2)
    if y_actual >y_predict:
      y_diff =y_actual-y_predict
    else:
      y_diff =y_predict-y_actual
    wght = 1.0/float(1+(y_diff*y_diff))
    error_percent=float(y_diff*100)/float(y_actual)
    if error_percent <=ep:
      Gdata[i].append((soln,wght))
  y_test = y_test.reshape(y_test.shape[0],1)
  res_er=0
  y_mean=y_test.mean()
  for i in range(y_test.shape[0]):
    res_er += (y_mean-y_test[i][0])

  alpha=0.5
  fp=[]
  for t in beta:
    (W,Cov) =assignWeightsBeta(Gdata,t)
    (constraints,expression,fp_var)=addConstraints(Gdata,W,Cov,alpha)
    if len(fp)!=len(Gdata):
      fp=optimize(Gdata,alpha,constraints,expression,fp_var)
    fps[t].append(fp)
    fp.sort()
    mod_x_train = X_train[fp,:]
    mod_y_train = y_train[fp].reshape(len(fp),1)
    knn = neighbors.KNeighborsRegressor(k, weights='distance')
    knn.fit(mod_x_train, mod_y_train)
    rss1 =0
    for i in range(y_test.shape[0]):
      X= [list(X_test[i,:])]
      dists =knn.kneighbors(X)[0][0]
      nbrs =knn.kneighbors(X)[1][0]
      yhat=0
      sum_weights=0
      for j in range(len(dists)):
        wgt=1.0/(1+float(dists[j]*dists[j]))
        yhat += float(wgt*mod_y_train[nbrs[j]])
        sum_weights += wgt
      yhat =round(float(yhat)/float(sum_weights),2)
     # rss1 += np.power(yhat - y_test[i][0],2)
      ydiff= yhat - y_test[i][0]
      #if ydiff<0:
      #  ydiff=ydiff*-1
      rss1 +=ydiff*ydiff
    fp_rss= float(rss1)/float(y_test.shape[0])
    rss[t].append(fp_rss)
  pickle.dump((fps,rss),open(save_file,'wb'))


