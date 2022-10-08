import numpy as np
#from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.model_selection import train_test_split,KFold
from footprintCA import *
from footprint import *
from checkGCoverage_mod import *
from groundKernel_mod import *
import sys
from sklearn import neighbors,datasets,preprocessing
import networkx as nx


k=int(sys.argv[1])
#itr=int(sys.argv[2])
dataset = datasets.load_boston()
data_X, data_Y = dataset.data, dataset.target
#data_X= (data_X-data_X.min(axis=0))/(data_X.max(axis=0)-data_X.min(axis=0))
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
cv = KFold(n_splits=5, shuffle=True,random_state=33)
avg_fpca_rss=0
avg_fpca1_rss=0
avg_fpor_rss=0
avg_fp_rs=0
avg_fp_wrs=0
avg_fp_rc=0
avg_cov_rs=0
avg_cov_wrs=0
avg_cov_rc=0
avg_sanity_rs=0
avg_sanity_wrs=0
avg_sanity_rc=0
for itr in range(1,15):
 epsilon= itr*itr
 avg_fpca_rss=0
 avg_fpca1_rss=0
 avg_fpor_rss=0
 avg_fp_rs=0
 avg_fp_wrs=0
 avg_fp_rc=0
 avg_cov_rs=0
 avg_cov_wrs=0
 avg_cov_rc=0
 avg_sanity_rs=0
 avg_sanity_wrs=0
 avg_sanity_rc=0
 for train,test in cv.split(data_X):
  X_train=data_X[train,:]
  y_train=data_Y[train]
  X_test=data_X[test,:]
  y_test=data_Y[test]
  nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X_train)
  distances, indices = nbrs.kneighbors(X_train)
  Gdata= []
  #wp= open("data/classification_3nn.txt",'w')
  for i in range(len(indices)):
    itm = str(i)+'<-'
    y_actual = int(y_train[i])
    y_predict=0
    weights={} 
    sum_weights=0
    for j in range(indices[i].shape[0]):
      nbr=indices[i][j]
      if nbr != i:
        itm += str(nbr)+','
        weights[nbr]= 1.0/float(distances[i][j]*distances[i][j])
        y_predict += float(weights[nbr]*y_train[nbr])
        sum_weights +=weights[nbr]
    y_predict =round(float(y_predict)/float(sum_weights),2)
    if y_actual >y_predict:
      y_diff =y_actual-y_predict
    else:
      y_diff =y_predict-y_actual
    wght = 1.0/float(1+(y_diff*y_diff))
    itm += '#'+str(y_diff)+':'
    if y_diff*y_diff <=epsilon:
      Gdata.append(itm)
    else:
      Gdata.append(str(i)+'<-')
  y_test = y_test.reshape(y_test.shape[0],1)
  res_er=0
  y_mean=y_test.mean()
  for i in range(y_test.shape[0]):
    res_er += (y_mean-y_test[i][0])
  fp_ca = FootprintCA()
  (ret_score,reachability) =fp_ca.retentionScore(Gdata)
  fp = fp_ca.footprint(ret_score,reachability)
  avg_fp_rs += len(fp)
  cov=coverage(Gdata,fp)
  avg_cov_rs +=cov
  sanity=sanityRate(Gdata,fp)
  avg_sanity_rs +=sanity
  fp_comp =[i for i in range(X_train.shape[0]) if i not in fp]
  fp.sort()
  mod_x_train = X_train[fp,:]
  mod_y_train = y_train[fp].reshape(len(fp),1)
  knn = neighbors.KNeighborsRegressor(k, weights='distance')
  knn.fit(mod_x_train, mod_y_train)
#  y_predict = knn.predict(X_test)
  rss1 =0
  for i in range(y_test.shape[0]):
    X= [list(X_test[i,:])]
    dists =knn.kneighbors(X)[0][0]
    nbrs =knn.kneighbors(X)[1][0]
    yhat=0
    sum_weights=0
    for j in range(len(dists)):
      wgt=1.0/float(dists[j]*dists[j])
      yhat += float(wgt*mod_y_train[nbrs[j]])
      sum_weights += wgt
    yhat =round(float(yhat)/float(sum_weights),2)
    rss1 += np.power(yhat - y_test[i][0],2)
#      print i,y_predict[i],y_test[i][0],yhat
  """  
  rss1 =0
  for i in range(y_test.shape[0]):
    rss1 += np.power(y_predict[i][0] - y_test[i][0],2)
  """
  fpca_rss= float(rss1)/float(y_test.shape[0])
  fpca_size= len(fp)
  
  fp_ca = FootprintCA()
#  (ret_score,reachability) =fp_ca.retentionScore(Gdata)
  (ret_score,reachability) =fp_ca.retentionScoreWeighted(Gdata)
  fp = fp_ca.footprint(ret_score,reachability)
  avg_fp_wrs += len(fp)
  avg_cov_wrs +=coverage(Gdata,fp)
  avg_sanity_wrs +=sanityRate(Gdata,fp)
  fp_comp =[i for i in range(X_train.shape[0]) if i not in fp]
  fp.sort()
  mod_x_train = X_train[fp,:]
  mod_y_train = y_train[fp].reshape(len(fp),1)
  knn = neighbors.KNeighborsRegressor(k, weights='distance')
  knn.fit(mod_x_train, mod_y_train)
 # y_predict = knn.predict(X_test)
  rss1 =0
  for i in range(y_test.shape[0]):
    X= [list(X_test[i,:])]
    dists =knn.kneighbors(X)[0][0]
    nbrs =knn.kneighbors(X)[1][0]
    yhat=0
    sum_weights=0
    for j in range(len(dists)):
      wgt=1.0/float(dists[j]*dists[j])
      yhat += float(wgt*mod_y_train[nbrs[j]])
      sum_weights += wgt
    yhat =round(float(yhat)/float(sum_weights),2)
    rss1 += np.power(yhat - y_test[i][0],2)
    #print i,y_predict[i],y_test[i][0],yhat
  """
  rss1 =0
  for i in range(y_test.shape[0]):
    rss1 += np.power(y_predict[i][0] - y_test[i][0],2)
  """
  fpca_rss1= float(rss1)/float(y_test.shape[0])
  fpca_size1= len(fp)

  
  fp_or = Footprint()
  fp = fp_or.footprintOr(Gdata)
  avg_fp_rc += len(fp)
  avg_cov_rc +=coverage(Gdata,fp)
  avg_sanity_rc +=sanityRate(Gdata,fp)
  fp_comp =[i for i in range(X_train.shape[0]) if i not in fp]
  fp.sort()
  mod_x_train = X_train[fp,:]
  mod_y_train = y_train[fp].reshape(len(fp),1)
  knn = neighbors.KNeighborsRegressor(k, weights='distance')
  knn.fit(mod_x_train, mod_y_train)
  #y_predict = knn.predict(X_test)
  rss1 =0
  for i in range(y_test.shape[0]):
    X= [list(X_test[i,:])]
    dists =knn.kneighbors(X)[0][0]
    nbrs =knn.kneighbors(X)[1][0]
    yhat=0
    sum_weights=0
    for j in range(len(dists)):
      wgt=1.0/float(dists[j]*dists[j])
      yhat += float(wgt*mod_y_train[nbrs[j]])
      sum_weights += wgt
    yhat =round(float(yhat)/float(sum_weights),2)
    rss1 += np.power(yhat - y_test[i][0],2)
 #   print i,y_predict[i],y_test[i][0],yhat
  """
  rss1 =0
  for i in range(y_test.shape[0]):
    rss1 += np.power(y_predict[i][0] - y_test[i][0],2)
  """
  fpor_rss= float(rss1)/float(y_test.shape[0])
  fpor_size =len(fp)
  r1= 1- (fpca_rss/float(res_er))
  r2= 1- (fpca_rss1/float(res_er))
  r3= 1- (fpor_rss/float(res_er))
 # print 'i=',itr,'fpca MSE=',fpca_rss,fpca_rss1,'fpor MSE=',fpor_rss
  avg_fpca_rss +=fpca_rss
  avg_fpca1_rss +=fpca_rss1
  avg_fpor_rss +=fpor_rss

 avg_fp_rs =float(avg_fp_rs)/5.0
 avg_fp_wrs =float(avg_fp_wrs)/5.0
 avg_fp_rc =float(avg_fp_rc)/5.0
 avg_cov_rs =float(avg_cov_rs)/5.0
 avg_cov_wrs =float(avg_cov_wrs)/5.0
 avg_cov_rc =float(avg_cov_rc)/5.0
 avg_sanity_rs =float(avg_sanity_rs)/5.0
 avg_sanity_wrs =float(avg_sanity_wrs)/5.0
 avg_sanity_rc =float(avg_sanity_rc)/5.0
 avg_fpca_rss =float(avg_fpca_rss)/5.0
 avg_fpca1_rss =float(avg_fpca1_rss)/5.0
 avg_fpor_rss =float(avg_fpor_rss)/5.0
#print 'FPCA :', avg_fpca_rss, 'FPOR :',avg_fpor_rss
 print epsilon,k,'\t',avg_fp_rs,'\t',avg_fp_wrs,'\t',avg_fp_rc,'\t',avg_cov_rs,'\t',avg_cov_wrs,'\t',avg_cov_rc,'\t',avg_sanity_rs,'\t',avg_sanity_wrs,'\t',avg_sanity_rc,'\t',avg_fpca_rss,'\t',avg_fpca1_rss,'\t',avg_fpor_rss
 epsilon +=10
#print k, avg_fp_rs,avg_fp_wrs,avg_fp_rc
#print k, avg_cov_rs,avg_cov_wrs,avg_cov_rc
#print k, avg_sanity_rs,avg_sanity_wrs,avg_sanity_rc
#print best_rss,best_epsilon
