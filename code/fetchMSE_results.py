import pickle
from footprint import *
from sklearn.model_selection import train_test_split,KFold
from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier
from optimalFootprint import *
from loadDataset import *
import math

def predict_mae(x_train,y_train,X_test,y_test):
  knn = neighbors.KNeighborsRegressor(k, weights='distance')
  knn.fit(x_train, y_train)
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
    ydiff= yhat - y_test[i]
#    if ydiff<0:
#      ydiff=ydiff*-1
    rss1 +=ydiff*ydiff
  mae=float(rss1)/float(y_test.shape[0])
  return mae


dataset='automobile'
ep=5
model=['rc','wrs',0,0.5,1]
(fps,rss)=pickle.load(open(dataset+'/mse/fp_regression_result_'+dataset+'-ep'+str(ep)+'.p'))
print dataset.upper(),ep
#(data_X,data_Y)=auto_mpg()
#(data_X,data_Y)=boston()
#(data_X,data_Y)=diabetes()
(data_X,data_Y)=automobile()
#(data_X,data_Y)=hardware()
data_X= (data_X-data_X.min(axis=0))/(data_X.max(axis=0)-data_X.min(axis=0))
cv = KFold(n_splits=5, shuffle=True,random_state=3)
k=1
itr=0
loss={}
avg_fp_size={}
avg_loss={}
avg_rss={}
avg_rmse={}
for train,test in cv.split(data_X):
  X_train=data_X[train,:]
  y_train=data_Y[train]
  X_test=data_X[test,:]
  y_test=data_Y[test]
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
  
  (W,Cov) =assignWeights(Gdata)
  for i in model:
    if i not in loss:
      loss[i]=[]
      avg_fp_size[i]=0
      avg_loss[i]=0
      avg_rss[i]=0
      avg_rmse[i]=0
#    if i=='rc':
#      fp_or = Footprint()
#      fp = fp_or.footprintOr(Gdata)
#      fps[i][itr]=fp
#      print len(fp)
    l=computeLoss(W,fps[i][itr],Gdata)
    loss[i].append(l)
    print i, len(set(fps[i][itr])),l
    avg_fp_size[i] += float(len(set(fps[i][itr])))/float(X_train.shape[0])*100
    avg_loss[i]+= l
    fp=list(set(fps[i][itr]))
    mod_x_train = X_train[fp,:]
    mod_y_train = y_train[fp].reshape(len(fp),1)
    avg_rss[i] += predict_mae(mod_x_train,mod_y_train, X_test, y_test)#rss[i][itr]
#    avg_rss[i] +=rss[i][itr]
    avg_rmse[i]+= math.sqrt(rss[i][itr])
  itr+=1
for i in model:
  avg_fp_size[i]=float(avg_fp_size[i])/5.0
  avg_loss[i]=float(avg_loss[i])/5.0
  avg_rss[i]=float(avg_rss[i])/5.0
  avg_rmse[i]=float(avg_rmse[i])/5.0
  print i, round(avg_fp_size[i],2), round(avg_loss[i],2),round(avg_rss[i],2)

