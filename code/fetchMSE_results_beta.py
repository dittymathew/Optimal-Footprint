import pickle
from footprint import *
from sklearn.model_selection import train_test_split,KFold
from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier
from optimalFootprint import *
from loadDataset import *
import math

dataset='autompg'
ep=5
model=['rc','wrs',0,0.2,0.4,0.5,0.6,0.8,1]
beta=[0.2,0.4,0.6,0.8,1]
(fps,rss)=pickle.load(open(dataset+'/mse/fp_regression_result_'+dataset+'-ep'+str(ep)+'_beta.p'))
print dataset.upper(),ep
(data_X,data_Y)=auto_mpg()
#(data_X,data_Y)=automobile()
#(data_X,data_Y)=boston()
#(data_X,data_Y)=diabetes()
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
  print train.shape,X_train.shape
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
  
  for i in beta:
    (W,Cov) =assignWeightsBeta(Gdata,i)
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
    avg_rss[i] += rss[i][itr]
    avg_rmse[i]+= math.sqrt(rss[i][itr])
  itr+=1
for i in beta:
  avg_fp_size[i]=float(avg_fp_size[i])/5.0
  avg_loss[i]=float(avg_loss[i])/5.0
  avg_rss[i]=float(avg_rss[i])/5.0
  avg_rmse[i]=float(avg_rmse[i])/5.0
  print i, round(avg_fp_size[i],2), round(avg_loss[i],2),round(avg_rss[i],2),round(avg_rmse[i],2)

