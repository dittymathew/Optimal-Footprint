import pickle
from footprint import *
from optimalFootprint import *
from loadDataset import *
import math
from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier

dataset='automobile'
ep=5
model=['rc','rs','wrs',0,0.2,0.4,0.5,0.6,0.8,1]
FP=pickle.load(open(dataset+'/tradeoff/'+dataset+'_tradeoff_ep_'+str(ep)+'.p'))
print dataset.upper(),ep
(data_X,data_Y)=automobile()
#(data_X,data_Y)=auto_mpg()
#(data_X,data_Y)=boston()
#(data_X,data_Y)=hardware()
data_X= (data_X-data_X.min(axis=0))/(data_X.max(axis=0)-data_X.min(axis=0))
k=1
nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(data_X)
distances, indices = nbrs.kneighbors(data_X)
Gdata= {}
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
  if error_percent <=ep:
    Gdata[i].append((soln,wght))

(W,Cov) =assignWeights(Gdata)
for i in model:
  loss=computeLoss(W,FP[i],Gdata)
  print i, len(set(FP[i])),round(loss,2)

