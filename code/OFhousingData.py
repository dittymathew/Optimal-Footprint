import numpy as np
from cvxpy import *
import dccp
#from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from sklearn.model_selection import train_test_split,KFold
from footprintCA import *
from optimize import *
import sys
from sklearn import neighbors,datasets,preprocessing
import networkx as nx
from footprint import *

def optimize(CB,fp):
  x=Bool(len(CB))
  exp=[]
  cov_vec=[]
  for j in range(len(CB)):
    exp_vec=[x[i]*w[i,j] for i in range(len(CB))]
    sub_vec=[x[i]*C[i,j] for i in range(len(CB))]
    cov_vec.append(sum(sub_vec))
    exp_val=1-max_elemwise(exp_vec)
    exp.append(exp_val)
  objective =Minimize(sum(exp)+sum_entries(x))
  constraints=[]
  for i in range(len(cov_vec)):
    constraints.append((1<=cov_vec[i]))
  prob = Problem(objective, constraints)
  x.value=fp
  val=objective.value
  print x.value.sum(),val-x.value.sum(),val
#  return x.value
k=1

dataset = datasets.load_diabetes()
data_X, data_Y = dataset.data, dataset.target
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
  if error_percent <=10:
    Gdata[i].append((soln,wght))
#print Gdata
fp_ca = FootprintCA()
(ret_score,reachability) =fp_ca.retentionScore(Gdata)
fp1 = fp_ca.footprint(ret_score,reachability)
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
"""
idleCases=[]
coveringCases=[]
for i in range(len(CB)):
  if C[i,:].sum()>1:
    coveringCases.append(i)
  if len(CB[i])==0:
    idleCases.append(i)
fp_ca = FootprintCA()
(ret_score,reachability) =fp_ca.retentionScoreWeighted(CB)
fp1 = fp_ca.footprint(ret_score,reachability)
(ret_score,reachability) =fp_ca.retentionScore(CB)
fp3 = fp_ca.footprint(ret_score,reachability)
fp_rc= Footprint()
fp2=fp_rc.footprintOr(CB)
for c in idleCases:
  fp1.append(c)
  fp2.append(c)
  fp3.append(c)

#print len(fp1),len(coveringCases)
s_fp1=[0 for i in range(len(CB))]
s_fp2=[0 for i in range(len(CB))]
s_fp3=[0 for i in range(len(CB))]
for itm in fp1:
  s_fp1[itm]=1
for itm in fp2:
  s_fp2[itm]=1
for itm in fp3:
  s_fp3[itm]=1
print 'Weighted RetScore',
optimize(CB,s_fp1)
print 'Relative Coverage',
optimize(CB,s_fp2)
print 'Retention Score',
optimize(CB,s_fp3)
"""
#  candidates=getCandidates(C,k)
alpha=0
x=Bool(len(CB))
#x=[1,0,0,1,0]
v=Variable(len(CB),len(CB))
c=Bool(len(CB),len(CB))
exp=[]
cov_vec=[]
constraints=[]
#constraints=[0<=v,v<=1]
for j in range(len(CB)):
  sub_vec=[]
  exp_vec=[]
  for i in range(len(CB)):
    if alpha!=0:
      constraints.append(v[j,i]<=x[i]*w[i,j])
      constraints.append(v[j,i]<=c[j,i])
    sub_vec.append(x[i]*C[i,j])
  #  exp_vec.append(1-v[j,i])
  exp_val=1-sum_entries(v[j,:])
  if alpha!=0:
    constraints.append(sum_entries(c[j,:])==1)
   # exp_vec.append(x[i]*(w[i,j]-1))
#  exp_vec=[x[i]*w[i,j] for i in range(len(CB))]
#  sub_vec=[x[i]*C[i,j] for i in range(len(CB))]
  cov_vec.append(sum(sub_vec))
  #exp_val=max_elemwise(exp_vec)
  exp.append(exp_val)
for i in range(len(cov_vec)):
  constraints.append((1<=cov_vec[i]))
alpha=0
while alpha<=0:
  print 'alpha=',alpha
  objective =Minimize(alpha*sum(exp)+(1-alpha)*sum_entries(x))
  prob = Problem(objective, constraints)
  prob.solve(solver='GUROBI')
  val=objective.value
  fp_card=x.value.sum()
  loss= (val-(1-alpha)*fp_card)/alpha
  print alpha,'\t',fp_card,'\t',loss,'\t',val
  """
  s_fp4=[0 for i in range(len(CB))]
  for i in range(len(CB)):
    if x.value[i]==1:
      s_fp4[i]=1
  print 'Optimal loss'
  optimize(CB,s_fp4)
  """
  print
  alpha +=0.1


#for i in range(20):
"""
  opt_val='Inf'
#  subsets=subsetOfSize(len(CB),k)
  m=k
  flag=1
  candidates=[]
  while flag==1:
    flag=0
    print m,len(candidates)
    for subset in itertools.combinations(coveringCases, m):
      print len(candidates),
      s1=[0 for i in range(len(CB))]
      for v in subset:
        s1[v]=1
      x.value=s1
      if check_coverage(cov_vec) ==True:
        candidates.append(s1)
        flag=1
    m =m-1
  for subset in candidates:
    x.value=subset
    prob.solve(method="dccp")
    val=objective.value
    print x.value,val
    if val<opt_val:
      opt_val=val
      opt_x=x.value
  print opt_x
  print opt_val
  return (opt_x,opt_val)
  """
