import numpy as np
from datetime import datetime
from cvxpy import *

def assignWeights(CB):
  cb_size=len(CB)
  C=[0 for i in range(cb_size*cb_size)]
  C=np.reshape(np.matrix(C),(cb_size,cb_size))
  w=np.matrix(np.zeros((cb_size,cb_size)))
  for i in CB.keys():
    for (soln,wgt) in CB[i]:
      for j in soln:
        w[j,i] = wgt
        C[j,i] = 1
    w[i,i]=1
    C[i,i]=1
  print 'Constructed weight matrix. Current time ',str(datetime.now())
  return (w,C)

def assignWeightsBeta(CB,threshold):
  cb_size=len(CB)
  C=[0 for i in range(cb_size*cb_size)]
  C=np.reshape(np.matrix(C),(cb_size,cb_size))
  w=np.matrix(np.zeros((cb_size,cb_size)))
  for i in CB.keys():
    for (soln,wgt) in CB[i]:
      for j in soln:
        w[j,i] = wgt
        if w[j,i]>=threshold:
          C[j,i] = 1
    w[i,i]=1
    C[i,i]=1
  print 'Constructed weight matrix. Current time ',str(datetime.now())
  return (w,C)

def addConstraints(CB,w,C,alpha):
  x=Bool(len(CB))
  v=Variable(len(CB))
  c=Bool(len(CB),len(CB))
  exp=[]
  cov_vec=[]
  constraints=[]
  for j in range(len(CB)):
    sub_vec=[]
    exp_vec=[]
    for i in range(len(CB)):
      if alpha!=0:
        constraints.append(v[j]>=x[i]*w[i,j])
        constraints.append(v[j]<=x[i]*w[i,j]+1-c[j,i])
      sub_vec.append(x[i]*C[i,j])
    exp_val=1-v[j]
#    constraints.append(exp_val<1)
    if alpha!=0:
      constraints.append(sum_entries(c[j,:])==1)
    cov_vec.append(sum(sub_vec)) #   exp.append(exp_val)
    exp.append(exp_val)
  for i in range(len(cov_vec)):
    constraints.append((1<=cov_vec[i]))
  print 'Added constraints. Current time ',str(datetime.now())
  return (constraints,exp,x)


def optimize(CB,alpha,constraints,exp,x):
  if alpha==-1:
    objective =Minimize(sum(exp)+sum_entries(x))
  else:
    objective =Minimize(alpha*sum(exp)+(1-alpha)*sum_entries(x))
  prob = Problem(objective, constraints)
  prob.solve(solver='GUROBI')
  print 'Problem solved for alpha=',alpha,'. Current time ',str(datetime.now())
  val=objective.value
  fp_card=x.value.sum()
  if alpha==-1:
    loss= val-fp_card
  else:
    loss= (val-(1-alpha)*fp_card)/alpha
  print alpha,'\t',int(fp_card),'\t',round(loss,2),'\t',round(fp_card+loss,2)
  fp=[]
  for i in range(len(CB)):
    if x.value[i]==1:
      fp.append(i)
  return fp

def computeLoss(w,fp,CB):
  loss=0
  for i in range(len(CB)):
    max_wgt=0
    for j in range(len(CB)):
      if j in fp:
        if w[j,i] >max_wgt:
          max_wgt=w[j,i]
    loss+= 1-max_wgt
  return loss
  
