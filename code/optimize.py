from footprintCA import *
import networkx as nx
from cvxpy import *
import dccp
#from ncvx import *
import numpy as np
import itertools


def subsetOfSize(N,k):
  S= [i for i in range(0,N)]
  subsets=[]
  sset = list(itertools.combinations(S, k))
  for set1 in sset:
    s1=[0 for i in range(N)]
    for itm in set1:
      s1[itm]=1
    subsets.append(s1)
  return subsets
    

def binary_list(b,size):
  n=int(bin(b)[2:])
  l=[]
  for i in range(size):
    r=n%10
    l.append(r)
    n=n/10
  l.reverse()
  return l

def check_coverage(cov_vec):
  for i in range(len(cov_vec)):
    if cov_vec[i].value<1:
      return False
  return True

def loss(c,solns,FP):
  print FP
  max_w=0
  for (soln,wgt) in solns:
    if set(soln)<set(FP):
      if wgt > max_w:
        max_w= wgt
  loss= 1-max_w
  return loss

def getCandidates(C,k):
  candidates=[]
  N=C.shape[0]
  S=range(N)
  m=k
  while m>0:
    flg=0
    for subset in itertools.combinations(S, m):
      or_=[0 for i in range(N)]
      for itm in subset:
        or_= np.bitwise_or(or_,C[itm,:])
      if or_.sum()==N:
        flg=1
        print subset
        candidates.append(subset)
    if flg==0:
      break
    m =m-1
  print candidates

def optimalFp(CB,k):
  cb_size=len(CB)
  C=[0 for i in range(cb_size*cb_size)]
  C=np.reshape(np.matrix(C),(cb_size,cb_size))
  w=np.matrix(np.zeros((cb_size,cb_size)))
  
  for i in CB.keys():
    for (soln,wgt) in CB[i]:
      for j in soln:
        w[j-1,i-1] = wgt
        C[j-1,i-1] = 1
        print i-1,j-1,wgt
    w[i-1,i-1]=1
    C[i-1,i-1]=1
  w_max=w.max()
  w_min=w.min()
  print w_max,w_min
  for i in range(len(CB)):
    for j in range(len(CB)):
      if i !=j:
        w[i,j]= float(w[i,j]-w_min)/float(w_max-w_min)
  print w.max(),w.min()
  coveringCases=[]
  for i in range(len(CB)):
    if C[i,:].sum()>1:
      coveringCases.append(i)
  fp_ca = FootprintCA()
  (ret_score,reachability) =fp_ca.retentionScore(CB)
  fp1 = fp_ca.footprint(ret_score,reachability)
  print len(fp1),len(coveringCases)
  s_fp=[0 for i in range(len(CB))]
  for itm in fp1:
    s_fp[itm]=1
#  candidates=getCandidates(C,k)
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
#  constraints=[sum_entries(x)<=len(fp1)]
  constraints=[]
  for i in range(len(cov_vec)):
    constraints.append((1<=cov_vec[i]))
  prob = Problem(objective, constraints)
#  x.value=s_fp
#  val=objective.value
#  print val
  prob.solve(method="dccp")
  val=objective.value
  print val
  return (x.value,val)
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
