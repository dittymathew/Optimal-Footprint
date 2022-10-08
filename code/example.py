from footprintCA import *
from footprint import *
from optimalFootprint import *
import itertools
import networkx as nx
from cvxpy import *
#from ncvx import *
import numpy as np
import dccp

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


#Gdata = ['1<-4#0.4:','2<-1#1:3#0.2','3<-2#0.3:4#0.5','4<-5#0.3:1#0.9','5<-4#1']
CB ={0:[([3],0.5)],1:[([0],1),([2],0.2)], 2:[([1],0.3),([3],0.4)],3:[([4],0.3),([0],0.9)],4:[([3],1)]}
#CB ={1:[([4],0.4)],2:[([1],1),([3],0.2)], 3:[([2],0.3),([4],0.5)],4:[([5],0.3),([1],0.9)],5:[([4],0.1)]}
cb_size=len(CB)
(W,Cov) =assignWeights(CB,0.5)
(constraints,expression,fp_var)=addConstraints(CB,W,Cov,-1)
fp=optimize(CB,-1,constraints,expression,fp_var)
print fp
M=[0 for i in range(cb_size*cb_size)]
M=np.reshape(np.matrix(M),(cb_size,cb_size))
w=np.matrix(np.zeros((cb_size,cb_size)))
for i in CB.keys():
  for (soln,wgt) in CB[i]:
    for j in soln:
      w[j,i] = wgt
      M[j,i] = 1
      print i,j,wgt
  w[i,i]=1
  M[i,i]=1

fp_ca = FootprintCA()
(ret_score,reachability) =fp_ca.retentionScore(CB)
fp1 = fp_ca.footprint(ret_score,reachability)
print ret_score
print 'Weighted'
print 'Retention Score',fp1
(ret_score,reachability) =fp_ca.retentionScoreWeighted(CB)
fp1 = fp_ca.footprint(ret_score,reachability)
print ret_score
print 'weight2'

#ret_score =fp_ca.retentionScore1(G)
#print ret_score
print 'Weighted Retention Score',fp1
fp_or = Footprint()
fp = fp_or.footprintOr(CB)
print fp
"""
fp_greedy_size =len(fp1)
greedy_loss=0
for c in CB:
  if c not in fp1:
    greedy_loss += 1#loss(c,CB[c],fp1)
  else:
    max_w=0
    for j in range(len(CB)):
      if w[c-1,j]>max_w and j!=c-1:
        max_w=w[c-1,j]
    greedy_loss += 1- max_w

print 'FP greedy Size:', fp_greedy_size
print 'Greedy Loss:', greedy_loss


 # print i-1,i-1,1
print w
print M

#x =Variable(len(CB))
x=Bool(len(CB))

#exp= [x[i]*(1 - max_entries(w[i,:])) for i in range(0,len(CB))]
#exp= sum([(1 - x[i]*w[i,:]) for i in range(0,len(CB))])
exp=[]
#cov_vec = [sum(x[i]*M[:,i]) for i in range(0,len(CB))]
cov_vec=[]
for j in range(len(CB)):
  exp_vec=[x[i]*w[i,j] for i in range(len(CB))]
  sub_vec=[x[i]*M[i,j] for i in range(len(CB))]
  cov_vec.append(sum(sub_vec))
  exp_val=1-max_elemwise(exp_vec)
  exp.append(exp_val)
print exp

#cov=w*x
objective =Minimize(sum(exp)+sum_entries(x))
#constraints=[sum_entries(x)<=fp_greedy_size, 0<=x,x<=1]
constraints=[sum_entries(x)<=fp_greedy_size]
for i in range(len(cov_vec)):
  constraints.append((1<=cov_vec[i]))
prob = Problem(objective, constraints)

#x.value=[1 for i in range(len(CB))]

#prob.solve(method="dccp")

opt_val='Inf'
S=range(len(CB))
m=len(fp1)
flag=1
candidates=[]
while flag==1:
  flag=0
  print m,len(candidates)
  for subset in itertools.combinations(S, m):
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
#for b in range(1,2**len(CB)):
#  x.value=binary_list(b,len(CB))
  if check_coverage(cov_vec) ==True:
    prob.solve(method="dccp")
    val=objective.value
    print x.value,val
    if val<opt_val:
      opt_val=val
      opt_x=x.value
#print dccp.convexify_obj(objective).value
print opt_x
print opt_val
"""
