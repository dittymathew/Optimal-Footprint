import networkx as nx
import math
from copy import deepcopy
import operator
import sys

class Footprint:
  def normalize(self,dict1):
    max_val = max(dict1.iteritems(), key=operator.itemgetter(1))[1]
    min_val = min(dict1.iteritems(), key=operator.itemgetter(1))[1]
    norm_dict = {}
    for itm in dict1:
      n_val = 1+float(dict1[itm]-min_val)/float(max_val-min_val)
      norm_dict[itm] = round(n_val,15)
    return norm_dict

  def rankdict(self,dict1):
    values = []
    for itm in dict1 :
      if dict1[itm] not in values:
        values.append(dict1[itm])
    values.sort()
    rankeddict = {itm:values.index(dict1[itm]) for itm in dict1}
    return rankeddict

  def footprintOr(self,Gdata):
    coverage={}
    reachability={}
    rel_cov ={}
    G= nx.DiGraph()
    for w in Gdata:
      wdefs = Gdata[w]
      for wdef in wdefs:
        c=wdef[0][0]
        G.add_edge(c,w)
      if w not in G.nodes():
        G.add_node(w)
    for c in G.nodes():
      coverage[c]=G.successors(c) 
      reachability[c]= G.predecessors(c)
    for c in G.nodes():
      rel_cov[c] =0
      for c1 in coverage[c]:
        rel_cov[c] += float(1)/float(len(reachability[c1]))
    sorted_relcov = sorted(rel_cov.iteritems(), key=operator.itemgetter(1))
    sorted_relcov.reverse()
    FP=[]
    changes =True
    while changes==True:
      changes = False
      for i in range(len(sorted_relcov)):
        c= sorted_relcov[i][0]
        if c not in FP  and list(set(reachability[c])&set(FP))==[]:
          changes = True
          FP.append(c)
    return FP
