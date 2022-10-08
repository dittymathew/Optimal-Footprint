import networkx as nx
import math
from copy import deepcopy
import operator
import sys

class FootprintCA:
  def __init__(self):
    self.a=None

  def normalize(self, dict1):
    max_val = max(dict1.iteritems(), key=operator.itemgetter(1))[1]
    min_val = min(dict1.iteritems(), key=operator.itemgetter(1))[1]
    norm_dict = {}
    for itm in dict1:
      n_val = 1+ float(dict1[itm]-min_val)/float(max_val-min_val)
#      norm_dict[itm] = float(format(n_val, '.6f'))
      norm_dict[itm] = round(n_val,10)
    return norm_dict

  def rankdict(self, dict1):
    values = []
    for itm in dict1 :
      if dict1[itm] not in values:
        values.append(dict1[itm])
  
    values.sort()
    rankeddict = {itm:values.index(dict1[itm]) for itm in dict1}
    return rankeddict

  def retentionScore(self,CB):
    solvents=[]
    reachability={}
    loc_cov ={}
    words =[]
    for w in CB:
      w_defs=CB[w]
      if len(w_defs)>0:
        reachability[w] =[]
        words.append(w)
        for wdef in w_defs:
          solvent =[]
          solution =wdef[0]
          for dw in solution:
              solvent.append(dw)
              if dw not in words:
                words.append(dw)
              if dw not in loc_cov:
                loc_cov[dw] =[]
              loc_cov[dw].append(w)
          solvent.sort()
          if solvent != []:
            reachability[w].append([solvent])  
            if solvent not in solvents:
              solvents.append(solvent)
    retentionscore = {}
    for c in words:
      if c in loc_cov:
        cov_cases= loc_cov[c]
      else:
        cov_cases=[]
      score = 0
      for c1 in cov_cases:
        alt_solns=[]
        for solvent in reachability[c1]:
          if c not in solvent:
            alt_solns.append(solvent)
        cov_score = 1.0/float(len(alt_solns)+1)
        support_cases =[]
        for solvent in reachability[c1]:
           solvent =solvent[0]
           if c in solvent:
            score += cov_score/float(len(solvent))
      retentionscore[c] =score
    retentionscore =self.normalize(retentionscore)
    prev_retentionscore =deepcopy(retentionscore)
    inf =float('inf')
    n=0
    k=1
    changes=True
    while changes == True:
      pre_prev_retentionscore =deepcopy(prev_retentionscore)
      prev_retentionscore =deepcopy(retentionscore)
      for c in reachability:
        score =0
        if c in loc_cov:
          for v in loc_cov[c]: # covered cases
            cover_score = prev_retentionscore[v]
            support_cases =[]
            for solvent in reachability[v]:
              solvent =solvent[0]
              if c in solvent:
                support_score=inf
                for case in solvent:
                  sup_c_score= prev_retentionscore[case]
                  if sup_c_score <support_score:
                    support_score = sup_c_score
                score += float(cover_score*support_score)/float(len(solvent))
          retentionscore[c]=score
      retentionscore =self.normalize(retentionscore)
      changes =False
      for v in retentionscore:
        if float(format(retentionscore[v],'.5f')) !=float(format(prev_retentionscore[v],'.5f')):
#          print v, retentionscore[v],prev_retentionscore[v]
          if n !=0 :
            if float(format(pre_prev_retentionscore[v],'.5f')) != float(format(retentionscore[v],'.5f')):
              changes =True
              break
          else:
            changes =True
            break
      k+=1
    #print k
    sorted_retscore = sorted(retentionscore.iteritems(), key=operator.itemgetter(1))
    sorted_retscore.reverse()
    return (sorted_retscore,reachability)

  def retentionScoreWeighted(self,CB):
    solvents=[]
    reachability={}
    loc_cov ={}
    words =[]
    cnt=1
    for w in CB:
      w_defs=CB[w]
      if len(w_defs)>0:
        reachability[w] =[]
        words.append(w)
        for wdef in w_defs:
            solvent =[]
            solution =wdef[0]
            wght = wdef[1]
            for dw in solution :
                solvent.append(dw)
                if dw not in words:
                  words.append(dw)
                if dw not in loc_cov:
                  loc_cov[dw] =[]
                if w not in loc_cov[dw]:
                  loc_cov[dw].append(w)
            solvent.sort()
            if solvent != []:
              reachability[w].append([solvent,wght])  
              if solvent not in solvents:
                solvents.append(solvent)
    retentionscore = {}
    for c in words:
      if c in loc_cov:
        cov_cases= loc_cov[c]
      else:
        cov_cases=[]
      score = 0
      for c1 in cov_cases:
        alt_solns=[]
        for solvent in reachability[c1]:
          if c not in solvent:
            alt_solns.append(solvent)
        cov_score = 1.0/float(len(alt_solns)+1)
        support_cases =[]
        for [solvent,wght] in reachability[c1]:
           if c in solvent:
            score += float(cov_score)*float(wght)/float(len(solvent))
      retentionscore[c] =score
    retentionscore =self.normalize(retentionscore)
    prev_retentionscore =deepcopy(retentionscore)
    inf =float('inf')
    changes=True
    while changes == True :
      pre_prev_retentionscore =deepcopy(prev_retentionscore)
      prev_retentionscore =deepcopy(retentionscore)
      for c in reachability:
        if c in loc_cov:
          cov_cases= loc_cov[c]
        else:
          cov_cases=[]
        score = 0
        for c1 in cov_cases:
          c_score =0
          for [solvent,wght] in reachability[c1]:
            if c in solvent:
              solvent_cost =inf
              for ck in solvent:
                ck_score =prev_retentionscore[ck]
                if ck_score < solvent_cost:
                  solvent_cost = ck_score   
              cov_score = float(prev_retentionscore[c1])*float(wght)
              c_score += float(cov_score)*float(solvent_cost)/float(len(solvent))
          score += c_score
        retentionscore[c] =score
      retentionscore =self.normalize(retentionscore)
      changes =False
      n=0
      for v in retentionscore:
        if float(format(retentionscore[v],'.5f')) !=float(format(prev_retentionscore[v],'.5f')):
#          print v, retentionscore[v],prev_retentionscore[v]
          if n !=0 :
            if float(format(pre_prev_retentionscore[v],'.5f')) != float(format(retentionscore[v],'.5f')):
              changes =True
              break
  
          else:
            changes =True
            break
    sorted_retscore = sorted(retentionscore.iteritems(), key=operator.itemgetter(1))
    sorted_retscore.reverse()
    
    return (sorted_retscore,reachability)

  def footprint(self,sorted_retscore,reachability):
    FP=[]
    changes =True
    while changes==True:
      changes = False
      for i in range(len(sorted_retscore)):
        c= sorted_retscore[i][0]
        flag=0
        if c in reachability:
          for solution in reachability[c]:
            c_solvent =solution[0]
            if (set(c_solvent) <= set(FP)) ==True:
              flag=1
              break
          if flag==0 and c not in FP:
            changes = True
            FP.append(c)
    return FP
