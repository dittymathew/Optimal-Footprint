from sklearn import neighbors,datasets,preprocessing
import numpy as np

def diabetes():
  dataset = datasets.load_diabetes()
  data_X, data_Y = dataset.data, dataset.target
  return (data_X,data_Y)

def boston():
  dataset = datasets.load_boston()
  data_X, data_Y = dataset.data, dataset.target
  return (data_X,data_Y)

def auto_mpg():
  f=open('data/auto-mpg.csv','r')
  data = f.readlines()
  data_X= []
  data_Y=[]
  for i in range(1,len(data)):
    line = data[i].rstrip().split('\t')
    temp =[float(x) for x in line[1:]]
    data_X.append(temp)
    data_Y.append(float(line[0]))
  data_X=np.array(data_X)
  data_Y=np.array(data_Y)
  return (data_X,data_Y)

def forestfires():
  f=open('data/forestfires.csv','r')
  data = f.readlines()
  data_X= []
  data_Y=[]
  feats=[0,1,4,5,6,7,8,9,10,11]
  for i in range(1,len(data)):
    line = data[i].rstrip().split('\t')
    temp =[float(line[j]) for j in feats]
    data_X.append(temp)
    data_Y.append(float(line[12]))
  data_X=np.array(data_X)
  data_Y=np.array(data_Y)
  return (data_X,data_Y)

def automobile():
  f=open('data/automobile.csv','r')
  data = f.readlines()
  data_X= []
  data_Y=[]
  feats=[0,9,10,11,12,13,18,19,20,21,22,23]
  for i in range(1,len(data)):
    line = data[i].rstrip().split('\t')
    temp =[float(line[j]) for j in feats]
    data_X.append(temp)
    data_Y.append(float(line[24]))
  data_X=np.array(data_X)
  data_Y=np.array(data_Y)
  return (data_X,data_Y)



def hardware():
  f=open('data/machine.csv','r')
  data = f.readlines()
  data_X= []
  data_Y=[]
  for i in range(1,len(data)):
    line = data[i].rstrip().split('\t')
    temp =[float(x) for x in line[2:9]]
    data_X.append(temp)
    data_Y.append(float(line[9]))
  data_X=np.array(data_X)
  data_Y=np.array(data_Y)
  return (data_X,data_Y)
