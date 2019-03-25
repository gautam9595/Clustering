import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import davies_bouldin_score

iris=load_iris()
score=[]
for i in range(2,int(math.sqrt(len(iris.data)))):
    label=KMeans(n_clusters=i).fit_predict(iris.data)
    score.append(davies_bouldin_score(iris.data,label))
    
print('The Score is ',score)    
plt.plot(range(2,int(math.sqrt(len(iris.data)))),score)
