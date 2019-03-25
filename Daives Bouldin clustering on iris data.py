import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import davies_bouldin_score

#Loading Dataset
iris=load_iris()
data=pd.DataFrame(iris.data,columns=iris.feature_names)

#Building Model
model=davies_bouldin_score(iris.data,iris.target)

print('The Daives Bouldin score is ',model)
