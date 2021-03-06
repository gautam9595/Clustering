import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import skfuzzy as fuzz

# Loading DataSet
from sklearn import datasets
iris = datasets.load_iris()
x = pd.DataFrame(iris.data, columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
y = pd.DataFrame(iris.target, columns=['Target'])

#Scaling Down
scaler = StandardScaler()
X_std = scaler.fit_transform(x)

#Creating subplots
fig1,axes1 = plt.subplots(3, 3, figsize=(8, 8))

#Stacking Data
alldata = np.vstack((a['component_1'], a['component_2']))
fpcs = []

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen'] 

for ncenters, ax in enumerate(axes1.reshape(-1), 2):
    model= fuzz.cluster.cmeans(alldata, ncenters, 2, error=0.005, maxiter=1000)

    # Store final fuzzy partition Co-efficient values for later plots
    fpcs.append(model[6])

    # Ploting assigned clusters for each data point in training set
    #u--> final fuzzy c partitioned Matrix
    cluster_membership = np.argmax(model[1], axis=0)
    
    for j in range(ncenters):
        ax.plot(a['component_1'][cluster_membership == j],a['component_2'][cluster_membership == j], '.', color=colors[j])

    # Marking the center of each fuzzy cluster
    for pt in cntr:
        ax.plot(pt[0], pt[1],'rs')

    ax.set_title('Centers = {0}; FPC = {1:.2f}'.format(ncenters, model[6]))
    ax.axis('off')
    
fig1.tight_layout()
