# importanto as bibliotecas
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pyclustering.cluster.kmedoids import kmedoids
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from sklearn.metrics import *
from sklearn.model_selection import *
from ucimlrepo import fetch_ucirepo
from sklearn.cluster import KMeans # usado pra agrupar amostras com características semelhantes
from yellowbrick.cluster import KElbowVisualizer
from sklearn import metrics

# importing data and creating the dataframe
# fetch dataset
cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

# data (as pandas dataframes)
features = cdc_diabetes_health_indicators.data.features
targets = cdc_diabetes_health_indicators.data.targets

print(features)
print(targets)

# método Elbow -> para encontrar o número ideal de clusters
# Instanciar o modelo KMeans
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10))

# Fit the data to the visualizer
visualizer.fit(features)

# Draw/show/poof the data
visualizer.poof()

# apply kmeans to the number of clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(features)
cluster_labels = kmeans.fit_predict(features)

kmeans.cluster_centers_

# calculate the silhouette coefficient
silhouette_avg = metrics.silhouette_score(features, cluster_labels)
print ('silhouette coefficient for the above clutering = ', silhouette_avg)

# calculate the purity
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

purity = purity_score(target, cluster_labels)
print ('Purity for the above clutering = ', purity)
