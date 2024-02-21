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


# fetch dataset 
cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
  
# data (as pandas dataframes) 
x = cdc_diabetes_health_indicators.data.features 
y = cdc_diabetes_health_indicators.data.targets 

# método Elbow
features = cdc_diabetes_health_indicators.loc[:, 0:7] 
target = cdc_diabetes_health_indicators.loc[:, -1]

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10))

visualizer.fit(features)    # Fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data

def load_csv("Users\labou\OneDrive\Documentos\UFPE\CODING\Projeto SI"):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
                dataset.append(row)
            return dataset

#CÁLCULO DO KMEANS

def euclidean_distance(row1, row2):
 distance = 0.0
 for i in range(len(row1)-1):
    distance += (row1[i] - row2[i])**2
    return sqrt(distance)

def get_neighbors(train, test_row, num_neighbors):
  distances = list()
  for train_row in train:
    dist = euclidean_distance(test_row, train_row)
    distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
  for i in range(num_neighbors):
    neighbors.append(distances[i][0])
    return neighbors

neighbors = get_neighbors(dataset, dataset[0], 3)
for neighbor in neighbors:
 print(neighbor)
