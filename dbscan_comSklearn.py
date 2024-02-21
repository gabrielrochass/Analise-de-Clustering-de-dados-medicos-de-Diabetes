df = features.iloc[:, [3, 18]]
print(df)

# Executando o DBSCAN
clustering = DBSCAN(eps=1, min_samples=5).fit(df)
labels = clustering.labels_
print(labels)

# Criando uma cópia do DataFrame com os rótulos dos clusters
DBSCAN_dataset = df.copy()
DBSCAN_dataset.loc[:, 'Cluster'] = clustering.labels_

# Visualizando os clusters
plt.figure(figsize=(8, 7))
sns.scatterplot(x=df.iloc[:, 0],
                y=df.iloc[:, 1],
                hue=DBSCAN_dataset['Cluster'],
                palette='viridis')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.title('DBSCAN Clustering')
plt.show()

# Contando os clusters
cluster_counts = DBSCAN_dataset['Cluster'].value_counts().to_frame()
print(cluster_counts)
