# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 06:28:39 2020

@author: ROSA
"""

#Se importan la librerias a utilizar
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
########## PREPARAR LA DATA ##########
#Importamos los datos de la misma librería de scikit-learn
dataset = datasets.load_breast_cancer()
print(dataset)
########## ENTENDIMIENTO DE LA DATA ##########
#Verifico la información contenida en el dataset
#print('Información en el dataset:')
#print(dataset.keys())
#print()
##Verifico las características del dataset
#print('Características del dataset:')
#print(dataset.DESCR)
#Seleccionamos todas las columnas
X = dataset.data
#Defino los datos correspondientes a las etiquetas
y = dataset.target
plt.scatter(X[:,[0]],X[:,[1]])
plt.show
kmedias=KMeans(n_clusters=2,random_state=0).fit(X,y)
########## IMPLEMENTACIÓN DE K VECINOS MÁS CERCANOS ##########
from sklearn.model_selection import train_test_split
#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#Defino el algoritmo a utilizar
from sklearn.neighbors import KNeighborsClassifier
algoritmo = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
#Entreno el modelo
algoritmo.fit(X_train, y_train)
#Realizo una predicción
y_pred = algoritmo.predict(X_test)
#Verifico la matriz de Confusión
from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(y_test, y_pred)
print('Matriz de Confusión:')
print(matriz)
#Calculo la precisión del modelo
from sklearn.metrics import precision_score
precision = precision_score(y_test, y_pred)
print('Precisión del modelo:')
print(precision)


# Creando el k-Means para los 2 grupos encontrados
kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)

# Visualizacion grafica de los clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 50, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'Cluster 2')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of cancer')
plt.legend()
plt.show()

print('grupos en etiqueta:')
print(kmedias.labels_[:10])
print('centroide de la clasificacion:')
print(kmedias.cluster_centers_[:5])

