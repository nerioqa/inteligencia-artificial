# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:55:51 2020

@author: EPIS
"""

from sklearn import datasets
dataset = datasets.load_breast_cancer()
X = dataset.data
y = dataset.target
########## IMPLEMENTACIÓN DE K VECINOS MÁS CERCANOS ##########
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#Defino el algoritmo a utilizar
from sklearn.neighbors import KNeighborsClassifier
algoritmo = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
#Entreno el modelo
algoritmo.fit(X_train, y_train)
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
# Trazamos los puntos proyectados y muestramos el puntaje de evaluación
import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap='cividis')
plt.show()
