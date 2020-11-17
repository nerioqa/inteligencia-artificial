# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:43:57 2020

@author: EPIS
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X=np.array([[1,1],[1,0],[0,2],[2,4],[3,5]])
plt.scatter(X[:,[0]],X[:,[1]])
plt.show

kmedias=KMeans(n_clusters=2).fit(X)
print('MOSTRANDO ETIQUETAS')
print(kmedias.labels_)

print('CENTROIDES DE LA CLASIFICACION')
print(kmedias.cluster_centers_)
plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], X[:, 1], c=kmedias.labels_, s=50, cmap='viridis')

centros=kmedias.cluster_centers_
plt.scatter(centros[:,0], centros[:,1], c='red', s=100, alpha=0.5)
plt.show()

#print('PREDICIENDO')
#print(kmedias.predict([2,2].reshape(-1)))

x_nuevo=np.array([[2,3]])
print('PREDICCION PARA EL PUNTO 2,3')
print(kmedias.predict(x_nuevo))

