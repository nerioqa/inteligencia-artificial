


from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt

################################       1       ####################################
frutas = pd.read_csv("data_frutas.csv")
frutas.groupby('nombre_fruta').size()
print("Conteo por tipos de fruta:\n",frutas.groupby('nombre_fruta').size())

################################       2       ####################################
X = frutas.drop(['etiqueta','subtipo_fruta','nombre_fruta'],axis=1)
y = np.array(frutas['nombre_fruta'])
plt.scatter(frutas['peso'], frutas['color'],label='data', color='blue')

################################       3       ####################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=6)
vectores = SVC()
vectores.fit(X_train, y_train)
scoreCSV = vectores.score(X_train, y_train)
print("Score del entrenamiento con SVM:", scoreCSV)
logistica = linear_model.LogisticRegression()
logistica.fit(X_train,y_train)
scoreLogistica = logistica.score(X_train, y_train)
print("Score del entrenamiento regresión Logística:", scoreLogistica)

################################       4       ####################################
Xarray = np.array(X).tolist()
X_nuevo = [i for i in Xarray if i[1]==i[2]]

if scoreCSV >= scoreLogistica:
    print(vectores.predict(X_nuevo))
else:
    print(logistica.predict(X_nuevo))

################################       5       ####################################
X_lineal = frutas.get(['ancho','alto'])
Y_lineal = frutas['peso']

regresion_lineal = linear_model.LinearRegression() #por tratarse de una predicción de un valor numerico rela, es regresión lineal múltiple
regresion_lineal.fit(X_lineal, Y_lineal)
print("Predicción para la sandía: ",regresion_lineal.predict([[35,40]]))

