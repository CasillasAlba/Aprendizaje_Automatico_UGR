# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:59:36 2020

@author: Alba
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import re
import warnings 
warnings.filterwarnings("ignore")


from sklearn.decomposition import PCA


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

#from time import time

# Opciones para ver todas las filas y columnas por salida
pd.set_option('display.max_columns', 5)
pd.set_option('display.max_rows', 10)

def leer_datos(archivo, separador=None):
    #Leemos los datos del archivo y lo guardamos en un Dataframe
    
    # NOTA:
    # Usando: pd.read_csv(archivo) , la primera fila al mostrar los datos se visualiza:
    # 0  1   6  15  12  1.1  0.1  ...  6.3  14.1  7.4  1.3  0.24  0.25  0.26
    # es decir, con números flotantes que no corresponden a los datos. Esto sucede porque con pandas
    # tiene que haber una primera fila que actúa como cabecera
    # He resuelto esta situacion consultando en: https://stackoverflow.com/questions/28382735/python-pandas-does-not-read-the-first-row-of-csv-file
    
    if separador == None:
        datos = pd.read_csv(archivo, header=None)
    else:
        datos = pd.read_csv(archivo, sep=separador , header=None)
    
    return datos


def separar_datos(data):
    #Recoge los valores del dataframe
    valores = data.values
    
    #Todas las columnas menos la ultima
    X = valores[:, :-1]
    #La última columna
    Y = valores[:, -1]
    
    return X,Y



"""

    FUNCIONES PARA REALIZAR CROSS-VALIDATION
    
    
"""

def cross_Validation(models, X, Y):
    #cv = StratifiedKFold(n_splits = 10 , shuffle = True, random_state = 1)
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=1)
    
    means = []
    deviations = []
    
    for model in models:
        scores = cross_val_score(model, X, Y, scoring= 'neg_mean_squared_error', cv=cv)
        
        #Guardamos la media y la esviación de la validación cruzada en los vectores
        means.append(abs(scores.mean()))
        deviations.append(np.std(scores))
        
        
    return means, deviations  
    



"""
    Función que estima el valor de Eout a partir de Etest 
    siguiendo la fórmula explicada en la documentación de la práctica
"""


def cota_Etest(Etest):
    Eout = np.sqrt(1/(2*X_test.size)*np.log(2/0.05))
    return Etest + Eout






"""

    PROBLEMA DE REGRESIÓN
    COMMUNITIES AND CRIMES
  
"""


###########################################################
#                                                         #
#                   MAIN DEL PROGRAMA                     #
#                                                         #
###########################################################


# Comenzamos leyendo los datos del problema

print("Leemos los datos: ")
data = leer_datos('datos/communities.data')

print("Mostramos los datos: ")
print(data.head())
print(data.shape)
print()

input("\n--- Pulsar tecla para continuar ---\n")

"""

    Al leer el archivo communities.names , encontramos una lista con la siguiente estructura:
        @attribute state numeric  --> [@attribute nombre tipo_var]
    es decir, los headers de nuestro conjunto de datos. Como los datos varian mucho entre unas columnas 
    y otras, considero que facilita el entendimiento del problema añadir los headers para ver con qué
    estamos tratando.
    
    Para ello, hago uso de la librería requests, basandome en la información obtenida en:
    https://es.python-requests.org/es/latest/user/quickstart.html
    https://realpython.com/python-requests/
    
    y la librería re basado en expersiones regulares, ya que tiene la función 'findall': 
        findall() module is used to search for “all”  occurrences that match a given pattern. 
        ** \w = letters ( Match alphanumeric character, including "_")
    donde cogeremos de communities.names aquellas sentencias que empiecen por @attributes y le siga texto
    
    He consultado la página:
    https://www.guru99.com/python-regular-expressions-complete-tutorial.html
        

"""

names_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names'
r = requests.get(names_url)
columns = re.findall(r'\@attribute (\w*)', r.text)
del r
data.columns = columns
print("Mostramos los datos con Header incluido ")
print(data.head())
print()
# Attribute Information: (122 predictive, 5 non-predictive, 1 goal) (communities.names)
# Eliminaremos las 5 primeras columnas ya que contienen valores no predictivos



input("\n--- Pulsar tecla para continuar ---\n")


data = data.drop(columns=['state','county','community', 'communityname','fold'], axis=1)

# Eliminamos las filas a las que les falten datos

# Reemplazamos ? por NaN
data = data.replace(to_replace='?', value=np.NaN)

print("Columnas a las que le faltan valores: ")
c_miss_values = data.columns[data.isnull().any()]
print(c_miss_values)
print(c_miss_values.shape)
print()
print(data[c_miss_values[0:13]].describe())
print()
print(data[c_miss_values[13:25]].describe())

print()

input("\n--- Pulsar tecla para continuar ---\n")


print("Eliminamos las filas que no contengan todos los datos: ")
# Eliminamos todas aquellas columnas que contengan valores NaN
data = data.dropna(axis=1)
print(data.head())
print(data.shape)
print()
#print(data.describe())

print("Columnas a las que le faltan valores: ")
c_miss_values = data.columns[data.isnull().any()]
print(c_miss_values)

print()

input("\n--- Pulsar tecla para continuar ---\n")


"""
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
nonViolent = cols[cols[12:17]]
nonViolent_labels = ['burglPerPop', 'larcPerPop', 'autoTheftPerPop','arsonsPerPop', 'nonViolPerPop']
sns.boxplot(data=nonViolent)
ax.set(title="Crimenes no violentos")
ax.set_xticklabels(nonViolent_labels)
plt.show()
"""

X, Y = separar_datos(data)

# Dividir los datos en training y test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=1, shuffle=True)


"""

CÓDIGO CON EL QUE SE MUESTRA LA CORRELACIÓN DE PEARSON
No se ha añadido en la memoria porque debido a la gran cantiddad de valores
es imposible distinguir ni sacar ningun conclusión

cols = data.columns
train_df = pd.DataFrame(columns=cols, data=np.c_[X_train, Y_train])
#corr = train_df.corr()
#print(corr.style.background_gradient(cmap='Spectral'))
plt.figure(figsize=(100,100))
sns.heatmap(train_df.corr(), vmin=-1, cmap='YlGnBu', annot=True)
plt.title("Correlación de Pearson")
plt.show()
"""


print("Preprocesando datos...")
print("Aplicamos PCA...")
# Técnica PCA para disminuir la dimensaionalidad
pca = PCA(0.98)
print("Dimensiones antes de aplicar PCA: ", X_train.shape[1])
pca.fit(X_train)
X_train = pca.transform(X_train)
print("Dimensiones depsués de aplicar PCA: ", X_train.shape[1])
X_test = pca.transform(X_test)


input("\n--- Pulsar tecla para continuar ---\n")


# Vcetor con los nombres de los modelos que vamos a utilizar
models = []
model_names = ['LR' , 'LR Ridge' , 'LR Lasso' , 'SVR' , 'SGDR l2' ,
               'SGDR l1', 'RFR']

#Añadimos los modelos a los que vamos a aplicarle la validación cruzada

models.append(LinearRegression())
models.append(Ridge())
models.append(Lasso())
models.append(SVR())
models.append(SGDRegressor(penalty='l1'))
models.append(SGDRegressor(penalty='l2'))
models.append(RandomForestRegressor(n_estimators=100, random_state=1))

# Aplicamos Validación cruzada
means, deviations = cross_Validation(models, X_train, Y_train)
    


print("Mostramos los resultados tras aplicar cross validation: ")

out_df = pd.DataFrame(index=model_names, columns=['Medias' , 'Desviación'],
                      data=[[mean,dev] for mean, dev in zip(means, deviations)])
print(out_df)


input("\n--- Pulsar tecla para continuar ---\n")

print(" A continuación se mostrarán los datos de Etest, Eout y Matriz de resultados para todos los modelos entrenados [necesario para la documentacion]")


input("\n--- Pulsar tecla para continuar ---\n")


print()
print(" REGRESIÓN LINEAL SIN REGULARIZACIÓN")
logistic = LinearRegression().fit(X_train, Y_train)
predicted = logistic.predict(X_test)
print()
print("- Etest: ", mean_squared_error(Y_test, predicted))
print("- Estimacion del out: ", cota_Etest(mean_squared_error(Y_test, logistic.predict(X_test))))

print("- Coeficiente de estimación ", r2_score(Y_test, predicted))



print("---------------------------------------------------")


print()
print(" REGRESIÓN LINEAL + RIGDE")
logistic = Ridge().fit(X_train, Y_train)
predicted = logistic.predict(X_test)
print()
print("- Etest: ", mean_squared_error(Y_test, predicted))
print("- Estimacion del out: ", cota_Etest(mean_squared_error(Y_test, logistic.predict(X_test))))



print("- Coeficiente de estimación ", r2_score(Y_test, predicted))

print("---------------------------------------------------")


print()
print(" REGRESIÓN LINEAL + LASSO")
logistic = Lasso().fit(X_train, Y_train)
predicted = logistic.predict(X_test)
#result = classification_report(Y_test, predicted)
print()
print("- Etest: ", mean_squared_error(Y_test, predicted))
print("- Estimacion del out: ", cota_Etest(mean_squared_error(Y_test, logistic.predict(X_test))))

print("- Coeficiente de estimación ", r2_score(Y_test, predicted))


print("---------------------------------------------------")



print()
print(" SVR ")
logistic = SVR().fit(X_train, Y_train)
predicted = logistic.predict(X_test)
#result = classification_report(Y_test, predicted)
print()
print("- Etest: ", mean_squared_error(Y_test, predicted))
print("- Estimacion del out: ", cota_Etest(mean_squared_error(Y_test, logistic.predict(X_test))))

print("- Coeficiente de estimación ", r2_score(Y_test, predicted))


print("---------------------------------------------------")


print()
print(" SGD REGRESSOR + RIGDE")
logistic = SGDRegressor(penalty='l2').fit(X_train, Y_train)
predicted = logistic.predict(X_test)
#result = classification_report(Y_test, predicted)
print()
print("- Etest: ", mean_squared_error(Y_test, predicted))
print("- Estimacion del out: ", cota_Etest(mean_squared_error(Y_test, logistic.predict(X_test))))

print("- Coeficiente de estimación ", r2_score(Y_test, predicted))


print("---------------------------------------------------")


print()
print(" SGD REGRESSOR + LASSO ")
logistic = SGDRegressor(penalty='l1').fit(X_train, Y_train)
predicted = logistic.predict(X_test)
#result = classification_report(Y_test, predicted)
print()
print("- Etest: ", mean_squared_error(Y_test, predicted))
print("- Estimacion del out: ", cota_Etest(mean_squared_error(Y_test, logistic.predict(X_test))))

print("- Coeficiente de estimación ", r2_score(Y_test, predicted))


print("---------------------------------------------------")


print()
print(" RANDOM FOREST REGRESSOR")
logistic = RandomForestRegressor(n_estimators=100, random_state=1).fit(X_train,Y_train)
predicted = logistic.predict(X_test)
#result = classification_report(Y_test, predicted)
print()

print("- Etest: ", mean_squared_error(Y_test, predicted))
print("- Estimacion del out: ", cota_Etest(mean_squared_error(Y_test, logistic.predict(X_test))))

print("- Coeficiente de estimación ", r2_score(Y_test, predicted))



input("\n--- Pulsar tecla para continuar ---\n")

# Resultados mejor modelo

print("Tras ajustar nuestro modelo elegido....")

print()
print(" REGRESIÓN LINEAL + RIGDE")
logistic = Ridge().fit(X_train, Y_train)
predicted = logistic.predict(X_test)
print()
print("- Etest: ", mean_squared_error(Y_test, predicted))
print("- Estimacion del out: ", cota_Etest(mean_squared_error(Y_test, logistic.predict(X_test))))

print("- Coeficiente de estimación ", r2_score(Y_test, predicted))

print("Gráfica comparativa")
sns.scatterplot(x=Y_test, y=predicted)
plt.xlabel("Valores reales")
plt.ylabel("Valores predichos")
plt.title(" Valores reales vs Valores predichos")
plt.show()
