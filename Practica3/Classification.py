# -*- coding: utf-8 -*-
"""
Created on Mon May 25 12:35:11 2020

@author: Alba
"""


import numpy as np
import pandas as pd
import seaborn as sns
#import mlxtend
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings("ignore")

from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
#from mlxtend.plotting import plot_learning_curves

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

#from time import time

pd.set_option('display.max_columns', 5)
pd.set_option('display.max_rows', 10)


# Función para leer los datos de un archivo
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html

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


# Función que separa la muestra en sus características y etiquetas
    
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
    
    -- HE HECHO UNA PARA CADA MODELO --
    
"""

def cross_Validation_LR(X, Y, cv):
    #Posibles parámetros de C
    c_list = [0.01 , 0.1 , 1.0]
   
    
    means = []
    deviations = []
    
    for c in c_list:
        model = LogisticRegression(multi_class = 'multinomial', C=c, solver='lbfgs', random_state=1)
        scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv)
        
        #Guardamos la media y la esviación de la validación cruzada en los vectores
        means.append(scores.mean())
        deviations.append(np.std(scores))

    return means, deviations  
    
def cross_Validation_PLA(X, Y, cv):
    means = []
    deviations = []
    
    model = Perceptron(penalty='l1', n_jobs=-1, random_state=1)
    scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv)
        
    means.append(scores.mean())
    deviations.append(np.std(scores))
        
    model = Perceptron(penalty='l2', n_jobs=-1, random_state=1)
    scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv)
        
    means.append(scores.mean())
    deviations.append(np.std(scores))
    
    return means, deviations
    
    
def cross_Validation_SGD(X, Y, cv):
    means = []
    deviations = []
    
    model = SGDClassifier(penalty='l1',  n_jobs=-1, loss='squared_hinge' , random_state=1)
    scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv)
        
    means.append(scores.mean())
    deviations.append(np.std(scores))
        
    model = SGDClassifier(penalty='l2', n_jobs=-1,  loss='squared_hinge' , random_state=1)
    scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv)
        
    means.append(scores.mean())
    deviations.append(np.std(scores))
    
    return means, deviations


def cross_Validation_SVMC(X, Y, cv):
    #Posibles valores de C
    c_list = [0.01 , 0.1 , 1.0]
   
    
    means = []
    deviations = []
    
    for c in c_list:
        #model = LogisticRegression(C=c, solver='lbfgs')
        model = LinearSVC(C=c, random_state=1, loss='hinge')
        scores = cross_val_score(model, X, Y, scoring='accuracy', cv=cv)
        
        means.append(scores.mean())
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

    PROBLEMA DE CLASIFICACIÓN
  OPTICAL RECOGNITION OF HANDWRITTEN DIGITS
  
"""


###########################################################
#                                                         #
#                   MAIN DEL PROGRAMA                     #
#                                                         #
###########################################################

# Comenzamos leyendo los datos del problema

print("Leemos los datos: ")
data_train = leer_datos('datos/optdigits.tra')
data_test = leer_datos('datos/optdigits.tes')

print("Mostramos los datos de entrenamiento: ")
print(data_train)



input("\n--- Pulsar tecla para continuar ---\n")



# Eliminamos las filas a las que les falten datos
data_train = data_train.dropna()
data_test = data_test.dropna()

# Separamos la muestra en las características y las etiquetas
X_train, Y_train = separar_datos(data_train)
X_test, Y_test = separar_datos(data_test)



#Eliminamos los datos sin variabilidad
unusuable = VarianceThreshold(0.1)
X_train = unusuable.fit_transform(X_train)

unusuable = VarianceThreshold(0.1)
X_test = unusuable.fit_transform(X_test)

"""
#normalizer = Normalizer().fit(X_train)
normalizer = preprocessing.Normalizer()
X_train = normalizer.transform(X_train)
#normalizer = Normalizer().fit(X_test)
X_test = normalizer.transform(X_test)

"""


print("Preprocesamos los datos...")
print("Aplicando PCA...")
# Aplicamos técnica de PCA para el conjunto de entrenamiento y prueba
pca = PCA(0.95)
print("Dimensiones antes de aplicar PCA: ", X_train.shape[1])
pca.fit(X_train)
X_train = pca.transform(X_train)
print("Dimensiones depsués de aplicar PCA: ", X_train.shape[1])
X_test = pca.transform(X_test)


input("\n--- Pulsar tecla para continuar ---\n")

# Variable Local Outlier Factor con contaminacion del 0.1
LOF = LocalOutlierFactor(contamination=0.1)

# Eliminar los outliers sobre los datos de entrenamiento

LOF.fit(X_train,Y_train)

# Generamos un vector con valores mas cercanos a -1 cuanto más nos se aprpoxima el dato a la media.
outliers = LOF.negative_outlier_factor_

# Eliminamos los datos que consideramos demasiado alejados (menores que -1.5)
X_train = X_train[outliers > -1.5]
Y_train = Y_train[outliers > -1.5]

# Eliminamos los datos no válidos
X_train[np.isnan(X_train).any(axis=1)]
Y_train[np.isnan(X_train).any(axis=1)]



# Aplicamos la Validacion cruzada

# Obtenemos los indices segun el numero de particiones para la cross validation
cv = StratifiedKFold(n_splits = 10 , shuffle = True, random_state = 1)

# Vector de nombres de los modelos a los que aplicarle cross validation
model_names = ['MLR c=0.01' , 'MLR c=0.1' , 'MLR c=1.0' , 'PLA l1' ,
               'PLA l2' , 'SGD l1', 'SGD l2', 'SVMC c=0.01' , 'SVMC c=0.1' , 'SVMC c=1.0' ]

# Llamamos a las funciones de cross validation
#start = time()
means_lr, deviations_lr = cross_Validation_LR(X_train, Y_train, cv)
means_pla , deviations_pla = cross_Validation_PLA(X_train, Y_train, cv)
means_sgd , deviations_sgd = cross_Validation_SGD(X_train, Y_train, cv)
means_svm , deviations_svm = cross_Validation_SVMC(X_train, Y_train, cv)
#end = time() - start
#print("Tiempo usado para la validacion: ", end)

# Juntamos en un único vector todas las medias y todas las desviaciones
means = means_lr + means_pla + means_sgd + means_svm
deviations = deviations_lr + deviations_pla + deviations_sgd + deviations_svm


print("Mostramos los resultados de la validación cruzada: ")
# Mostramos resultados con un DataFrame
out_df = pd.DataFrame(index=model_names, columns=['Medias' , 'Desviación'],
                      data=[[mean,dev] for mean, dev in zip(means, deviations)])
print(out_df)

input("\n--- Pulsar tecla para continuar ---\n")

print(" A continuación se mostrarán los datos de Etest, Eout y Matriz de resultados para todos los modelos entrenados [necesario para la documentacion]")


input("\n--- Pulsar tecla para continuar ---\n")


# Entrenamos modelos para obtener los valores de Eout y Etest con los que 
# generamos las gráficas y tablas de la documentación


print()
print(" REGRESIÓN LOGÍSTICA MULTINOMIAL PARA C=0.01")
logistic = LogisticRegression(multi_class = 'multinomial',C=0.01, random_state=1).fit(X_train, Y_train)
predicted = logistic.predict(X_test)
print()
print("- Etest: ", 1-accuracy_score(Y_test, predicted))
print("- Estimacion del out: ", cota_Etest(1-accuracy_score(Y_test, logistic.predict(X_test))))

result = classification_report(Y_test, predicted)
print('\n- Matriz de resultados:\n',result)


print("---------------------------------------------------")


print()
print(" REGRESIÓN LOGÍSTICA MULTINOMIAL PARA C=0.1")
logistic = LogisticRegression(multi_class = 'multinomial',C=0.1 , random_state=1).fit(X_train, Y_train)
predicted = logistic.predict(X_test)
print()
print("- Etest: ", 1-accuracy_score(Y_test, predicted))
print("- Estimacion del out: ", cota_Etest(1-accuracy_score(Y_test, logistic.predict(X_test))))

result = classification_report(Y_test, predicted)
print('\n- Matriz de resultados:\n',result)


print("---------------------------------------------------")


print()
print(" REGRESIÓN LOGÍSTICA MULTINOMIAL PARA C=1.0")
logistic = LogisticRegression(multi_class = 'multinomial',C=1.0 , random_state=1).fit(X_train, Y_train)
predicted = logistic.predict(X_test)
print()
print("- Etest: ", 1-accuracy_score(Y_test, predicted))
print("- Estimacion del out: ", cota_Etest(1-accuracy_score(Y_test, logistic.predict(X_test))))

result = classification_report(Y_test, predicted)
print('\n- Matriz de resultados:\n',result)


print("---------------------------------------------------")


print()
print(" PERCEPTRON + REGULARIZACIÓN RIGDE")
logistic = Perceptron(penalty='l2', n_jobs=-1, random_state=1).fit(X_train, Y_train)
predicted = logistic.predict(X_test)
print()
print("- Etest: ", 1-accuracy_score(Y_test, predicted))
print("- Estimacion del out: ", cota_Etest(1-accuracy_score(Y_test, logistic.predict(X_test))))

result = classification_report(Y_test, predicted)
print('\n- Matriz de resultados:\n',result)


print("---------------------------------------------------")


print()
print(" PERCEPTRON + REGULARIZACIÓN LASSO")
logistic = Perceptron(penalty='l1', n_jobs=-1 , random_state=1).fit(X_train, Y_train)
predicted = logistic.predict(X_test)
print()
print("- Etest: ", 1-accuracy_score(Y_test, predicted))
print("- Estimacion del out: ", cota_Etest(1-accuracy_score(Y_test, logistic.predict(X_test))))

result = classification_report(Y_test, predicted)
print('\n- Matriz de resultados:\n',result)


print("---------------------------------------------------")


print()
print(" SGD + REGULARIZACIÓN RIGDE")
logistic = SGDClassifier(penalty='l2', loss='squared_hinge', n_jobs=-1 , random_state=1).fit(X_train,Y_train)
predicted = logistic.predict(X_test)
print()
print("- Etest: ", 1-accuracy_score(Y_test, predicted))
print("- Estimacion del out: ", cota_Etest(1-accuracy_score(Y_test, logistic.predict(X_test))))

result = classification_report(Y_test, predicted)
print('\n- Matriz de resultados:\n',result)


print("---------------------------------------------------")


print()
print(" SGD + REGULARIZACIÓN LASSO")
logistic = SGDClassifier(penalty='l1', loss='squared_hinge', n_jobs=-1 , random_state=1).fit(X_train,Y_train)
predicted = logistic.predict(X_test)
print()
print("- Etest: ", 1-accuracy_score(Y_test, predicted))
print("- Estimacion del out: ", cota_Etest(1-accuracy_score(Y_test, logistic.predict(X_test))))

result = classification_report(Y_test, predicted)
print('\n- Matriz de resultados:\n',result)


print("---------------------------------------------------")


print()
print(" SVM con Kernel lineal PARA C=0.01")
logistic = LinearSVC(C=0.01, random_state=1, loss='hinge').fit(X_train, Y_train)
predicted = logistic.predict(X_test)
print()
print("- Etest: ", 1-accuracy_score(Y_test, predicted))
print("- Estimacion del out: ", cota_Etest(1-accuracy_score(Y_test, logistic.predict(X_test))))

result = classification_report(Y_test, predicted)
print('\n- Matriz de resultados:\n',result)


print("---------------------------------------------------")


print()
print(" SVM con Kernel linealL PARA C=0.1")
logistic = LinearSVC(C=0.1, random_state=1, loss='hinge').fit(X_train, Y_train)
predicted = logistic.predict(X_test)
print()
print("- Etest: ", 1-accuracy_score(Y_test, predicted))
print("- Estimacion del out: ", cota_Etest(1-accuracy_score(Y_test, logistic.predict(X_test))))

result = classification_report(Y_test, predicted)
print('\n- Matriz de resultados:\n',result)


print("---------------------------------------------------")


print()
print(" SVM con Kernel linealL PARA C=1.0")
logistic = LinearSVC(C=1.0, random_state=1, loss='hinge').fit(X_train, Y_train)
predicted = logistic.predict(X_test)
print()
print("- Etest: ", 1-accuracy_score(Y_test, predicted))
print("- Estimacion del out: ", cota_Etest(1-accuracy_score(Y_test, logistic.predict(X_test))))

result = classification_report(Y_test, predicted)
print('\n- Matriz de resultados:\n',result)

print()
print()
print()


input("\n--- Pulsar tecla para continuar ---\n")

# Resultados mejor modelo

print("Tras ajustar nuestro modelo elegido....")

print(" REGRESIÓN LOGÍSTICA MULTINOMIAL PARA C=0.01")
logistic = LogisticRegression(multi_class = 'multinomial',C=0.01, random_state=1).fit(X_train, Y_train)
predicted = logistic.predict(X_test)
print()
print("- Etest: ", 1-accuracy_score(Y_test, predicted))
print("- Estimacion del out: ", cota_Etest(1-accuracy_score(Y_test, logistic.predict(X_test))))

result = classification_report(Y_test, predicted)
print('\n- Matriz de resultados:\n',result)


matrix = confusion_matrix(Y_test, predicted)

#Pintamos la curva de aprendizaje
#plot_learning_curves(X_train, Y_train, X_test, Y_test, logistic)
#plt.show()

print('\n- Matriz de confusion:\n')
# Tamaño figura
plt.figure(figsize=(9,9))
# Pintamos la matriz
sns.heatmap(matrix, annot=True, fmt=".0f", linewidths=.5, square=True, cmap=plt.cm.RdPu)
plt.ylabel("Etiqueta actual")
plt.xlabel("Etiqueta predecida")
plt.title("Matriz de confusión")
plt.show()
