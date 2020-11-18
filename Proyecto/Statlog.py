# -*- coding: utf-8 -*-
"""
PROBLEMA DE CLASIFICACIÓN - STATLOG

Autores: Alba Casillas Rodríguez y Francisco Javier Bolívar Expósito
"""

import warnings
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC


# Establecemos semilla para obtener resultados reproducibles
np.random.seed(500)


# Lectura de un conjunto de muestraas
def read_data(path, delim=' ', dtype=np.int32):
    data_set = np.loadtxt(path, dtype, None, delim)

    x = data_set[:, :-1]
    y = np.ravel(data_set[:, -1:])

    return x, y

def plot_class_distribution(labels, set_title):
    dist = np.array(np.unique(labels, return_counts=True)).T
    # Barplot
    sns.countplot(labels)
    # Legends
    plt.title(set_title + ' Class Distribution')
    plt.ylabel("Number of Samples")
    plt.xlabel("Class");
    # Values
    for i in dist[:, 0]:
        plt.text(i - 1, dist[i - 1, 1], dist[i - 1, 1])

    plt.show()

def plot_matrix_confusion(y_test, predicted):
    matrix = confusion_matrix(y_test, predicted)
    
    # Tamaño figura
    plt.figure(figsize=(9,9))
    # Pintamos la matriz
    sns.heatmap(matrix, annot=True, fmt=".0f", linewidths=.5, square=True, cmap=plt.cm.RdPu)
    plt.ylabel("Etiqueta actual")
    plt.xlabel("Etiqueta predecida")
    plt.title("Matriz de confusión")
    plt.show()
    
    
###########################################################
#                                                         #
#                   MAIN DEL PROGRAMA                     #
#                                                         #
###########################################################

# Comenzamos leyendo los datos del problema

print("Leemos los datos")
x_train, y_train = read_data('./datos/shuttle.trn')
x_test, y_test = read_data('./datos/shuttle.tst')

# Mostramos las primeras filas del dataset e información del mismo:
print()
print(pd.DataFrame(x_train).head().to_string())
print()
print(pd.DataFrame(x_train).info())

# Análisis del problema
# Estadísticas sobre las características
print(pd.DataFrame(x_train).describe().to_string())

input("\n--- Pulsar tecla para continuar ---\n")

# Comprobamos si existen datos perdidos en el dataset
print("¿Existen valores perdidos?: ", end='')
print(pd.DataFrame(np.vstack([x_train, x_test])).isnull().values.any())


input("\n--- Pulsar tecla para continuar ---\n")

# Vemos la distribución de clases tanto en el train set como en el test set
print("Distribución de clases para cada conjunto:")
plot_class_distribution(y_train, 'Training Set')
plot_class_distribution(y_test, 'Test Set')

input("\n--- Pulsar tecla para continuar ---\n")

print("Matriz de correlación entre los atributos:")
matrix_corr = pd.DataFrame(x_train).corr('pearson').round(3)
sns.heatmap(matrix_corr, annot = True)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

# Preprocesamiento de datos
x_train_pol = PolynomialFeatures(include_bias=False).fit_transform(x_train)
x_test_pol = PolynomialFeatures(include_bias=False).fit_transform(x_test)

# Normalización
x_train = StandardScaler(copy=False).fit_transform(x_train)
x_train_pol = StandardScaler(copy=False).fit_transform(x_train_pol)
x_test = StandardScaler(copy=False).fit_transform(x_test)
x_test_pol = StandardScaler(copy=False).fit_transform(x_test_pol)

# Selección de modelo y entrenamiento
# Se eligen los mejores hiperparámetros para los modelos 'LogisticRegression' y
# 'logRegPol' usando validación cruzada 5-fold partiendo el train set

# tras esto se entrena cada modelo usando todo el train set.
parameters_log = [{'penalty': ['none']},
                  {'penalty': ['l1', 'l2'], 'C': np.logspace(-3, 3, 7)}]
parameters_rf = [{'n_estimators': [10, 100, 250, 500],
                  'max_features': ['auto', 'sqrt', 'log2']}]
parameters_svm = [{'C': [0.1 , 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['rbf']},
                  {'C': [0.1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['poly']}]

columns_log = ['mean_fit_time', 'param_C', 'param_penalty', 'mean_test_score',
               'std_test_score', 'rank_test_score']
columns_rf = ['mean_fit_time', 'param_max_features', 'param_n_estimators','mean_test_score',
              'mean_score_time','std_score_time' ]
columns_svm = [ 'mean_fit_time', 'param_C' ,  'param_gamma', 'param_kernel' ,'mean_test_score',
                'std_test_score', 'rank_test_score']                                    


print("Ajustando los modelos ... ")

logReg = GridSearchCV(LogisticRegression(solver='saga', max_iter=1000), parameters_log, n_jobs=-1)
logReg.fit(x_train, y_train)
logRegPol = GridSearchCV(LogisticRegression(solver='saga', max_iter=1000), parameters_log, n_jobs=-1)
logRegPol.fit(x_train_pol, y_train)
randomForest = GridSearchCV(RandomForestClassifier(), parameters_rf, n_jobs=-1)
randomForest.fit(x_train, y_train)
svm = GridSearchCV(SVC(), parameters_svm, n_jobs = -1)
svm.fit(x_train, y_train)


print('CV para RL\n', pd.DataFrame(logReg.cv_results_, columns=columns_log).to_string())
print('CV para RL con combinación no lineal\n',
         pd.DataFrame(logRegPol.cv_results_, columns=columns_log).to_string())
print('CV para RF\n', 
      pd.DataFrame(randomForest.cv_results_, columns=columns_rf).to_string())
print('CV para SVM\n', 
      pd.DataFrame(svm.cv_results_, columns=columns_svm).to_string())

# Se muestran los hiperparámetros escogidos y Eval para ambos modelos
print('\nResultados de selección de hiperparámetros por validación cruzada')
print("LR Best hyperparameters: ", logReg.best_params_)
print("LR CV-Accuracy :", logReg.best_score_)


print("LRP Best hyperparameters: ", logRegPol.best_params_)
print("LRP CV-Accuracy :", logRegPol.best_score_)

print("RF Best hyperparameters : ", randomForest.best_params_)
print("RF CV-Accuracy :", randomForest.best_score_)

print("SVM Best hyperparameters : ", svm.best_params_)
print("SVM CV-Accuracy :", svm.best_score_)

input("\n--- Pulsar tecla para continuar ---\n")

# Predicción con los modelos entrenados del train y test set
print('Métricas de evaluación para los modelos entrenados para train y test')
print('LR Train-Accuracy: ', logReg.score(x_train, y_train))
print('LRP Train-Accuracy: ', logRegPol.score(x_train_pol, y_train))
print('RF Train-Accuracy: ', randomForest.score(x_train, y_train))
print('SVM Train-Accuracy: ', svm.score(x_train, y_train))

print('\nLR Test-Accuracy: ', logReg.score(x_test, y_test))
print('LRP Test-Accuracy: ', logRegPol.score(x_test_pol, y_test))
print('RF Test-Accuracy: ', randomForest.score(x_test, y_test))
print('SVM Test-Accuracy: ', svm.score(x_test, y_test))

input("\n--- Pulsar tecla para continuar ---\n")

print("Matrices de resultados y matriz de confusión para los modelos ajustados: ")
predicted = logReg.predict(x_test)
result = classification_report(y_test, predicted)
print("Matriz de resultados LR:\n", result)
print('\n- Matriz de confusion LR:\n')
plot_matrix_confusion(y_test, predicted)

predicted = logRegPol.predict(x_test_pol)
result = classification_report(y_test, predicted)
print("Matriz de resultados LRP:\n", result)
print('\n- Matriz de confusion LRP:\n')
plot_matrix_confusion(y_test, predicted)

predicted = randomForest.predict(x_test)
result = classification_report(y_test, predicted)
print("Matriz de resultados RF:\n", result)
print('\n- Matriz de confusion RF:\n')
plot_matrix_confusion(y_test, predicted)

predicted = svm.predict(x_test)
result = classification_report(y_test, predicted)
print("Matriz de resultados SVM:\n", result)
print('\n- Matriz de confusion SVM:\n')
plot_matrix_confusion(y_test, predicted)

