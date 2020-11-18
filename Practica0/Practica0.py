#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 09:16:21 2020

@author: Alba Casillas Rodirguex. Grupo de practicas 1
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import random
import math as mh


#Informacion obtenida de https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html

#Para leer la base de datos de iris de scikit-learn
#Los conjuntos de datos son siempre clases y caracteristicas, por tanto, "iris" es una matriz
from sklearn import datasets
iris = datasets.load_iris()

#La clase dataset de sklearn tiene dos metodos: 
#target que te devuelve la clase y data que te devuelve las caracteristicas

print("Obtenemos las carcteristicas (datos de entrada X) y la clase(Y)")

#Informacion obtenida de https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html

x = iris.data
y = iris.target

print("Mostramos las caracteristicas")
print(x)

#Obtenemos las dos ultimas columnas de X (las dos ultimas caracteristicas)
#matriz[:rango de filas,:rango de columnas]
#Ejemplo: matriz[:2,:] = coge las 2 primeras filas y todas las columnas
x = x[:,-2:]
print("Mostramos las dos utimas caracteristicas")
print(x)

#Visualizar con scatter plot
"""
De nuestra variable X, la cual esta formada por dos caracteristicas
la dividimos en otras dos variables, donde "otra_x" contiene lo valores de la
primera caracteristica y "otra_y" los valores que toma la segunda caracteristica
(siendo cada caracteristica una columna)

"""

otra_x = x[:,0]
otra_y = x[:,1]
color = [] 

#Segun el valor de Y (el cual puede tomar los valores 0, 1 o 2) le asignamos un color
#La funcion append sirve para añadir un valor al final de un vector

for i in range(len(y)):
    if y[i] == 0:
        color.append('r')
    if y[i] == 1:
        color.append('g')
    if y[i] == 2:
        color.append('b')


"""
Informacion obtenida en:
    https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html?highlight=scatter#matplotlib.pyplot.scatter
    https://es.mathworks.com/help/matlab/ref/scatter.html
"""
#scatter = plt.scatter([1,2,3], [4,5,6], c=[7,2,3])
#plt.legend(*scatter.legend_elements())
#plt.legend(*scatter.legend_elements(), loc="upper left", title="Clases")
#plt.legend(['Clase1','Clase2','Clase3'], loc="upper left", title="Clases")
        
scatter= plt.scatter(otra_x,otra_y,c=color)

#LEYENDA

"""
Informacion obtenida en:
    https://matplotlib.org/3.1.1/gallery/text_labels_and_annotations/custom_legends.html
    https://matplotlib.org/3.1.1/api/markers_api.html#module-matplotlib.markers
    
"""
   
# maker == la forma geometrica de la leyenda
# color == fondo del recuadro
# label == etiqueta
# markerfacecolor == color de la figura geometrica
# markersize == tamaño de la figura geometrica

legend_elements = [Line2D([0], [0], marker='o', color='w', label='Clase 0',
                          markerfacecolor='r', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Clase 1',
                          markerfacecolor='g', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Clase 2',
                          markerfacecolor='b', markersize=10)]
                   
plt.legend(handles=legend_elements, loc="upper left", title="Clases")

#Llamamos a show cada vez que queramos visualizar datos
plt.show()

#Parte 2 - TRAINING
"""
Informacion obtenida

http://exponentis.es/como-dividir-un-conjunto-de-entrenamiento-en-dos-partes-train-test-split
https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html
https://docs.python.org/2/library/random.html

"""

p_test = 0.20

matrix1 = []
matrix2 = []
matrix3 = []
submatrix1 = []
submatrix2 = []
submatrix3 = []

#Como hay que conservar la proporcion de elementos. Divido el conjunto X en los
#tipos de clase Y que hay

for i in range(len(x)):
 if y[i] == 0:
     matrix1.append(x[i])
 if y[i] == 1:
     matrix2.append(x[i])
 if y[i] == 2:
     matrix3.append(x[i])

#Tras la division, eligo aleatoriamente el 20% (test) del training de cada matriz
#y elimino dichos elementos de la matriz correspondiente

for i in range(int(len(matrix1)*p_test)):
    random1 = random.randint(0,len(matrix1)-1)
    random2 = random.randint(0,len(matrix2)-1)
    random3 = random.randint(0,len(matrix3)-1)
    
    submatrix1.append(matrix1[random1])
    matrix1.pop(random1)
    submatrix2.append(matrix2[random2])
    matrix2.pop(random2)
    submatrix3.append(matrix3[random3])
    matrix3.pop(random3)
#Ahora tengo 3 matrices con el 80% de los elementos cada una y 
#3 submatrices con el 20% de los elementos cada una.
#Para obtener el total, sumo las matrices (NO SE SUMAN LOS ELEMENTOS!!!!)
    
matrix_train = matrix1 + matrix2 + matrix3
matrix_test = submatrix1 + submatrix2 + submatrix3

train = matrix_train
test = matrix_test

#print("RESULTADOS")
#print("TRAIN")
#print(train)
#print("TEST")
#print(test)

"""
Tambien podriamos haber resuelto el ejercicio usando la funcion:
    model_selection.train_test_split
donde pasamos como parametro los conjuntos X e Y y el porcentaje de datos
para el train y test

"""
#Parte3

#https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html
#numpy.linspace = Return evenly spaced numbers over a specified interval.
cien_numeros = np.linspace(0,2*mh.pi,100)
print("Mostramos los 100 numeros")
print(cien_numeros)

suma = []
senos = []
cosenos = []

for i in range(len(cien_numeros)):
    senos.append(np.sin(cien_numeros[i]))
    cosenos.append(np.cos(cien_numeros[i]))
    
    suma.append(senos[i] + cosenos[i])
 
#https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html
    
plt.plot(cien_numeros, senos, c='black',ls='--')
plt.plot(cien_numeros, cosenos, c='blue', ls='--')
plt.plot(cien_numeros, suma, c='red', ls='--')

plt.show()
