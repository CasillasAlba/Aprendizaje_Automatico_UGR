#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 21:25:48 2020

@author: Alba Casillas Rodriguez
"""

import numpy as np
import matplotlib.pyplot as plt

# Fijamos la semilla
np.random.seed(1)

N = 100

# Inicialmente comencé con MAX_ITER = 500 , pero tras observar que no todos los
# vectores convergían, cambié el valor a 2000

MAX_ITER = 2000
DIFERENCIA = 0.01
TASA_APREND = 0.01

def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_recta(intervalo):
    points = np.random.uniform(intervalo[0], intervalo[1], size=(2, 2))
    x1 = points[0,0]
    x2 = points[1,0]
    y1 = points[0,1]
    y2 = points[1,1]
    # y = a*x + b
    a = (y2-y1)/(x2-x1) # Calculo de la pendiente.
    b = y1 - a*x1       # Calculo del termino independiente.
    
    return a, b

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)


# EJERCICIO 2.1: ALGORITMO PERCEPTRON

"""   

    EL PLA terminará cuando prediga bien todos las etiquetas del conjunto de datos
por ello iteramos el bucle y para cada dato calculamos mediante la función signo,
el valor de la etiqueta y comprobamos si coincide con el valor de label o no; si no 
coincide se actualiza w y seguimos en el bucle
De esta manera, si hay un error en la muestra se quedara en bucle infinito hasta que
llegue al maximo de iteraciones (max_iter) y rompa el bucle con el break.

"""

def ajusta_PLA(datos, label, max_iter, vini):
    err = True
    iteracion = 0
    
    w = np.copy(vini)
    
    while err:
        err = False
        
        iteracion = iteracion + 1
        
        for i in range(len(datos)):
            if(signo(np.dot(np.transpose(w),datos[i])) != label[i]):
                w = w + label[i]*datos[i]
                err = True

        if iteracion == max_iter:
            break
    
    
    return w, iteracion



# Generamos la muestra de puntos 2D
X = simula_unif(N,2,[-50,50])

# Simulamos los parámetros v = (a,b) de la recta
a,b = simula_recta([-50,50])

# Obtenemos las etiquetas para cada punto a la recta simulada
Y = []

for coord in X:
    Y.append(f(coord[0], coord[1], a, b))

# Convertimos la lista en un array
Y = np.array(Y)

#Necesitamos añadie una columna de N unos ya que dentro del algoritmo PLA
#necesitaremos multiplicar por la traspuesta de X
X = np.c_[np.ones(N), X]

print("Apartado 2.a.1.a) datos simulados en el vector a 0")
v_ini = np.array([0.0,0.0,0.0])

w, num_iter = ajusta_PLA(X, Y, MAX_ITER, v_ini)

print("PROBAMOS CON MAX_ITER = 500")

print("Apartado 2.a.1.a) vector cero" )

print("w vale: ", w ," para un numero de ", num_iter, " iteraciones")

input("\n--- Pulsar tecla para continuar ---\n")

print("Apartado 2.a.1.b)  vectores de numeros aleatorios en [0, 1] ")
  
# Random initializations
iterations = []
valores_aleatorios = []

for i in range(0,10):
    
    """
    Aprovecho la funcion simula unif para inicializar los valores iniciales del vector
    A esta funcion le hago un "reshape(-1)" para que se generen en una única fila, 
    es decir: 
        simula_unif(3, 1, [0.0, 1.0]) genera, por ejemplo, 
        [[0.90853515]
        [0.62336012]
        [0.01582124]]
        
        simula_unif(3, 1, [0.0, 1.0]).reshape(-1), generará:
        [0.90853515 0.62336012 0.01582124]
    """
    
    v_ini = simula_unif(3, 1, [0.0, 1.0]).reshape(-1)
    valores_aleatorios.append(v_ini)
    w, num_iter = ajusta_PLA(X, Y, MAX_ITER, v_ini)
    iterations.append(num_iter)
    
    print('w_0 = ',v_ini)
    print('Valor w: ' , w, ' Num. iteraciones: ' , num_iter)    
    
 
print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

input("\n--- Pulsar tecla para continuar ---\n")



# Ahora con los datos del ejercicio 1.2.b

print("Apartado 2.a.2.a) Datos simulados en el vector a 0 con los datos del apartado 2b del ejercicio 1 ")

#Le añadimos ruido a la muestra con esta función ya implementada en el ejercicio 1
    
def aniade_ruido(Y,porcentaje=0.1):
    num_positivos = int(np.where(Y == 1)[0].shape[0]*porcentaje)
    num_negativos = int(np.where(Y == -1)[0].shape[0]*porcentaje)
    
    indices_pos = np.random.choice(np.where(Y == 1)[0], size = num_positivos, replace=False)
    indices_pos = np.array(indices_pos)
    indices_neg = np.random.choice(np.where(Y == -1)[0], size = num_negativos, replace=False)
    indices_neg = np.array(indices_neg)
    
    return indices_pos, indices_neg


indices_pos, indices_neg = aniade_ruido(Y)

#Le cambiamos los valores a los indices seleccoinados

Y[indices_pos] = -1
Y[indices_neg] = 1

v_ini = np.array([0.0,0.0,0.0])

w, num_iter = ajusta_PLA(X, Y, MAX_ITER, v_ini)

print("w vale: ", w ," para un numero de ", num_iter, " iteraciones")

input("\n--- Pulsar tecla para continuar ---\n")

print("Apartado 2.a.2.b)  vectores de numeros aleatorios en [0, 1] ")
  

iterations = []


for i in range(0,10):    
    #De esta manera, los puntos aleatorios seran los mismos a los del apartado anterior
    v_ini = valores_aleatorios[i]
    w, num_iter = ajusta_PLA(X, Y, MAX_ITER, v_ini)
    iterations.append(num_iter)
    
    print('w_0 = ',v_ini)
    print('Valor w: ' , w, ' Num. iteraciones: ' , num_iter)    
    
 
print('Valor medio de iteraciones necesario para converger: {}'.format(np.mean(np.asarray(iterations))))

input("\n--- Pulsar tecla para continuar ---\n")



# EJERCICIO 3: REGRESIÓN LOGÍSTICA CON STOCHASTIC GRADIENT DESCENT

# Esta función calcula el error logístico, siguiendo la fórmula
# indicada en la memoria de la práctica

def calcula_error(data, labels, w):
    N = data.shape[0]
    error = 0.0

    for x, y in zip(data, labels):
        error = error + np.log(1+np.exp(-y * (np.dot(np.transpose(w),x))))
        
    return error / N


# Función gradiente que se ejecutará en el sgdRL, sigue la estructura
# de la fórmula explicada en la memoria
    
def gradiente(x, y, w):
    return (-(y*x)/(1 + np.exp(y*np.dot(np.transpose(w),x))))


"""

Partimos de un vector de pesos a 0, el cual se irá actualizando mientras que
||w(t-1) y w(t)|| < 0.01.

Usaremos un batch de tamaño N=1, siguiendo el pseudocódigo proporcionado en la
asignatura, y se iterarán los datos en un orden aleatorio ya que en cada época
se realizará una permutación de los datos.

w = w - (tasa_aprend * grad)

donde la tasa de aprendizaje = 0.01
y grad es el valor resultado de ejecutar la función gradiente

"""

def sgdRL(datos, label, diferencia, tasa_aprend):
    w = np.array([0.0,0.0,0.0])
    num_elem = datos.shape[0]
    
    #para calcular  ||w(t-1) y w(t)||
    normal = float("inf")
    
    w_actual = np.copy(w)

    while normal > diferencia:
        # Realizo la permutacion en los indices para que se mantenga
        # igual tanto en el conjunto de datos como en el de etiquetas
        indices = np.random.permutation(num_elem)
        datos = datos[indices, :]
        label = label[indices]
        
        w = np.copy(w_actual)
        
        #https://realpython.com/python-zip-function/
        for x, y in zip(datos, label):
            grad = gradiente(x, y, w_actual)
            w_actual = w_actual - (tasa_aprend * grad)

        #función de numpy que calcula la normal
        normal = np.linalg.norm(w-w_actual)
    
    return w_actual

#Geneamos la muestra de los datos y le asignamos la etiqueta 
    
X = simula_unif(100,2,[0.0,2.0])
a,b = simula_recta([0.0,2.0])

Y = []

for coord in X:
    Y.append(f(coord[0], coord[1], a, b))

Y = np.array(Y)

# Añadimos columna de 1s al principio 
X = np.c_[np.ones(100), X]

diferencia = 0.01
tasa_aprend = 0.01
w = sgdRL(X, Y, diferencia, tasa_aprend)


#Mostramos los datos

label1 = []
label2 = []
for i in range(0,len(Y)):
    if Y[i] == 1:
        label1.append(X[i])
    else:
        label2.append(X[i])
        
label1 = np.array(label1)
label2 = np.array(label2)

plt.scatter(label1[:,1],label1[:,2],c='cyan',label="Grupo 1")
plt.scatter(label2[:,1],label2[:,2],c='yellow',label="Grupo -1")
plt.plot([0.0,2.0], [-w[0] / w[2], (-w[0] - 2.0 * w[1]) / w[2]], 'r-', c='magenta', linewidth=3)
plt.axis([0.0,2.0,0.0,2.0])
plt.xlabel('$Eje_x$')
plt.ylabel('$Eje_y$')
plt.title('Grafica de los puntos con etiquetas y su separación')
plt.legend()
plt.show()

print("w = " , w)


Ein = calcula_error(X,Y,w)

print(" El valor del error Ein: ", Ein)

input("\n--- Pulsar tecla para continuar ---\n")


# Usar la muestra de datos etiquetada para encontrar nuestra solución g y estimar Eout
# usando para ello un número suficientemente grande de nuevas muestras (>999).

# Generamos mas de 999 puntos para las nuevas muestras

num_puntos = 1500
X_test = simula_unif(num_puntos,2,[0.0,2.0])

Y_test = []

for coord in X_test:
    Y_test.append(f(coord[0], coord[1], a, b))

Y_test = np.array(Y_test)

# Añadimos columna de 1s al principio 
X_test = np.c_[np.ones(num_puntos), X_test]


label1 = []
label2 = []
for i in range(0,len(Y_test)):
    if Y_test[i] == 1:
        label1.append(X_test[i])
    else:
        label2.append(X_test[i])
        
label1 = np.array(label1)
label2 = np.array(label2)


plt.scatter(label1[:,1],label1[:,2],c='cyan',label="Grupo 1")
plt.scatter(label2[:,1],label2[:,2],c='yellow',label="Grupo -1")
#plt.scatter(X_test[:, 1], X_test[:, 2], c=Y_test)
plt.plot([0.0,2.0], [-w[0] / w[2], (-w[0] - 2.0 * w[1]) / w[2]], 'r-', c='magenta', linewidth=3)
plt.axis([0.0,2.0,0.0,2.0])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Grafica datos agrupados con muestra de 1500 puntos')
plt.legend()
plt.show() 

Eout = calcula_error(X_test,Y_test,w)

print(" El valor del error Eout: ", Eout)


input("\n--- Pulsar tecla para continuar ---\n")
