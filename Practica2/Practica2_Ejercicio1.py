#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 09:54:02 2020

@author: Alba Casillas Rodríguez
"""

import numpy as np
import matplotlib.pyplot as plt


# Fijamos la semilla
np.random.seed(1)
N = 50

def simula_unif(N, dim, rango):
	return np.random.uniform(rango[0],rango[1],(N,dim))

def simula_gaus(N, dim, sigma):
    media = 0    
    out = np.zeros((N,dim),np.float64)        
    for i in range(N):
        # Para cada columna dim se emplea un sigma determinado. Es decir, para 
        # la primera columna (eje X) se usará una N(0,sqrt(sigma[0])) 
        # y para la segunda (eje Y) N(0,sqrt(sigma[1]))
        out[i,:] = np.random.normal(loc=media, scale=np.sqrt(sigma), size=dim)
    
    return out


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


# EJERCICIO 1.1: Dibujar una gráfica con la nube de puntos de salida correspondiente
#Generamos N(50) puntos 2D aleatorios con simula unif
x = simula_unif(N, 2, [-50,50])

#Mostramos los puntos:

plt.scatter(x[:,0],x[:,1])
#El simbolo '$' pone el numero de la x como un subindice
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Gráfica de puntos con simula_unif')
plt.show()


#Generamos N(50) punto aleatorios con simula gau
x = simula_gaus(N, 2, np.array([5,7]))

#Mostramos los puntos:

plt.scatter(x[:,0],x[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Gráfica de puntos con simula_gaus')
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


# EJERCICIO 1.2: Dibujar una gráfica con la nube de puntos de salida correspondiente

# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1

def f(x, y, a, b):
	return signo(y - a*x - b)

# Generamos la muestra de puntos 2D
X = simula_unif(100,2,[-50,50])

# Simulamos los parámetros v = (a,b) de la recta
a,b = simula_recta([-50,50])

# Obtenemos las etiquetas para cada punto a la recta simulada
Y = []

for coord in X:
    Y.append(f(coord[0], coord[1], a, b))

# Convertimos la lista en un array
Y = np.array(Y)

# Dibujamos una gráfica donde los puntos muestren el resultado de su etiqueta,
# junto con la recta usada para ello. 

# Creamos la recta, la cual tiene la forma y = a*x+b: 

valores_x = np.linspace(-50, 50, Y.size)
recta_y = (a * valores_x) + b

label1 = []
label2 = []
for i in range(0,len(Y)):
    if Y[i] == 1:
        label1.append(X[i])
    else:
        label2.append(X[i])
        
label1 = np.array(label1)
label2 = np.array(label2)

# Mostramos los datos:

plt.scatter(label1[:,0],label1[:,1],c='cyan',label="Grupo 1")
plt.scatter(label2[:,0],label2[:,1],c='yellow',label="Grupo -1")
plt.plot(valores_x, recta_y, 'r-', c='magenta', linewidth=3)
plt.xlabel('$Eje_x$')
plt.ylabel('$Eje_y$')
plt.title('Grafica de los puntos con etiquetas y su separación')
plt.legend()
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")

# 1.2.b. Dibujar una gráfica donde los puntos muestren el resultado de su etiqueta, junto con la recta usada para ello
# Array con 10% de indices aleatorios para introducir ruido

"""
    Para esta función calculamos el número de elementos, tanto positivos
    como negativos, a los que queremos aplicarle el ruido, y obtenemos
    aleatoriamente las posiciones de los elementos
    
    Se ha consultado las paginas:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.where.html
    https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.choice.html
        
"""

def aniade_ruido(Y,porcentaje=0.1):
    num_positivos = int(np.where(Y == 1)[0].shape[0]*porcentaje)
    num_negativos = int(np.where(Y == -1)[0].shape[0]*porcentaje)
    
    indices_pos = np.random.choice(np.where(Y == 1)[0], size = num_positivos, replace=False)
    indices_pos = np.array(indices_pos)
    indices_neg = np.random.choice(np.where(Y == -1)[0], size = num_negativos, replace=False)
    indices_neg = np.array(indices_neg)
    
    return indices_pos, indices_neg


indices_pos, indices_neg = aniade_ruido(Y)

#Le cambiamos el valor a las etiquetas de los indices seleccionados
Y[indices_pos] = -1
Y[indices_neg] = 1

#Mostramos los datos:

valores_x = np.linspace(-50, 50, len(Y))
recta_y = (a * valores_x) + b

label1 = []
label2 = []
for i in range(0,len(Y)):
    if Y[i] == 1:
        label1.append(X[i])
    else:
        label2.append(X[i])
        
label1 = np.array(label1)
label2 = np.array(label2)

# Plot de la separación de datos SGD

plt.scatter(label1[:,0],label1[:,1],c='cyan',label="Grupo 1")
plt.scatter(label2[:,0],label2[:,1],c='yellow',label="Grupo -1")
plt.plot(valores_x, recta_y, 'r-', c='magenta', linewidth=3)
plt.xlabel('$Eje_x$')
plt.ylabel('$Eje_y$')
plt.title('Grafica de los puntos con etiquetas y su separación')
plt.legend()
plt.show()   
    
input("\n--- Pulsar tecla para continuar ---\n")



# EJERCICIO 1.3: Supongamos ahora que las siguientes funciones definen la frontera de clasificación de los puntos de la muestra en lugar de una recta

def plot_datos_cuad(X, y, fz, title='Point cloud plot', xaxis='x axis', yaxis='y axis'):
    #Preparar datos
    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    border_xy = (max_xy-min_xy)*0.01
    
    #Generar grid de predicciones
    xx, yy = np.mgrid[min_xy[0]-border_xy[0]:max_xy[0]+border_xy[0]+0.001:border_xy[0], 
                      min_xy[1]-border_xy[1]:max_xy[1]+border_xy[1]+0.001:border_xy[1]]
    grid = np.c_[xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()]
    pred_y = fz(grid)
    # pred_y[(pred_y>-1) & (pred_y<1)]
    pred_y = np.clip(pred_y, -1, 1).reshape(xx.shape)
    
    #Plot
    f, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, pred_y, 50, cmap='RdBu',vmin=-1, vmax=1)
    ax_c = f.colorbar(contour)
    ax_c.set_label('$f(x, y)$')
    ax_c.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, linewidth=2, 
                cmap="RdYlBu", edgecolor='white')
    
    XX, YY = np.meshgrid(np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]),np.linspace(round(min(min_xy)), round(max(max_xy)),X.shape[0]))
    positions = np.vstack([XX.ravel(), YY.ravel()])
    ax.contour(XX,YY,fz(positions.T).reshape(X.shape[0],X.shape[0]),[0], colors='black')
    
    ax.set(
       xlim=(min_xy[0]-border_xy[0], max_xy[0]+border_xy[0]), 
       ylim=(min_xy[1]-border_xy[1], max_xy[1]+border_xy[1]),
       xlabel=xaxis, ylabel=yaxis)
    plt.title(title)
    plt.show()
    
    
# Implementamos las 4 funciones dadas en el enunciado


print('GRÁFICA DE LA PRIMERA FUNCION ')
def f_1(X):
    return (X[:, 0] - 10) ** 2 + (X[:, 1] - 20) ** 2 - 400

# Mantenemos X como el conjunto de datos calculado en uno de los anteriores apartados
Y = f_1(X)

indices_pos, indices_neg = aniade_ruido(Y)

Y[indices_pos] = -1
Y[indices_neg] = 1

plot_datos_cuad(X,Y,f_1)
input("\n--- Pulsar tecla para continuar ---\n")



print('GRÁFICA DE LA SEGUNDA FUNCION ')
def f_2(X):
    return 0.5 * (X[:, 0] + 10) ** 2 + (X[:, 1] - 20) ** 2 - 400

Y = f_2(X)

indices_pos, indices_neg = aniade_ruido(Y)

Y[indices_pos] = -1
Y[indices_neg] = 1

plot_datos_cuad(X,Y,f_2)
input("\n--- Pulsar tecla para continuar ---\n")



print('GRÁFICA DE LA TERCERA FUNCION ')
def f_3(X):
    return 0.5 * (X[:, 0] - 10) ** 2 - (X[:, 1] + 20) ** 2 - 400

Y = f_3(X)

indices_pos, indices_neg = aniade_ruido(Y)

Y[indices_pos] = -1
Y[indices_neg] = 1

plot_datos_cuad(X,Y,f_3)
input("\n--- Pulsar tecla para continuar ---\n")



print('GRÁFICA DE LA CUARTA FUNCION ')
def f_4(X):
    return X[:, 1] - (20 * X[:, 0] ** 2)  - 5 * X[:, 0]  + 3

Y = f_4(X)

indices_pos, indices_neg = aniade_ruido(Y)

Y[indices_pos] = -1
Y[indices_neg] = 1

plot_datos_cuad(X,Y,f_4)


input("\n--- Pulsar tecla para continuar ---\n")

