#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 18:37:22 2020

EJERCICIO 2: Rregresión lineal

@author: Alba Casillas Rodriguez
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.lines import Line2D

print('Ejercicio 1\n')

label5 = 1
label1 = -1
M = 128

# Funcion para leer los datos
def readData(file_x, file_y):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la 1 o la 5
	for i in range(0,datay.size):
		if datay[i] == 5 or datay[i] == 1:
			if datay[i] == 5:
				y.append(label5)
			else:
				y.append(label1)
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y

"""
Función que calcula el error cuadrático.
Nos servirá para calcular Ein y Eout

"""

def Err(x,y,w):
    error = 0.0
    
    for i in range(x.shape[0]):
        error = error + (w.T.dot(x[i]) - y[i])**2
        
    return error/x.shape[0]
    

# Gradiente Descendente Estocastico

"""
Para cada conjunto de datos (x e y), un minibatches estará formado des la posición
 pos*128 hasta pos*128+128 (es decir, de 0-128, de 129-257...)
 
"""

def calcula_minibaches(pos, x, y):    
      minibaches_x = x[pos*128:pos*128+128]
      minibaches_y = y[pos*128:pos*128+128]
      
      return minibaches_x, minibaches_y
 

"""

    ALGORITMO GRADIENTE DESCENDENTE ESTOCÁSTICO
    
    Generamos tantos minibatches como número de minibatches hemos indicado (128).
    Para cada uno, recorreremos sus datos, los cuales usaremos para calcular h_x
    
    Tras esto, deberemos de recorrer las características de cada dato de cada
    minibatche y aplicar la fórmula explicada en la documentación (sacada de los
    apuntes de clase)
    
"""  
   
def sgd(x,y,numero_minibaches,w):

    for i in range(numero_minibaches):
       minibanches_x , minibanches_y = calcula_minibaches(i,x,y)
           
       for j in range(minibanches_x.shape[0]): 
          #Recorrer dentro de cada dato de cada minibache todas sus columnas
          #shape[0] te dice el numero de filas y shape[1] el numero de columnas
          h_x = np.dot(minibanches_x[j], w.T)

          resultado = w.copy()
          suma = 0
          
          for c in range(minibanches_x.shape[1]):
              suma = h_x - minibanches_y[j]
              resultado[c] = resultado[c] - (2/minibanches_x.shape[0])*( minibanches_x[j][c] * suma)
          
          w = resultado.copy()

    return w

# Pseudoinversa	(fórmula explicada en la documentación)
def pseudoinverse(x,y):
    return np.dot(np.dot((np.linalg.pinv(np.dot(x.T,x))),x.T),y)





# Simula datos en un cuadrado [-size,size]x[-size,size]
def simula_unif(N, d, size):
	return np.random.uniform(-size,size,(N,d))


"""
    Le asignamos una etiqueta a cada punto de la muestra generada
    por simula_unif. 
    
"""

def funcion_y(x):
    y = []
    
    for i in range(len(x)):
        y.append(np.sign((x[i][0] - 0.2)**2 + x[i][1]**2 - 0.6))
        
    return y
 
    
"""

 Añadimos ruido a un 10% aleatorio de la muestra.
 Lo haremos cambiando el signo al valor 
 
"""

def aniadir_ruido(y):
    for i in range(int(len(y)*0.10)):
        numero = np.random.randint(0,1000)
        y[numero] = -y[numero]


        
# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy')
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy')

numero_minibaches = int(len(x)/M)
w = np.array([0.0,0.0,0.0])

w = sgd(x,y,numero_minibaches,w)



"""

    CODIGO PARA MOSTRAR LOS RESULTADOS DEL SGD Y PSEUDO INVERSA
    
"""

print("Resultados del error en el algoritmo del SGD")
print("W",w)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

# Separando etiquetas para poder escribir leyenda en el plot
label1 = []
label5 = []
for i in range(0,len(y)):
    if y[i] == 1:
        label5.append(x[i])
    else:
        label1.append(x[i])
label5 = np.array(label5)
label1 = np.array(label1)

# Plot de la separación de datos SGD

plt.scatter(label5[:,1],label5[:,2],c='cyan',label="5")
plt.scatter(label1[:,1],label1[:,2],c='yellow',label="1")
plt.plot([0, 1], [-w[0]/w[2], -w[0]/w[2]-w[1]/w[2]])
plt.xlabel('Intensidad')
plt.ylabel('Simetria')
plt.legend()
plt.title('SGD')
plt.show()


print()
input("Pulse para continuar")
print()

# Aplicación del algoritmo de la pseudoinversa
w = pseudoinverse(x,y)
print("Resultados del error en el algoritmo de pseudoinversa")
print("W",w)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

# Plot de la separación de datos pseudoinversa

plt.scatter(label5[:,1],label5[:,2],c='cyan',label="5")
plt.scatter(label1[:,1],label1[:,2],c='yellow',label="1")
plt.plot([0, 1], [-w[0]/w[2], -w[0]/w[2]-w[1]/w[2]])
plt.xlabel('Intensidad')
plt.ylabel('Simetria')
plt.legend()
plt.title('Pseudoinversa')
plt.show()

print()
print()
print()

input("Pulse para continuar")

print("Apartado 2: ")

print("a: Generar una muestra de entrenamiento de N = 1000 puntos X = [−1, 1] × [−1, 1].")

N = 1000
size = 1
d = 2

x = simula_unif(1000,d,size)
print("Pintamos el mapa de puntos 2D")
plt.scatter(x[:,0],x[:,1])
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Muestra generada con simula_unif")
plt.show()


print("b: Implementamos la funcion para asignar una etiqueta a cada punto de la muestra")

y = funcion_y(x)

print("MAPA DE ETIQUETAS ANTES DE AÑADIR RUIDO")

plt.scatter(x[:,0],x[:,1],c=y)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Dataset definido antes de la introducción del ruido")
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Grupo 1',
                          markerfacecolor='y', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Grupo -1',
                          markerfacecolor='purple', markersize=10)]
                   
plt.legend(handles=legend_elements, loc="upper left", title="Grupos")
plt.show()

print("Le añadimos ruido a un 10% de las mismas")

aniadir_ruido(y)

print()
print("MAPA DE ETIQUETAS DESPUES DE AÑADIR RUIDO")

plt.scatter(x[:,0],x[:,1],c=y)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Dataset definido tras la introducción del ruido")
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Grupo 1',
                          markerfacecolor='y', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Grupo -1',
                          markerfacecolor='purple', markersize=10)]
                   
plt.legend(handles=legend_elements, loc="upper left", title="Grupos")
plt.show()

print("c: APlicamos el algoritmo SGD")

x = np.c_[np.ones(1000), x]
numero_minibaches = int(len(x)/M)
w = np.array([0.0,0.0,0.0])   
 
w = sgd(x,y,numero_minibaches,w)
     
print(w)
print ("Ein: ", Err(x,y,w))

plt.scatter(x[:,1],x[:,2],c=y)
plt.plot([-1, 1], [(-w[0] + w[1])/w[2], (-w[0] - w[1])/w[2]])
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Modelo de regresión lineal")
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Grupo 1',
                          markerfacecolor='y', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Grupo -1',
                          markerfacecolor='purple', markersize=10)]
                   
plt.legend(handles=legend_elements, loc="upper left", title="Grupos")
plt.axis([-1,1,-1,1])
plt.show()


print("d: Aplicamos el algotimo 1000 veces")

contador = 0
valor_errores_in = 0
valor_errores_out = 0

while(contador < 1000):
    contador = contador + 1
    
    #Aplicaremos el SGD con el conjunto de datos de entrenamiento
    #con el que calcularemos el Ein
    
    x = simula_unif(1000,d,size)
    y = funcion_y(x)
    aniadir_ruido(y)
    x = np.c_[np.ones(1000), x]
    
    numero_minibaches = int(len(x)/M)
    w = np.array([0.0,0.0,0.0])
    
    w = sgd(x,y,numero_minibaches,w)
     
    err_in = Err(x,y,w)
    valor_errores_in = valor_errores_in + err_in
    
    #Calculamos otros 1000 puntos a posteriori para el conjunto test
    #y calcular el Eout
    
    x_test = simula_unif(1000,d,size)
    y_test = funcion_y(x_test)
    x_test = np.c_[np.ones(1000), x_test]
    err_out = Err(x_test,y_test,w)
    valor_errores_out = valor_errores_out + err_out
    
    
    
print ('Bondad del resultado para grad. descendente estocastico:\n')
print("Ein medio", valor_errores_in/contador)
print("Eout medio ", valor_errores_out/contador)


print()
print()
input("Pulse para continuar")
print()

print("Segundo experimento")

x = simula_unif(1000,d,size)
y = funcion_y(x)
aniadir_ruido(y)
x = np.c_[np.ones(1000),x]
x = np.c_[x, np.ones(1000)]
x = np.c_[x, np.ones(1000)]
x = np.c_[x, np.ones(1000)]
x[:,3] = x[:,1]*x[:,2]
x[:,4] = x[:,1]**2
x[:,5] = x[:,2]**2


print("c: APlicamos el algoritmo SGD")

w = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
w = sgd(x,y,numero_minibaches,w)

print(w)
print ("Ein: ", Err(x,y,w))

plt.scatter(x[:,1],x[:,2],c=y)
#https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.contour.html
x_range = np.arange(-1,1,0.025)
y_range = np.arange(-1,1,0.025)
valor_x, valor_y = np.meshgrid(x_range,y_range) 
func = w[0] + valor_x*w[1] + valor_y*w[2] + valor_x*valor_y*w[3] + ((valor_x)**2)*w[4] + ((valor_y)**2)*w[5]
plt.contour(valor_x,valor_y,func,[0])
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Modelo de regresión lineal")
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Grupo 1',
                          markerfacecolor='y', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Grupo -1',
                          markerfacecolor='purple', markersize=10)]
                   
plt.legend(handles=legend_elements, loc="upper left", title="Grupos")
plt.axis([-1,1,-1,1])
plt.show()




print("d: Aplicamos el algotimo 1000 veces")

contador = 0
valor_errores_in = 0
valor_errores_out = 0
w = np.array([0.0,0.0,0.0,0.0,0.0,0.0])

errout = 0

while(contador < 1000):
    contador = contador + 1
    
    
    #Aplicaremos el SGD con el conjunto de datos de entrenamiento
    #con el que calcularemos el Ein
    
    x = simula_unif(N,d,size)
    y = funcion_y(x)
    aniadir_ruido(y)
    x = np.c_[np.ones(1000),x]
    x = np.c_[x, np.ones(1000)]
    x = np.c_[x, np.ones(1000)]
    x = np.c_[x, np.ones(1000)]
    x[:,3] = x[:,1]*x[:,2]
    x[:,4] = x[:,1]**2
    x[:,5] = x[:,2]**2
       
  
    w = sgd(x,y,numero_minibaches,w)
     
    err_in = Err(x,y,w)
    valor_errores_in = valor_errores_in + err_in
    
 
    #Calculamos otros 1000 puntos a posteriori para el conjunto test
    #y calcular el Eout
    
    x_test = simula_unif(N,d,size)
    y_test = funcion_y(x_test)
    aniadir_ruido(y_test)
    x_test = np.c_[np.ones(1000),x_test]
    x_test = np.c_[x_test, np.ones(1000)]
    x_test = np.c_[x_test, np.ones(1000)]
    x_test = np.c_[x_test, np.ones(1000)]
    x_test[:,3] = x_test[:,1]*x_test[:,2]
    x_test[:,4] = x_test[:,1]**2
    x_test[:,5] = x_test[:,2]**2

    err_out = Err(x_test,y_test,w)
    valor_errores_out = valor_errores_out + err_out

    
    
    
print ('Bondad del resultado para grad. descendente estocastico:\n')
print("Ein medio", valor_errores_in/contador)
print("Eout medio ", valor_errores_out/contador)
