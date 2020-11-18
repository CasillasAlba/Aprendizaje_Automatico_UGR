#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 05:19:53 2020

@author: Alba Casillas Rodríguez

"""

import numpy as np
import matplotlib.pyplot as plt

# Fijamos la semilla
np.random.seed(1)
max_iter = 2
#BONUS: Clasificación de Dígitos


# La funcion np.sign(0) da 0, lo que nos puede dar problemas
def signo(x):
	if x >= 0:
		return 1
	return -1


"""
Función que calcula el error cuadrático.
Nos servirá para calcular Ein y Eout

"""

def Err(x,y,w):
    error = 0.0
    
    for i in range(x.shape[0]):
        error = error + (w.T.dot(x[i]) - y[i])**2
        
    return error/x.shape[0]


def pseudoinverse(x,y):
    return np.dot(np.dot((np.linalg.pinv(np.dot(x.T,x))),x.T),y)


"""
Función que calcula el error logistico
Nos servirá para calcular Ein y Eout

"""

def error(data, labels, w):
    N = data.shape[0]
    error = 0.0

    for x, y in zip(data, labels):
        error = error + np.log(1+np.exp(-y * (np.dot(np.transpose(w),x))))
        
    return error / N



"""
1. Set the pocket weight vector ŵ to w(0) of PLA
2. for t=1,…,T do
3. Run PLA for one update to obtain w(t+1)
4. Evaluate Ein( w(t+1))
5. If w(t+1) is better than ŵ in terms of Ein( w(t+1)) , set ŵ= w(t+1)
6. Return ŵ

"""

def PLA_pocket(datos, label, max_iter, w):
    w_mejor = np.copy(w)
    err_mejor = error(datos,label,w_mejor)
    

    iteracion = 1
    
    for iteracion in range(max_iter):
        w_antigua = np.copy(w)
        
        for i in range(datos.shape[0]):
            if(signo(np.dot(np.transpose(w),datos[i])) != label[i]):
                w = w + label[i]*datos[i]
                
            err_w = error(datos,label,w)
            
            if err_w < err_mejor:
                w_mejor= np.copy(w)
                err_mejor = err_w
                
        if (w == w_antigua).all():  
            return w_mejor

    return w_mejor
    


# Funcion para leer los datos
def readData(file_x, file_y, digits, labels):
	# Leemos los ficheros	
	datax = np.load(file_x)
	datay = np.load(file_y)
	y = []
	x = []	
	# Solo guardamos los datos cuya clase sea la digits[0] o la digits[1]
	for i in range(0,datay.size):
		if datay[i] == digits[0] or datay[i] == digits[1]:
			if datay[i] == digits[0]:
				y.append(labels[0])
			else:
				y.append(labels[1])
			x.append(np.array([1, datax[i][0], datax[i][1]]))
			
	x = np.array(x, np.float64)
	y = np.array(y, np.float64)
	
	return x, y



# Lectura de los datos de entrenamiento
x, y = readData('datos/X_train.npy', 'datos/y_train.npy', [4,8], [-1,1])
# Lectura de los datos para el test
x_test, y_test = readData('datos/X_test.npy', 'datos/y_test.npy', [4,8], [-1,1])

#Generar grácos separados (en color) de los datos de entrenamiento y test
#junto con la función estimada.


fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(np.squeeze(x_test[np.where(y_test == -1),1]), np.squeeze(x_test[np.where(y_test == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x_test[np.where(y_test == 1),1]), np.squeeze(x_test[np.where(y_test == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TEST)')
ax.set_xlim((0, 1))
plt.legend()
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")

print("Aplicaremos el modelo de Regresión Lineal - Pseudoinversa ")

# Aplicación del algoritmo de la pseudoinversa
w = pseudoinverse(x,y)

print("Resultados del error en el algoritmo de pseudoinversa")
print("W: ", w)
print ('Bondad del resultado para grad. descendente estocastico:\n')
print ("Ein: ", Err(x,y,w))
print ("Eout: ", Err(x_test, y_test, w))

# Generamos los valores de los parametros X e Y para pintar la recta
# de la pseudo inversa que pasará por los datos

pseudo_x = np.linspace(0, 1, y.size)
pseudo_y = (-w[0] - w[1]*pseudo_x) / w[2]

#mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.plot(pseudo_x, pseudo_y, 'k-', c='magenta', linewidth=3, label='PSEUDO-INVERSA')
plt.legend()
plt.ylim(-7,1)
#plt.axis[0.0,10,7.0,1.0]
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

print("Aplicaremos el algoritmo PLA-Pocket")

#Generamos el vector a 0
w_inicial = np.array([0.0,0.0,0.0])

w_pla = PLA_pocket(x,y,max_iter,w_inicial)

print("Resultados del error en el algoritmo de PLA-pokcet")
print("W: ", w_pla)
print ('Bondad del resultado para Pla:\n')
print ("Ein: ", error(x,y,w_pla))
print ("Eout: ", error(x_test, y_test, w_pla))

# Generamos los valores de los parametros X e Y para pintar la recta
# del PLA pocket que pasará por los datos

pla_x = np.linspace(0, 1, y.size)
pla_y = (-w_pla[0] - w_pla[1]*pla_x) / w_pla[2]

#mostramos los datos
fig, ax = plt.subplots()
ax.plot(np.squeeze(x[np.where(y == -1),1]), np.squeeze(x[np.where(y == -1),2]), 'o', color='red', label='4')
ax.plot(np.squeeze(x[np.where(y == 1),1]), np.squeeze(x[np.where(y == 1),2]), 'o', color='blue', label='8')
ax.set(xlabel='Intensidad promedio', ylabel='Simetria', title='Digitos Manuscritos (TRAINING)')
ax.set_xlim((0, 1))
plt.plot(pseudo_x, pseudo_y, 'k-', c='magenta', linewidth=3, label='PLA-POCKET')
plt.legend()
plt.ylim(-7,1)
#plt.axis[0.0,10,7.0,1.0]
plt.show()


input("\n--- Pulsar tecla para continuar ---\n")

print("Obtenemos las cotas sobre el verdadero valor de EOut")

#Calculamos la cota 
Ein = error(x,y,w_pla)
Eout = np.sqrt((8/x.size)*np.log((4*((2*x.size)**3)+1)/0.05))
cota_etest = Ein + Eout


# Con esta funcion calculamos las cotas sobre el verdadero valor de Eout
# Basadas en Ein y Etest, que sera el valor "err" de entrada 

def cota(err, tamanio_muestra, delta):
  """Calcula cota superior de Eout.
  Argumentos posicionales:
  - err: El error estimado,
  - N: El tamaño de la muestra y
  - delta: La tolerancia a error.
  Devuelve:
  - Cota superior de Eout"""
  return err + np.sqrt(1/(2*tamanio_muestra)*(np.log(2/delta))) 

Ein = error(x, y, w_pla)
Etest = error(x_test, y_test, w_pla)
tamanio_muestra_ein = len(x)
tamanio_muestra_etest = len(x_test)
delta = 0.05

Eout_ein =cota(Ein, tamanio_muestra_ein, delta)
Eout_etest = cota(Etest, tamanio_muestra_etest, delta)

print("Cota superior de Eout (con Ein): ", Eout_ein)
print("Cota superior de Eout (con Etest): ", Eout_etest)

