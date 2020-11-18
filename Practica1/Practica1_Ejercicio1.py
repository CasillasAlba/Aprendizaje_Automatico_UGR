#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 11:34:17 2020

EJERCICIO 1: BÚSQUEDA ITERATIVA DE ÓPTIMOS

@author: Alba Casillas Rodriguez
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


"""

Funcion a usar:
   1 = funcion dada en el apartado 2 del ejercicio 1
   2 = funcion dada en el apartado 3 del ejercicio 1
   
   Establecemos lo valores iniciales proporcionados en el enunciado para
cada una de las funciones.
   Para la función del aparatdo 2 establecemos por defecto el valor de la tasa
de apredizaje. Para el apartado 3 se establecerá más adelante ya que esta podrá 
tomar dos posibles valores

"""

funcion = 2
MAX_ITERACIONES = 50
valores_funcion = []
numero_iteraciones = []

if funcion == 1: 
    valor_inicial1 = 1
    valor_inicial2 = 1
    tasa_aprendizaje = 0.1
elif funcion == 2:
    valor_inicial1 = 1
    valor_inicial2 = -1
else:
    print("Valor incorrecto")
    

"""

    CONJUNTO DE FUNCIONES PARA DEFINIR LAS FUNCIONES DADAS EN EL ENUNCIADO
                    Y SUS RESPECTIVAS DERIVADAS PARCIALES
                    
"""

#Funciones dadas en el enunciado
   
def funcion_E(u,v):
    if funcion == 1:
        return ((u*(np.e**v)-2*v*(np.e**-u))**2)
    else:
        return (u-2)**2 + 2*((v+2)**2) + 2*np.sin(2*np.pi*u)*np.sin(2*np.pi*v)

#Derivada de la funcion en "u" (ó "x")
def derivada_Eu(u,v):
    if funcion == 1:
        return 2*((np.e**v)*u - 2*v*(np.e**-u))*(2*v*(np.e**-u) + np.e*v)
    else:
        return 4*np.pi*np.sin(2*np.pi*v)*np.cos(2*np.pi*u) + 2*(u-2)

#Derivada de la función en "v" (ó "y")
def derivada_Ev(u,v):
    if funcion == 1:
        return 2*(u*(np.e**v) - 2*(np.e**-u))*(u*(np.e**v) - 2*(np.e**-u)*v)
    else:
        return 4*np.pi*np.sin(2*np.pi*u)*np.cos(2*np.pi*v) + 4*(v+2)

#Función que define el gradiente.
#Devuelve: vector de las derivadas parciales de la función
def gradE(u,v):
    return np.array([derivada_Eu(u,v), derivada_Ev(u,v)])



"""

                        ALGORITMO GRADIENTE DESCENDENTE
    
    **Implemento dos algoritmos de gradiente descendente ya que para el apartado 2
    es necesario usar un valor tope (10⁻14), el cual no es necesario en el apartado 3
    
    Declaramos el punto inicial pasando por argumento los valores iniciales deseados
    
    Iteramos hasta que el algoritmo llegue a un máximo de iteraciones (o bien 
    mientras que el valor de la función sea menor o igual que un tope)
        sumamos +1 al numero de iteraciones
        actualizamos w restándole a su valor anterior el gradiente * tasa_aprendizaje, la
        cual marca la velocidad de descenso del algoritmo
        
        **En la funcion del gradiente para el tercer apartado añadimos a dos vectores
        el valor de la funcion y las iteraciones, respectivamente, para poder mostrarlas
        a posteriori graficamente
        
"""

def gradiente_descendiente(valor_inicial1, valor_inicial2, tasa_aprendizaje):
    w_0 = np.array([valor_inicial1,valor_inicial2])
    contador = 0
    tope = np.float64(10**-14)
    
    while(contador < MAX_ITERACIONES) and (funcion_E(w_0[0], w_0[1]) > tope):
        contador = contador + 1
        
        w_0 = w_0 - tasa_aprendizaje*gradE(w_0[0],w_0[1])
        
        #print(funcion_E(w_0[0], w_0[1]))
        
    print("El numero de iteraciones es: ", contador)
    print("Las coordenadas(u,v) donde hemos encontrado el valor son: ", w_0[0], w_0[1])
    
    return w_0
    
        
def gradiente_descendiente_ej2(valor_inicial1, valor_inicial2, tasa_aprendizaje):
    w_0 = np.array([valor_inicial1,valor_inicial2])
    contador = 0
    minimo = float("inf")
    coordx_minima = 0
    coordy_minima = 0
   
    while contador < MAX_ITERACIONES:
        contador = contador + 1
        
        w_0 = w_0 - tasa_aprendizaje*gradE(w_0[0],w_0[1])
        
        if funcion_E(w_0[0], w_0[1]) < minimo :
            minimo = funcion_E(w_0[0], w_0[1])
            coordx_minima = w_0[0]
            coordy_minima = w_0[1]
            
        valores_funcion.append(funcion_E(w_0[0], w_0[1]))
        numero_iteraciones.append(contador)
        
    print("El valor minimo es: ", minimo)
    print("Las coordenadas(u,v) cuyo valor es minimo son: ", coordx_minima, " y " , coordy_minima)
    
    return w_0
    
"""
                FUNCION CON LA QUE GENERAMOS LAS GRAFICAS
                    
                
    Generamos las graficas haciendo uso de la libreria matplotlib. Como debemos generar
dos graficas con tasas de aprendizaje diferentes, le pasamos este valor por defecto.

"""

#https://matplotlib.org/3.2.1/api/_as_gen/matplotlib.pyplot.scatter.html

def mostrar_grafica(numero_iteraciones,valores_funcion, tasa_aprendizaje):   
    print("GRADIENTE DESDENDIENTE APARTADO 2 CON TASA DE APRENDIZAJE DEL ", tasa_aprendizaje)     
    v =[0,MAX_ITERACIONES,-2.0,3.5]
    plt.plot(numero_iteraciones, valores_funcion )
    plt.xlabel("ITERACIONES")
    plt.ylabel("E(u,v)")
    if tasa_aprendizaje == 0.01:
        titulo = "Gradiente descendiente con tasa de aprendizaje 0.01"
    elif tasa_aprendizaje == 0.1:
        titulo = "Gradiente descendiente con tasa de aprendizaje 0.1"
        
    plt.title(titulo)
    plt.axis(v)

    plt.show()


"""
                        MAIN DEL EJERCICIO 
    
    Dependiendo del valor de la variable "funcion", segun el apartado del ejercicio,
ejecutamos uno de los dos algoritmos implementados del gradiente.

    Cuando funcion = 2, ejecutamos dos veces el algoritmo para probar con los dos valores
de la tasa de aprendizaje.

"""


if funcion == 1:
    print("GRADIENTE DESDENDIENTE APARTADO 1")    
    w_0 = gradiente_descendiente(valor_inicial1, valor_inicial2, tasa_aprendizaje)
       
    
elif funcion == 2:
    print("GRADIENTE DESDENDIENTE APARTADO 2")
    
    input("Funcion con tasa de aprendizaje 0.01")
    tasa_aprendizaje = 0.01  
    w_0 = gradiente_descendiente_ej2(valor_inicial1, valor_inicial2, tasa_aprendizaje)
    mostrar_grafica(numero_iteraciones, valores_funcion, tasa_aprendizaje)
    
    print()
    print()
    
    #"Vacíamos" los vectores usados para las gráficas
    valores_funcion = []
    numero_iteraciones = []
    
    input("Funcion con tasa de aprendizaje 0.01")
    tasa_aprendizaje = 0.1
    w_0 = gradiente_descendiente_ej2(valor_inicial1, valor_inicial2, tasa_aprendizaje)
    mostrar_grafica(numero_iteraciones, valores_funcion, tasa_aprendizaje)

   
        

x = np.linspace(-30, 30, 50)
y = np.linspace(-30, 30, 50)
X, Y = np.meshgrid(x, y)
Z = funcion_E(X, Y) 
fig = plt.figure()
ax = Axes3D(fig)
surf = ax.plot_surface(X, Y, Z, edgecolor='none', rstride=1,
                        cstride=1, cmap='jet')
min_point = np.array([w_0[0],w_0[1]])
min_point_ = min_point[:, np.newaxis]
ax.plot(min_point_[0], min_point_[1], funcion_E(min_point_[0], min_point_[1]), 'r*', markersize=10)
ax.set(title='Ejercicio 1.2. Función sobre la que se calcula el descenso de gradiente')
ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('E(u,v)')

    
    