#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 23:30:48 2020

EJERCICIO 2: BONUS --- Método Newton

@author: Alba Casillas Rodriguez
"""

import numpy as np
import matplotlib.pyplot as plt


valor_inicial1 = 1
valor_inicial2 = -1
MAX_ITERACIONES = 50



"""

    CONJUNTO DE FUNCIONES PARA DEFINIR LA FUNCION DADA EN EL ENUNCIADO
       Y SUS RESPECTIVAS DERIVADAS PARCIALES (RIMERAS Y SEGUNDAS)
                    
"""

def funcion_E(u,v):
    return (u-2)**2 + 2*((v+2)**2) + 2*np.sin(2*np.pi*u)*np.sin(2*np.pi*v)

#Derivada de la funcion en "u" (ó "x")
def derivada_Eu(u,v):
    return 4*np.pi*np.sin(2*np.pi*v)*np.cos(2*np.pi*u) + 2*(u-2)

#Derivada de la función en "v" (ó "y")
def derivada_Ev(u,v):
    return 4*np.pi*np.sin(2*np.pi*u)*np.cos(2*np.pi*v) + 4*(v+2)

#Función que define el gradiente.
#Devuelve: vector de las derivadas parciales de la función
def gradE(u,v):
    return np.array([derivada_Eu(u,v), derivada_Ev(u,v)])
        

def derivada2_Eu(u,v):
    return 2 - 8*((np.pi)**2)*np.sin(2*np.pi*v)*np.sin(2*np.pi*u)

def derivada2_Ev(u,v):
    return 4*(1-2*((np.pi)**2)*np.sin(2*np.pi*u)*np.sin(2*np.pi*v))

# Derivada segunda de f respecto a x, y (equivale a derivada segunda resepcto a y, x)
def derivada2_Euv(x, y):
    return 8 * np.pi**2 * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)


"""

    Funcion que define la matriz Hessiana.
    La matriz Hessiana de una función f de n variables,
    es la matriz cuadrada nxn de las segundas derivadas parciales

"""
def hessiana(u,v):
    return np.array([[derivada2_Eu(u,v), derivada2_Euv(u,v)], [derivada2_Euv(u,v),derivada2_Ev(u,v) ]])


"""

    MÉTODO DE NEWTON
    
    Le pasamos como parámetros iniciales los valores (x,y) iniciales y el valor
    de la tasa de aprendizaje (la cual en este problema probarémos con los valores
    0.1 y 0.01)
    
    Tras establecer el w0 (inicial), declararemos también una serie de vectores donde
    guardaremos los valores de w y de la función en w, junto a las iteraciones. 
    Devolveremos estos vectores al final para poder pintar gráficamente la solución.
    
    El algoritmo se ejecutará hasta que cumpla el máximo de iteraciones dada en el
    enunciado (50). Por cada iteración, calculamos la inversa de la matriz
    Hessiana, y calculamos el gradiente de la función; ya que nuestros valores de
    los pesos de w seguirán la fórmula w -= tasa_aprendizaje * H⁻1 * gradiente(f(w)) 
    
    (Toda esta información está sacada de la diapositiva de teoría de clase)

"""

def Metodo_Newton(valor_inicial1, valor_inicial2, tasa_aprendizaje):

    w_0 = np.array([valor_inicial1,valor_inicial2])
    contador = 0

    valores_funcion = []
    valores_w = []
    numero_iteraciones = []
    
   
    while(contador < MAX_ITERACIONES):
        contador = contador + 1
        
        hessian = hessiana(w_0[0],w_0[1])
        hessian = np.linalg.inv(hessian)
        gradient = gradE(w_0[0],w_0[1])    
        
        w_0 = w_0 - tasa_aprendizaje * (hessian.dot(gradient))
        
        valores_funcion.append(funcion_E(w_0[0], w_0[1]))
        valores_w.append(w_0)
        numero_iteraciones.append(contador)
        
    return w_0, valores_funcion, valores_w, numero_iteraciones


"""

    Función para dibujar las gráficas.
    
    Sigue la misma estructura que la función de mismo nombre del ejercicio 1

"""

def mostrar_grafica(numero_iteraciones,valores_funcion, tasa_aprendizaje):   
    print("GRADIENTE DESDENDIENTE APARTADO 2 CON TASA DE APRENDIZAJE DEL ", tasa_aprendizaje)     
    plt.plot(numero_iteraciones, valores_funcion )
    plt.xlabel("ITERACIONES")
    plt.ylabel("E(u,v)")
    if tasa_aprendizaje == 0.01:
        titulo = "Metodo Newton con tasa de aprendizaje 0.01"
    elif tasa_aprendizaje == 0.1:
        titulo = "Metodo Newton con tasa de aprendizaje 0.1"
        
    plt.title(titulo)
    plt.show()
 

"""

    MAIN DEL PROGRAMA
    
    Probamos el algoritmo de Newton para las distintas tasas de aprendizaje
    
"""

input("Pulse para mostrar Metodo de Newton con tasa de 0.01")
tasa_aprendizaje = 0.01  
w, valores_funcion, valores_w, numero_iteraciones = Metodo_Newton(valor_inicial1, valor_inicial2, tasa_aprendizaje)
mostrar_grafica(numero_iteraciones, valores_funcion, tasa_aprendizaje)

#"Vacíamos" los vectores usados para las gráficas
valores_funcion = []
numero_iteraciones = []

input("Pulse para mostrar Metodo de Newton con tasa de 0.1")
tasa_aprendizaje = 0.1
w, valores_funcion, valores_w, numero_iteraciones = Metodo_Newton(valor_inicial1, valor_inicial2, tasa_aprendizaje)
mostrar_grafica(numero_iteraciones, valores_funcion, tasa_aprendizaje)

  

