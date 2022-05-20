import matplotlib.pyplot as plt
import numpy as np
from sympy import *
import streamlit as st
from sympy.abc import x


def construirGraficoInterpolado(lista_de_coordenadas):
    idx = len(lista_de_coordenadas) - 1  # Limita o intervalo do input em função do número de coordenadas fornecidos
    # xk = float(input("xk = "))  # Valor de x a ser substitution no polinômio P(x) obtido em result
    index = []  # Recebe os índices como número inteiro a partir da quantidade de coordenadas
    for i in range(idx + 1):
        index.append(i)

    coordinates_ref = index.copy()  # Lista cópia do índice sujeita a alterações no próximo laço for
    g = []  # Armazena os fatores do numerador de cada L calculado
    h = []  # Armazena os fatores do denominador de cada L calculado
    p_list = []  # Armazena os produtos yn * Ln(x)
    lagrange_list = []  # Armazena os valores de Ln(x) calculados para todas as combinações de xi

    for j in range(len(index)):
        del coordinates_ref[j]
        for i in range(len(coordinates_ref)):
            g.append("(x - {})".format(lista_de_coordenadas[coordinates_ref[i]][0]))
            h.append("({} - {})".format(lista_de_coordenadas[j][0], lista_de_coordenadas[coordinates_ref[i]][0]))
        numerador = expand("*".join(str(x) for x in g))
        denominador = expand("*".join(str(x) for x in h))
        lagrange = expand(numerador / denominador)
        lagrange_list.append(lagrange)
        g = []
        h = []
        coordinates_ref = index.copy()

    for i in range(len(lagrange_list)):
        p_list.append(lagrange_list[i] * lista_de_coordenadas[i][1])

    result = expand("+".join(str(x) for x in p_list))  # Armazena o polinômio que interpola o conjunto de pontos

    """
    print("Para xk = {} => P(x) = {}".format(xk, result.subs(x, xk)))
    print("P(x) = {}".format(simplify(result)))
    """

    x_coordinates = []  # Armazena a posição de x para cada sublista de coordinates
    y_coordinates = []  # Armazena a posição de y para cada sublista de coordinates

    for i in range(len(lista_de_coordenadas)):
        x_coordinates.append(lista_de_coordenadas[i][0])

    for i in range(len(lista_de_coordenadas)):
        y_coordinates.append(lista_de_coordenadas[i][1])

    interval = np.linspace(lista_de_coordenadas[0][0], lista_de_coordenadas[-1][0], num=100)
    fx = []
    for i in range(len(interval)):
        fx.append(result.subs(x, interval[i]))

    fig = plt.figure(facecolor='white')
    plt.plot(interval, fx, color='b')
    plt.grid(True, linewidth=.5, linestyle='--', color='r')
    plt.scatter(x_coordinates, y_coordinates, color='g')
    st.pyplot(fig)
