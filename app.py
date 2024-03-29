# -*- coding: utf-8 -*-

import pandas as pd
import streamlit as st
import numpy as np
from PIL import Image
from sympy import pi, symbols
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import copy

x = symbols("x")

fis, Ls, Es, As, points, elements, elementsComNos, indicesElementos, Ds, deslocamentosAgrupados, = [], [], [], [], [], \
                                                                                                   [], [], [], [], []
rows = []
vvs = []

dados_gerais = []

xp1s, yp1s, zp1s, xp2s, yp2s, zp2s = [], [], [], [], [], []

cteLs, cteMs, cteNs = [], [], []

def lerArquivo():
    uploaded_file = st.file_uploader("Escolha um arquivo")

    @st.cache(suppress_st_warning=True)
    def readCSVFile(file):
        if file is not None:
            return pd.read_csv(file)
        else:
            return st.error(body='This is an error')

    pandasToPythonList = readCSVFile(uploaded_file).values.tolist()

    for row in pandasToPythonList:
        rows.append(row)

    for i in range(len(rows)):
        xp1, yp1, zp1, xp2, yp2, zp2, Ei, D = [
            rows[i][0],
            rows[i][1],
            rows[i][2],
            rows[i][3],
            rows[i][4],
            rows[i][5],
            rows[i][6],
            rows[i][7]
        ]

        xp1s.append(float(xp1))
        yp1s.append(float(yp1))
        zp1s.append(float(zp1))
        xp2s.append(float(xp2))
        yp2s.append(float(yp2))
        zp2s.append(float(zp2))
        Es.append(Ei)
        Ds.append(D)

        if [xp1, yp1, zp1] not in points:
            points.append([xp1, yp1, zp1])
        if [xp2, yp2, zp2] not in points:
            points.append([xp2, yp2, zp2])

        elements.append([[float(xp1), float(yp1), float(zp1)], [float(xp2), float(yp2), float(zp2)]])
        comprimento = ((float(xp2) - float(xp1)) ** 2 +
                        (float(yp2) - float(yp1)) ** 2 +
                        (float(zp2) - float(zp1)) ** 2) ** .5

        cteL = (xp2s[i] - xp1s[i]) / comprimento
        cteM = (yp2s[i] - yp1s[i]) / comprimento
        cteN = (zp2s[i] - zp1s[i]) / comprimento

        Ls.append(comprimento)
        As.append(D ** 2)
        cteLs.append(cteL)
        cteMs.append(cteM)
        cteNs.append(cteN)

def lerInputsManuais():
    n_elementos = st.number_input("Número de elementos", step=1, key="elements_number")

    for i in range(n_elementos):
        with st.sidebar.expander("Elemento {}".format(i + 1)):
            xp1 = st.number_input(
                'xp1',
                key='xp1_{}'.format(i)
            )
            yp1 = st.number_input(
              "yp1",
              key='yp1_{}'.format(i)
            )
            zp1 = st.number_input(
              "zp1",
              key='zp1_{}'.format(i)
            )
            xp2 = st.number_input(
              "xp2",
              key='xp2_{}'.format(i)
            )
            yp2 = st.number_input(
              "yp2",
              key='yp2_{}'.format(i)
            )
            zp2 = st.number_input(
              "zp2",
              key='zp2_{}'.format(i)
            )
            option = st.selectbox(
            'Materiais',
            (
                'Estrutural (ASTM-A36)', 
                'ASTM-A709 Classe 345',
                'ASTM-A913 Classe 450',
                'ASTM-A992 Classe 345',
                'ASTM-A709 Classe 690',
                'Liga 1100-H14 (99% Al)',
                'Liga 2014-T6',
                'Liga 2024-T4',
                'Liga 5456-H116',
                'Liga 6061-T6',
                'Liga 7075-T6'
             ),
            key='material_option_{}'.format(i)
            )
            propriedades = [
                [400, 250, 200000000000],
                [450, 345, 200000000000],
                [550, 450, 200000000000],
                [450, 345, 200000000000],
                [760, 690, 200000000000],
                [110, 95, 70000000000],
                [455, 400, 75000000000],
                [470, 325, 73000000000],
                [315, 230, 72000000000],
                [260, 240, 70000000000],
                [570, 500, 72000000000]
            ]
        
            if option == 'Estrutural (ASTM-A36)':
                Ei = st.number_input("E (Pa)", value=propriedades[0][2], key='material_estrutural_{}'.format(i))
            elif option == 'ASTM-A709 Classe 345':
                Ei = st.number_input("E (Pa)", value=propriedades[1][2], key='material_709_{}'.format(i))
            elif option == 'ASTM-A913 Classe 450':
                Ei = st.number_input("E (Pa)", value=propriedades[2][2], key='material_913_{}'.format(i))
            elif option == 'ASTM-A992 Classe 345':
                Ei = st.number_input("E (Pa)", value=propriedades[3][2], key='material_992_{}'.format(i))
            elif option == 'ASTM-A709 Classe 690':
                Ei = st.number_input("E (Pa)", value=propriedades[4][2], key='material_709_{}'.format(i))
            elif option == 'Liga 1100-H14 (99% Al)':
                Ei = st.number_input("E (Pa)", value=propriedades[5][2], key='material_h14_{}'.format(i))
            elif option == 'Liga 2014-T6':
                Ei = st.number_input("E (Pa)", value=propriedades[6][2], key='material_t6_{}'.format(i))
            elif option == 'Liga 2024-T4':
                Ei = st.number_input("E (Pa)", value=propriedades[7][2], key='material_t4_{}'.format(i))
            elif option == 'Liga 5456-H116':
                Ei = st.number_input("E (Pa)", value=propriedades[8][2], key='material_h116_{}'.format(i))
            elif option == 'Liga 6061-T6':
                Ei = st.number_input("E (Pa)", value=propriedades[9][2], key='material_6061_{}'.format(i))
            else:
                Ei = st.number_input("E (Pa)", value=propriedades[10][2], key='material_7075_{}'.format(i))
            D = st.number_input(
              "D (m)",
              key='diametro_{}'.format(i)
            )
        xp1s.append(xp1)
        yp1s.append(yp1)
        zp1s.append(zp1)
        xp2s.append(xp2)
        yp2s.append(yp2)
        zp2s.append(zp2)
        Es.append(Ei)
        Ds.append(D)

    for i in range(n_elementos):
        rows.append([xp1s[i], yp1s[i], zp1s[i], xp2s[i], yp2s[i], zp2s[i], Es[i], Ds[i]])

        if [xp1s[i], yp1s[i], zp1s[i]] not in points:
            points.append([xp1s[i], yp1s[i], zp1s[i]])
        if [xp2s[i], yp2s[i], zp2s[i]] not in points:
            points.append([xp2s[i], yp2s[i], zp2s[i]])

        comprimento = ((float(xp2s[i]) - float(xp1s[i])) ** 2 +
                  (float(yp2s[i]) - float(yp1s[i])) ** 2 +
                  (float(zp2s[i]) - float(zp1s[i])) ** 2) ** .5

        cteL = (xp2s[i] - xp1s[i]) / comprimento
        cteM = (yp2s[i] - yp1s[i]) / comprimento
        cteN = (zp2s[i] - zp1s[i]) / comprimento

        elements.append([[xp1s[i], yp1s[i], zp1s[i]], [xp2s[i], yp2s[i], zp2s[i]]])
        Ls.append(comprimento)
        As.append(pi / 4 * Ds[i] ** 2)
        cteLs.append(cteL)
        cteMs.append(cteM)
        cteNs.append(cteN)

with st.sidebar.expander("Exemplos"):
    st.text('Básico')
    image = Image.open('./images/estrutura_2.png')
    st.image(image, use_column_width=True)
    with open('./csv_files/estrutura_3D_14.csv', 'rb') as f:
        st.download_button('Baixar', f, file_name='estrutura_3D_14.csv')

    st.text('Tensões')
    image = Image.open('./images/estrutura_11.png')
    st.image(image, use_column_width=True)
    with open('./csv_files/estrutura_3D_4.csv', 'rb') as f:
        st.download_button('Baixar', f, file_name='estrutura_3D_4.csv')
        
    st.text('Estrutura complexa')
    image = Image.open('./images/estrutura_70.png')
    st.image(image, use_column_width=True)
    with open('./csv_files/estrutura_3D_70.csv', 'rb') as f:
        st.download_button('Baixar', f, file_name='estrutura_3D_70.csv')

st.title("Método dos Elementos Finitos")

resposta = st.radio(
    "Entrada de dados?",
    ('Manual', 'Arquivo .csv'),
    key="radio_input",
    index=0
)

if resposta == 'Arquivo .csv':
   lerArquivo()
else:
   lerInputsManuais()


col1, col2 = st.columns(2)

fig = plt.figure(facecolor='white')
ax = fig.add_subplot(111, projection="3d")

colA, colB, colC, colD, colE = st.columns(5)

with colA:
    color_estrutura = st.color_picker('Estrutura', '#7159c1', key=33)
with colB:
    background_color = st.color_picker("Background", '#7159c1', key=44)
with colC:
    edge_color = st.color_picker("Cantos", '#7159c1', key=55)
with colD:
    axes_color = st.color_picker("Eixos", '#ff5f5f', key=66)
with colE:
    points_color = st.color_picker("Nós", '#ff5f5f', key=77)
    
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
fig.patch.set_facecolor(background_color)

ax.xaxis.pane.set_edgecolor(edge_color)
ax.yaxis.pane.set_edgecolor(edge_color)
ax.zaxis.pane.set_edgecolor(edge_color)
ax.set_facecolor(background_color)

ax.xaxis.label.set_color(axes_color)  # setting up X-axis label color to yellow
ax.yaxis.label.set_color(axes_color)
ax.zaxis.label.set_color(axes_color)

ax.tick_params(axis='x', colors=axes_color)  # setting up X-axis tick color to red
ax.tick_params(axis='y', colors=axes_color)
ax.tick_params(axis='z', colors=axes_color)

with col1:
    elevation = st.slider('Elevação', 90, 180, 90)
    azimuth = st.slider('Azimute', 0, 360, 270)
    scale_x = st.slider('Escala em x', 0.0, 10.0, 1.0, key='sc_x1', step=.25)
    scale_y = st.slider('Escala em y', 0.0, 10.0, 1.0, key='sc_y1', step=.25)
    scale_z = st.slider('Escala em z', 0.0, 10.0, 1.0, key='sc_z1', step=.25)
    
    ax.view_init(elevation, azimuth)
with col2:
    st.header("Vista geral")
    st.write("A estrutura mostrada apresenta {} nós e {} elementos".format(len(points), len(elements)))
    ax.set_box_aspect((scale_x, scale_y, scale_z))
    for i in range(len(elements)):
        xs, ys, zs = zip(elements[i][0], elements[i][1])
        ax.plot(xs, ys, zs, color=color_estrutura, linewidth=2)

    for i in range(len(points)):
        ax.scatter(float(points[i][0]), float(points[i][1]), points[i][2], color=points_color, s=12)

    st.pyplot(fig)
# -------------------------------------------------------------------------------------------
with st.sidebar.expander("Propriedades dos elementos"):
    if st.checkbox("Diâmetro"):
        modificarQuaisElementos = st.text_input(
            "Elementos (1 a {})".format(len(elements)),
            value='1',
            key='check_d',
            placeholder='1, 2, 4'
        )
        novoDiametro = st.number_input("D (m)", value=Ds[1])

        for i in range(len(modificarQuaisElementos.split(','))):
            Ds[int(modificarQuaisElementos.split(',')[i]) - 1] = novoDiametro
            As[int(modificarQuaisElementos.split(',')[i]) - 1] = float(novoDiametro) ** 2

    if st.checkbox("Módulo de elasticidade"):
        modificarQuaisElementos = st.text_input(
            "Elementos (1 a {})".format(len(elements)),
            value='1',
            key='check_E',
            placeholder='1, 2, 4'
        )
        option = st.selectbox(
            'Materiais',
            (
                'Estrutural (ASTM-A36)', 
                'ASTM-A709 Classe 345',
                'ASTM-A913 Classe 450',
                'ASTM-A992 Classe 345',
                'ASTM-A709 Classe 690',
                'Liga 1100-H14 (99% Al)',
                'Liga 2014-T6',
                'Liga 2024-T4',
                'Liga 5456-H116',
                'Liga 6061-T6',
                'Liga 7075-T6'
             ),
            key='material_option_1'
            )
        
        propriedades = [
            [400, 250, 200000000000],
            [450, 345, 200000000000],
            [550, 450, 200000000000],
            [450, 345, 200000000000],
            [760, 690, 200000000000],
            [110, 95, 70000000000],
            [455, 400, 75000000000],
            [470, 325, 73000000000],
            [315, 230, 72000000000],
            [260, 240, 70000000000],
            [570, 500, 72000000000]
        ]
        
        if option == 'Estrutural (ASTM-A36)':
            novoModuloE = st.number_input("E (Pa)", value=propriedades[0][2])
        elif option == 'ASTM-A709 Classe 345':
            novoModuloE = st.number_input("E (Pa)", value=propriedades[1][2])
        elif option == 'ASTM-A913 Classe 450':
            novoModuloE = st.number_input("E (Pa)", value=propriedades[2][2])
        elif option == 'ASTM-A992 Classe 345':
            novoModuloE = st.number_input("E (Pa)", value=propriedades[3][2])
        elif option == 'ASTM-A709 Classe 690':
            novoModuloE = st.number_input("E (Pa)", value=propriedades[4][2])
        elif option == 'Liga 1100-H14 (99% Al)':
            novoModuloE = st.number_input("E (Pa)", value=propriedades[5][2])
        elif option == 'Liga 2014-T6':
            novoModuloE = st.number_input("E (Pa)", value=propriedades[6][2])
        elif option == 'Liga 2024-T4':
            novoModuloE = st.number_input("E (Pa)", value=propriedades[7][2])
        elif option == 'Liga 5456-H116':
            novoModuloE = st.number_input("E (Pa)", value=propriedades[8][2])
        elif option == 'Liga 6061-T6':
            novoModuloE = st.number_input("E (Pa)", value=propriedades[9][2])
        else:
            novoModuloE = st.number_input("E (Pa)", value=propriedades[10][2])    
            

        for i in range(len(modificarQuaisElementos.split(','))):
            Es[int(modificarQuaisElementos.split(',')[i]) - 1] = novoModuloE

listaP, pontoNo = [], []

for item in elements:
    for j in range(2):
        listaP.append(item[j])

for i in range(len(listaP)):
    if listaP[i] not in pontoNo:
        pontoNo.append(listaP[i])

pontoNoAgrupado = []

for i in range(len(pontoNo)):
    pontoNoAgrupado.append([pontoNo[i], i + 1])

indicesElementos = copy.deepcopy(elements)

for i in range(len(elements)):
    for j in range(2):
        for k in range(len(pontoNoAgrupado)):
            if elements[i][j] == pontoNoAgrupado[k][0]:
                indicesElementos[i][j] = pontoNoAgrupado[k][1]

for i in range(len(indicesElementos)):
    elementsComNos.append([[indicesElementos[i][0], elements[i][0][0], elements[i][0][1], elements[i][0][2]],
                            [indicesElementos[i][1], elements[i][1][0], elements[i][1][1], elements[i][1][2]]])

lista = []

with st.expander("Matriz de rigidez de cada elemento"):
    st.header("Matriz de rigidez de cada elemento")
    for i in range(len(rows)):
        E = float(Es[i])
        A = float(As[i])
        L = float(Ls[i])
        l = cteLs[i]
        m = cteMs[i]
        n = cteNs[i]

        cte = E * A / L
        k = cte * np.array([
            [l ** 2, l * m, l * n, -l ** 2, -l * m, -l * n],
            [l * m, m ** 2, m * n, -l * m, -m ** 2, -m * n],
            [l * n, m * n, n ** 2, -l * n, -m * n, -n ** 2],
            [-l ** 2, -l * m, -l * n, l ** 2, l * m, l * n],
            [-l * m, -m ** 2, -m * n, l * m, m ** 2, m * n],
            [-l * n, -m * n, -n ** 2, l * n, m * n, n ** 2]
        ])

        lista.append(k)

        for n in range(len(k)):
            for j in range(len(k)):
                k[n][j] = float(format(float(k[n][j]), ".1f"))

        kLaTeXForm = np.array(k)

        st.subheader('Matrix de rigidez do elemento {}'.format(i + 1))
        st.write(kLaTeXForm)

# Define listaGlobal
listaGlobal = []

# Insere linhas em listaGlobal
for i in range(3 * len(pontoNo)):
    listaGlobal.append([])

# Insere zeros nas linhas de listaGlobal
for i in range(3 * len(pontoNo)):
    for j in range(3 * len(pontoNo)):
        listaGlobal[i].append(0)

# Cria uma lista com os índices duplos que servirão de referência aos índices do python
linha = []

for i in range(len(pontoNo)):
    linha.append(i + 1)
    linha.append(i + 1)
    linha.append(i + 1)

# indicesElementos = [[1, 2], [2, 4], [3, 4], [2, 3]]

indices = []

for i in range(len(rows)):
    indices.append([])

for j in range(len(indicesElementos)):
    for i in indicesElementos[j]:
        for item in range(len(linha)):
            if linha[item] == i:
                indices[j].append(item)

for k in range(len(lista)):
    for (newItem, i) in zip(indices[k], range(0, 6, 1)):    
        for (item, j) in zip(indices[k], range(0, 6, 1)):
            listaGlobal[newItem][item] += lista[k][i][j]

for i in range(len(listaGlobal)):
    for j in range(len(listaGlobal)):
        listaGlobal[i][j] = float(format(float(listaGlobal[i][j]), ".1f"))

listaGlobalNumpy = np.array(listaGlobal)

with st.expander("Matriz de rigidez global ({} x {})".format(3 * len(points), 3 * len(points))):
    st.write(listaGlobalNumpy)

# ----------------------------------------------------------------------------------------------------------------------

n_elementos_matriz_de_forcas = 3 * len(pontoNo)

forcas = []

# st.write("pontoNo", pontoNo)

pontoNoComRestricoes = []

for i in range(len(pontoNo)):
    with st.sidebar.expander("Nó {}".format(i + 1)):
        resposta = st.radio(
            "Quais as restrições?",
            ('X', 'Y', 'Z', 'XY', 'XZ', 'YZ', 'XYZ', 'L'),
            key="radio_{}".format(i),
            index=7
        )

        if resposta == 'X':
            forcas.append(['u{}:'.format(i + 1), "R"])
            forcas.append(['v{}:'.format(i + 1), 0])
            forcas.append(['w{}:'.format(i + 1), 0])
        elif resposta == 'Y':
            forcas.append(['u{}:'.format(i + 1), 0])
            forcas.append(['v{}:'.format(i + 1), "R"])
            forcas.append(['w{}:'.format(i + 1), 0])
        elif resposta == 'Z':
            forcas.append(['u{}:'.format(i + 1), 0])
            forcas.append(['v{}:'.format(i + 1), 0])
            forcas.append(['w{}:'.format(i + 1), "R"])
        elif resposta == 'XY':
            for j in (['u', 'v']):
                forcas.append(['{}{}'.format(j, i + 1), 'R'])
            forcas.append(['w{}:'.format(i + 1), 0])
        elif resposta == 'XZ':
            forcas.append(['u{}:'.format(i + 1), "R"])
            forcas.append(['v{}:'.format(i + 1), 0])
            forcas.append(['w{}:'.format(i + 1), "R"])
        elif resposta == 'YZ':
            forcas.append(['u{}:'.format(i + 1), 0])
            forcas.append(['v{}:'.format(i + 1), "R"])
            forcas.append(['w{}:'.format(i + 1), "R"])
        elif resposta == 'XYZ':
            forcas.append(['u{}:'.format(i + 1), "R"])
            forcas.append(['v{}:'.format(i + 1), "R"])
            forcas.append(['w{}:'.format(i + 1), "R"])
        else:
            for (m, n) in zip(['x', 'y', 'z'], ['u', 'v', 'w']):
                nova_resposta = st.number_input(
                    "F{} (N)".format(m),
                    value=0,
                    key="nr{}".format(i),
                )
                forcas.append(['{}{}'.format(n, i + 1), nova_resposta])
                
        pontoNoComRestricoes.append([pontoNo[i], resposta])


# st.write(pontoNoComRestricoes)
# st.write("new_forcas", forcas)
forcasFiltradoComUeV = []
forcasFiltrado = []

for i in range(int(len(forcas))):
    if type(forcas[i][1]) == float or type(forcas[i][1]) == int:
        forcasFiltradoComUeV.append(forcas[i])

# st.write("forcasFiltradoComUeV", forcasFiltradoComUeV)

for i in range(int(len(forcas))):
    if type(forcas[i][1]) == float or type(forcas[i][1]) == int:
        forcasFiltrado.append(forcas[i][1])

# st.write("forcasFiltrado", forcasFiltrado)
ccs = []

for item in forcas:
    if item[1] == 'R':
        ccs.append(forcas.index(item))

a = np.delete(listaGlobalNumpy, ccs, axis=1)
a = np.delete(a, ccs, axis=0)


numpyListInverse = np.linalg.inv(a)

deslocamentosNumpy = np.matmul(numpyListInverse, forcasFiltrado)

deslocamentosArray = deslocamentosNumpy.tolist()
deslocamentosComUeV = []

for i in range(len(forcasFiltradoComUeV)):
    deslocamentosComUeV.append(('{}'.format(forcasFiltradoComUeV[i][0]), deslocamentosArray[i]))

for i in range(len(forcas)):
    for j in range(len(deslocamentosComUeV)):
        if forcas[i][1] == 'R':
            forcas[i][1] = 0
        elif forcas[i][0] == deslocamentosComUeV[j][0]:
            forcas[i][1] = deslocamentosComUeV[j][1]

for i in range(0, len(forcas), 1):
    del forcas[i][0]

deslocamentosAgrupados = []

for (i, j) in zip(range(0, len(forcas), 3), range(len(forcas))):
    deslocamentosAgrupados.append([j + 1, forcas[i][0], forcas[i + 1][0], forcas[i + 2][0]])

col1, col2 = st.columns(2)

containerDeslocamentos = st.container()

deslocamentosConfiguradosParaATabela = copy.deepcopy(deslocamentosAgrupados)  # Qdo é "R"

for i in range(len(deslocamentosAgrupados)):
    for j in range(1, 4, 1):
        if deslocamentosConfiguradosParaATabela[i][j] == "R":
            deslocamentosConfiguradosParaATabela[i][j] = 0

deslocamentosAgrupadosTabela = []
with containerDeslocamentos.expander("Deslocamentos"):
    deslocamentosAgrupadosTabela.append(["Nó", "u (m)", "v (m)", "w (m)"])
    for i in range(len(deslocamentosConfiguradosParaATabela)):
        deslocamentosAgrupadosTabela.append(
            ["Nó {}".format(i + 1),
                format(deslocamentosConfiguradosParaATabela[i][1], ".2e"),
                format(deslocamentosConfiguradosParaATabela[i][2], ".2e"),
                format(deslocamentosConfiguradosParaATabela[i][3], ".2e")
                ]
        )
    st.write(np.array(deslocamentosAgrupadosTabela))

newElements = copy.deepcopy(elementsComNos)

for i in range(len(newElements)):
    for j in range(len(deslocamentosAgrupados)):
        for k in range(2):
            if newElements[i][k][0] == deslocamentosConfiguradosParaATabela[j][0]:
                newElements[i][k][1] += deslocamentosConfiguradosParaATabela[j][1]
                newElements[i][k][2] += deslocamentosConfiguradosParaATabela[j][2]

# st.write("newElements", newElements)

# Deleta os índices da primeira posição usados como referência
for i in range(len(newElements)):
    for j in range(2):
        del newElements[i][j][0]

novoComprimento = []
for i in range(len(newElements)):
    for j in range(1):
        comprimento = ((newElements[i][j][0] - newElements[i][j + 1][0]) ** 2 +
                        (newElements[i][j][1] - newElements[i][j + 1][1]) ** 2 +
                        (newElements[i][j][2] - newElements[i][j + 1][2]) ** 2) ** .5

        novoComprimento.append(comprimento)

deformacoes = []

# st.write("Ls", Ls)
for i in range(len(novoComprimento)):
    epsilon = (novoComprimento[i] - Ls[i]) / Ls[i]
    deformacoes.append(epsilon)

deformacoesTabela = []
with st.expander("Deformações"):
    deformacoesTabela.append(["Elemento", "%"])
    for i in range(len(deformacoes)):
        deformacoesTabela.append(["{}".format(i + 1),
                                    format(deformacoes[i],
                                            '.{}e'.format(2))
                                    ])

    st.write(np.array(deformacoesTabela))

tensoes = []

for i in range(len(deformacoes)):
    sigma = Es[i] * deformacoes[i]
    tensoes.append(sigma)

tensoesTabela = []
with st.expander("Tensões"):
    tensoesTabela.append(["Elemento", "MPa"])
    for i in range(len(tensoes)):
        tensoesTabela.append(["{}".format(i + 1),
                                format(tensoes[i] * 10 ** (-6),
                                        '.{}e'.format(2))
                                ])

    st.write(np.array(tensoesTabela))
    
ltotal = 0
for item in Ls:
    ltotal += item 

newPointsWithRep = copy.deepcopy(newElements)

# st.write("newPointsWithRep", newPointsWithRep)
newPoints = []

for i in range(len(newPointsWithRep)):
    for j in range(2):
        if newPointsWithRep[i][j] not in newPoints:
            newPoints.append(newPointsWithRep[i][j])

# st.write("elements", elements)
# st.write("points", points)
# Tensão nos elementos

# ----------------------------------------------------------------------------------------------------
# st.write("newPoints", newPoints)
                
with st.expander("Gráfico"):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111, projection="3d")

    # ------------------------------------------------------
    
    # ------------------------------------------------------

    
    col1, col2, col3, col4, col5 = st.columns(5)
    

    with st.sidebar.expander("Visualização"):
        with col1:
            color_estrutura = st.color_picker('Estrutura', '#EC2997')
        with col2:
            color_deformacao = st.color_picker('Deformação', '#A600F9')
        with col3:
            background_color = st.color_picker("Background", '#D6BEA9')
        with col4:
            edge_color = st.color_picker("Cantos", '#FF5B00')
        with col5:
            axes_color = st.color_picker("Eixos", '#830D0D')

        resposta = st.radio(
            "Gráficos",
            ('Estrutura', 'Estrutura + Deslocamentos', 'Estrutura + Deslocamentos + Tensões', 'Restrições'))
        numerar = st.checkbox("Numerar nós")
        numerar_pos_deformacao = st.checkbox("Numerar nós deslocados")
        numerar_elems = st.checkbox("Numerar elementos")
        cotas = st.checkbox("Cotas e Grid")
        rotulos = st.checkbox("Rótulos nos eixos (x, y, z)")
        

        xs_min_max, ys_min_max, zs_min_max = [], [], []

        for i in range(len(newElements)):
            for j in range(2):
                xs_min_max.append(newElements[i][j][0])
                ys_min_max.append(newElements[i][j][1])
                zs_min_max.append(newElements[i][j][2])

        st.write("Rotação")
        elevation = st.slider('Elevação', 0, 180, 40, key=123321)
        azimuth = st.slider('Azimute', 0, 360, 225, key='12_azi')
        st.write('Escala dos eixos')
        scale_x = st.slider('Escala em x', 0, 10, 1, key='sc_x')        
        scale_y = st.slider('Escala em y', 0, 10, 1, key='sc_y')
        scale_z = st.slider('Escala em z', 0, 10, 1, key='sc_z')
        escala_barras = st.number_input('Escala das barras', min_value=30)

    if resposta == 'Estrutura':
        for i in range(len(elements)):
            xs, ys, zs = zip(elements[i][0], elements[i][1])
            ax.plot(xs, ys, zs, color=color_estrutura, linewidth=Ds[i] * escala_barras)

        for i in range(len(points)):
            ax.scatter(float(points[i][0]), float(points[i][1]), points[i][2], s=10)

    elif resposta == 'Estrutura + Deslocamentos':
        for i in range(len(elements)):
            xs, ys, zs = zip(elements[i][0], elements[i][1])
            ax.plot(xs, ys, zs, color=(0, 0, 1, .1), linewidth=Ds[i] * escala_barras)

        for i in range(len(points)):
            ax.scatter(float(points[i][0]), float(points[i][1]), points[i][2], s=5)

        for i in range(len(newElements)):
            xs, ys, zs = zip(newElements[i][0], newElements[i][1])
            ax.plot(xs, ys, zs, color=color_deformacao, linewidth=Ds[i] * escala_barras)
    elif resposta == 'Estrutura + Deslocamentos + Tensões':
        for k in range(len(elements)):
            xs, ys, zs = zip(elements[k][0], elements[k][1])
            ax.plot(xs, ys, zs, color=(0, 0, 0, .1), linewidth=Ds[k] * 5)

        for i in range(len(newElements)):
            xs, ys, zs = zip(newElements[i][0], newElements[i][1])

            red_patch = mpatches.Patch(color='red', label='Maior tração')
            green_patch = mpatches.Patch(color=(0, 1, 0), label='Neutro')
            blue_patch = mpatches.Patch(color='blue', label='Maior compressão')

            for j in range(len(newPoints)):
                ax.scatter(float(newPoints[j][0]), float(newPoints[j][1]), newPoints[j][2], s=2, color='yellow')

            ax.legend(handles=[red_patch, green_patch, blue_patch])

            if tensoes[i] > .1 * max(tensoes):
                ax.plot(xs, ys, zs, color=(tensoes[i] / max(tensoes), 0, 0), linewidth=Ds[i] * escala_barras)
            elif tensoes[i] < .1 * min(tensoes):
                ax.plot(xs, ys, zs, color=(0, 0, abs(tensoes[i] / min(tensoes))), linewidth=Ds[i] * escala_barras)
            elif .1 * max(tensoes) > tensoes[i] >= .1 * min(tensoes):
                ax.plot(xs, ys, zs, color=(0, 1, 0), linewidth=Ds[i] * escala_barras)
    else:
        for i in range(len(elements)):
            xs, ys, zs = zip(elements[i][0], elements[i][1])
            ax.plot(xs, ys, zs, color=color_estrutura, linewidth=Ds[i] * escala_barras)

        for i in range(len(points)):
            if pontoNoComRestricoes[i][1] == 'X':
                ax.scatter(float(points[i][0]), float(points[i][1]), points[i][2], s=20, color='blue')
            elif pontoNoComRestricoes[i][1] == 'Y':
                ax.scatter(float(points[i][0]), float(points[i][1]), points[i][2], s=20, color='blue')
            elif pontoNoComRestricoes[i][1] == 'Z':
                ax.scatter(float(points[i][0]), float(points[i][1]), points[i][2], s=20, color='blue')
            elif pontoNoComRestricoes[i][1] == 'XY':
                ax.scatter(float(points[i][0]), float(points[i][1]), points[i][2], s=20, color='blue')
            elif pontoNoComRestricoes[i][1] == 'XZ':
                ax.scatter(float(points[i][0]), float(points[i][1]), points[i][2], s=20, color='magenta')
            elif pontoNoComRestricoes[i][1] == 'YZ':
                ax.scatter(float(points[i][0]), float(points[i][1]), points[i][2], s=20, color='blue')
            elif pontoNoComRestricoes[i][1] == 'XYZ':
                x1 = points[i][0]
                x2 = points[i][1]
                x3 = points[i][2]
                
                u = np.array([
                    [.5/2 + x1, -.5 + x2, .5/2 + x3],
                    [-.5/2 + x1, -.5 + x2, .5/2 + x3],
                    [-.5/2 + x1, -.5 + x2, -.5/2 + x3],
                    [.5/2 + x1, -.5 + x2, -.5/2 + x3],
                    [0 + x1, 0 + x2, 0 + x3]
                ])
                
                v = np.array([
                    [-.5 + x1, .5/2 + x2, -.5/2 + x3],
                    [-.5 + x1, .5/2 + x2, .5/2 + x3],
                    [-.5 + x1, -.5/2 + x2, .5/2 + x3],
                    [-.5 + x1, -.5/2 + x2, -.5/2 + x3],
                    [0 + x1, 0 + x2, 0 + x3]
                ])
                
                w = np.array([
                    [.5/2 + x1, -.5/2 + x2, -.5 + x3],
                    [.5/2 + x1, .5/2 + x2, -.5 + x3],
                    [-.5/2 + x1, .5/2 + x2, -.5 + x3],
                    [-.5/2 + x1, -.5/2 + x2, -.5 + x3],
                    [0 + x1, 0 + x2, 0 + x3]
                ])

                # generate list of sides' polygons of our pyramid
                v_verts = [[v[0],v[1],v[4]], [v[0],v[3],v[4]],
                [v[2],v[1],v[4]], [v[2],v[3],v[4]], [v[0],v[1],v[2],v[3]]]
                
                u_verts = [[u[0],u[1],u[4]], [u[0],u[3],u[4]],
                [u[2],u[1],u[4]], [u[2],u[3],u[4]], [u[0],u[1],u[2],u[3]]]
                
                w_verts = [[w[0],w[1],w[4]], [w[0],w[3],w[4]],
                [w[2],w[1],w[4]], [w[2],w[3],w[4]], [w[0],w[1],w[2],w[3]]]
                
                ax.add_collection3d(Poly3DCollection(v_verts, 
                facecolors=(0.7, 0, 0), linewidths=.5, edgecolors='r', alpha=0.7))
                
                ax.add_collection3d(Poly3DCollection(u_verts, 
                facecolors=(0, 0, .7), linewidths=.5, edgecolors='b', alpha=0.7))
                
                ax.add_collection3d(Poly3DCollection(w_verts, 
                facecolors=(0, .7, 0), linewidths=.5, edgecolors='g', alpha=0.7))
            else:
                ax.scatter(float(points[i][0]), float(points[i][1]), points[i][2], s=5, color=(0, 1, 0))

    if numerar:
        for i in range(len(pontoNo)):
            ax.text(pontoNo[i][0] + .3, pontoNo[i][1] + .3, pontoNo[i][2] + .3,
                    "{}".format(i + 1), color='black', ha='left', va='top', size=5)

    # Numerando os elementos
    mid_point_new_elements = []

    for i in range(len(newElements)):
        mid_point_new_elements.append([])

    for j in range(3):
        for i in range(len(newElements)):
            mid_point = (newElements[i][0][j] + newElements[i][1][j]) * .5
            mid_point_new_elements[i].append(mid_point)

    if numerar_elems:
        for i in range(len(mid_point_new_elements)):
            ax.text(mid_point_new_elements[i][0] + .2, mid_point_new_elements[i][1] + .2,
                    mid_point_new_elements[i][2] + .2,
                    "{}".format(i + 1), color='black', ha='left', va='bottom', size=4)

    if numerar_pos_deformacao:
        for m in range(len(newPoints)):
            ax.text(newPoints[m][0] + .2, newPoints[m][1] + .2, newPoints[m][2] + .2,
                    "{}'".format(m + 1), color='black', ha='left', va='bottom', size=5)

    ax.set_xlim(min(xs_min_max) - 1, max(xs_min_max) + 1)
    ax.set_ylim(min(ys_min_max) - 1, max(ys_min_max) + 1)
    ax.set_zlim(min(zs_min_max) - 1, max(zs_min_max) + 1)
    ax.set_box_aspect((scale_x, scale_y, scale_z))
    
    if rotulos:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    if not cotas:
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_zticks([])

    ax.xaxis.pane.set_edgecolor(edge_color)
    ax.yaxis.pane.set_edgecolor(edge_color)
    ax.zaxis.pane.set_edgecolor(edge_color)
    ax.set_facecolor(background_color)
    fig.patch.set_facecolor(background_color)

    ax.xaxis.label.set_color(axes_color)  # setting up X-axis label color to yellow
    ax.yaxis.label.set_color(axes_color)
    ax.zaxis.label.set_color(axes_color)

    ax.tick_params(axis='x', colors=axes_color, labelsize=5)  # setting up X-axis tick color to red
    ax.tick_params(axis='y', colors=axes_color, labelsize=5)
    ax.tick_params(axis='z', colors=axes_color, labelsize=5)

    ax.view_init(elevation, azimuth)

    st.pyplot(fig)
    
# Insere os dados de cada elemento

for i in range(len(elements)):
    dados_gerais.append(["{}".format(int(i + 1)), xp1s[i], xp2s[i], zp1s[i], xp2s[i], yp2s[i], zp2s[i], format(Ls[i], '.2e'), format(Es[i], '.2e'), format(Ds[i], '.2e')])


plt.savefig('plot_3D.pdf')
plt.savefig('plot_3D.svg')
plt.savefig('plot_3D.jpg')

deslocamentosNotacao = []
for i in range(len(deslocamentosAgrupados)):
    deslocamentosNotacao.append(['{}'.format(i + 1), format(deslocamentosAgrupados[i][1], '.2e'), format(deslocamentosAgrupados[i][2], '.2e'), format(deslocamentosAgrupados[i][3], '.2e')])

if st.button('Gerar dados'):

    del tensoesTabela[0]
    del deformacoesTabela[0]
    
    df_gerais = pd.DataFrame(np.array(dados_gerais), columns=['Elem.', 'x(p1) (m)', 'y(p1) (m)', 'z(p1) (m)', 'x(p2) (m)', 'y(p2) (m)', 'z(p2) (m)', 'L (m)', 'Ei (Pa)', 'D (m)'])
    df_deslocamentos = pd.DataFrame(np.array(deslocamentosNotacao), columns = ['Elem.', 'u (m)', 'v (m)', 'w (m)'])
    df_tensoes = pd.DataFrame(np.array(tensoesTabela), columns = ['Elem.', 'Sigma (MPa)'])
    df_deformacoes = pd.DataFrame(np.array(deformacoesTabela), columns = ['Elem.', 'Epsilon (%)'])
    
    gerais_html = df_gerais.to_html()
    deslocamentos_html = df_deslocamentos.to_html()
    tensoes_html = df_tensoes.to_html()
    deformacoes_html = df_deformacoes.to_html()

    text_file = open("index.html", "w")
    text_file.write(f'''
    <!DOCTYPE html>
    <html lang="pt">   
    <head>
        <meta http-equiv="Content-Type" content="text/html;charset=UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500&display=swap');
            
            @import url(https://fonts.googleapis.com/css?family=Open+Sans:400,600);

            *, *:before, *:after {'{'}
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            {'}'}
            body {'{'}
                font-family: 'Open Sans', sans-serif;
                width: 100%;
                height: 100vh;
                background-color: #7159c1;
                padding: 20px;
            {'}'}
            table {'{'}
                background: #012B39;
                border-radius: 0.25em;
                border-collapse: collapse;
                margin: 1em;
            {'}'}
            th {'{'}
                border-bottom: 1px solid #364043;
                color: #E2B842;
                font-size: 0.85em;
                font-weight: 600;
                padding: 0.5em 1em;
                text-align: left;
            {'}'}
            td {'{'}
                color: #fff;
                font-weight: 400;
                padding: 0.65em 1em;
            {'}'}
            .disabled td {'{'}
                color: #4F5F64;
            {'}'}
                tbody tr {'{'}
                transition: background 0.25s ease;
            {'}'}
            tbody tr:hover {'{'}
                background: #014055;
            {'}'}
            
        </style>
        <title>Dados da estrutura</title>
    </head>

    <body>
        <h1>Informa&#231;&#245;es dos elementos</h1>
        {gerais_html}
        <h2>Comprimento total de barras: {format(ltotal, '.2e')} m</h2>
        <h1>Deslocamentos nodais</h1>
        {deslocamentos_html}
        <h1>Tens&#245;es por elemento</h1>
        {tensoes_html}
        <h2>Tens&#227;o m&#225;xima de tra&#231;&#227;o: {format(max(tensoes)*10**(-6), '.2e')} MPa</h2>
        <h2>Tens&#227;o m&#225;xima de compress&#227;o: {format(min(tensoes)*10**(-6), '.2e')} MPa</h2>
        <h1>Deforma&#231;&#245;es</h1>
        {deformacoes_html}
        <h2>Alongamento m&#225;ximo: {format(max(deformacoes), '.2e')}</h2>
        <h2>Encurtamento m&#225;ximo: {format(min(deformacoes), '.2e')}</h2>
    </body>
    </html>
    ''')

    text_file.close()
    
    try:    
        with open('./index.html', 'rb') as f:
            st.download_button('Baixar', f, file_name='data.html')
    except ValueError:
        st.write("Erro") 

with st.sidebar.expander("Download do gráfico"):
    resposta = st.radio(
    "SVG ou PDF",
    ('SVG', 'PDF'),
    key="radio_input",
    index=0
)  
    if resposta == 'SVG':
        with open('./plot_3D.svg', 'rb') as f:
            st.download_button('Baixar', f, file_name='plot_3D.svg')
    else:
        with open('./plot_3D.pdf', 'rb') as f:
            st.download_button('Baixar', f, file_name='plot_3D.pdf')