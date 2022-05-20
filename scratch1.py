from sympy import pi, sin, cos, symbols, acos
import numpy as np
import array_to_latex as a2l
from pylatex import Document, Section, Subsection, Math, Matrix, NoEscape, Subsubsection

# Subs in matrix: https://docs.sympy.org/latest/modules/matrices/matrices.html#operations-on-entries

x = symbols("x")

fis, Ls, Es, As, points, elements = [], [], [], [], [], []
x1, y1 = 1, 0

n_elementos = int(input("n_elementos: "))
coords = int(input("n_coords: "))
vvs = []

xp1s, yp1s, xp2s, yp2s = [], [], [], []

for i in range(n_elementos):
    print("Elemento {}".format(i + 1))
    xp1, yp1 = float(input("x(p1): ")), float(input("y(p1): "))
    print(15 * "-")
    xp2, yp2, Ei, D = [
        float(input("x(p2): ")),
        float(input("y(p2): ")),
        float(input("Módulo de elasticidade: ")),
        float(input("Diâmetro: "))
    ]

    xp1s.append(xp1)
    yp1s.append(yp1)
    xp2s.append(xp2)
    yp2s.append(yp2)

    if [xp1, yp1] not in points:
        points.append([xp1, yp1])
    if [xp2, yp2] not in points:
        points.append([xp2, yp2])

    elements.append([[xp1, yp1, 0], [xp2, yp2, 0]])

    comprimento = ((xp2 - xp1) ** 2 + (yp2 - yp1) ** 2) ** .5

    vvs.append([xp2 - xp1, yp2 - yp1])
    Ls.append(comprimento)
    Es.append(Ei)
    As.append(pi / 4 * D ** 2)

print(90 * "-")
print("vvs:", vvs)
print("points:", points)
print("elements:", elements)

for i in range(len(vvs)):
    cosAlpha = (x1 * vvs[i][0] + y1 * vvs[i][1]) / (
            ((x1 ** 2 + y1 ** 2) ** .5) * ((vvs[i][0] ** 2 + vvs[i][1] ** 2) ** .5))

    if (vvs[i][0] >= 0 and vvs[i][1] >= 0) or (vvs[i][0] <= 0 <= vvs[i][1]):
        fis.append(float(acos(cosAlpha) * 180 / pi))
    elif vvs[i][0] <= 0 and vvs[i][1] <= 0:
        fis.append(float(acos(cosAlpha) * 180 / pi + 2 * (180 - acos(cosAlpha) * 180 / pi)))
    elif vvs[i][0] >= 0 >= vvs[i][1]:
        fis.append(float(acos(cosAlpha) * 180 / pi + 360 - 2 * acos(cosAlpha) * 180 / pi))
    else:
        fis.append("ok")

print(90 * "-")
print("ângulos: {}".format(fis))
print(90 * "-")
print("comprimentos: {}".format(Ls))

lista = []

for i in range(len(fis)):
    fi = float(fis[i] * pi / 180)
    E = float(Es[i])
    A = float(As[i])
    L = float(Ls[i])

    cte = E * A / L * .001

    aA = cte * (cos(x) ** 2).subs(x, fi)
    bB = cte * (cos(x) * sin(x)).subs(x, fi)
    cC = cte * (-cos(x) ** 2).subs(x, fi)
    dD = cte * (-cos(x) * sin(x)).subs(x, fi)
    eE = cte * (cos(x) * sin(x)).subs(x, fi)
    fF = cte * (sin(x) ** 2).subs(x, fi)
    gG = cte * (-cos(x) * sin(x)).subs(x, fi)
    hH = cte * (-sin(x) ** 2).subs(x, fi)
    iI = cte * (-cos(x) ** 2).subs(x, fi)
    jJ = cte * (-cos(x) * sin(x)).subs(x, fi)
    kK = cte * (cos(x) ** 2).subs(x, fi)
    lL = cte * (cos(x) * sin(x)).subs(x, fi)
    mM = cte * (-cos(x) * sin(x)).subs(x, fi)
    nN = cte * (-sin(x) ** 2).subs(x, fi)
    oO = cte * (cos(x) * sin(x)).subs(x, fi)
    pP = cte * (sin(x) ** 2).subs(x, fi)

    k = [
        [aA, bB, cC, dD],
        [eE, fF, gG, hH],
        [iI, jJ, kK, lL],
        [mM, nN, oO, pP]
    ]

    lista.append(k)

    for n in range(len(k)):
        for j in range(len(k)):
            k[n][j] = float(format(float(k[n][j]), ".1f"))

    print(90 * "-")
    print('Elemento {}: {}'.format(i + 1, k))
    print(90 * "-")
    kLaTeXForm = np.array(k)
    a2l.to_ltx(kLaTeXForm)

# Define listaGlobal
listaGlobal = []

# Insere linhas em listaGlobal
for i in range(2 * coords):
    listaGlobal.append([])

# Insere zeros nas linhas de listaGlobal
for i in range(2 * coords):
    for j in range(2 * coords):
        listaGlobal[i].append(0)

print(90 * "-")
print("listaGlobal: {}".format(listaGlobal))

# Cria uma lista com os índices duplos que servirão de referência aos índices do python
linha = []

for i in range(coords):
    linha.append(i + 1)
    linha.append(i + 1)

print(90 * "-")
print("linha: {}".format(linha))

indicesElementos = []
# indicesElementos = [[1, 2], [2, 4], [3, 4], [2, 3]]

print(90 * "-")
print("Elementos e nós associados: ")
print(90 * "-")
for i in range(n_elementos):
    print("Elemento {}: ".format(i + 1))
    n1 = int(input("Nó 1: "))
    n2 = int(input("Nó 2: "))
    print(30 * "-")
    indicesElementos.append([n1, n2])

print("indicesElementos: {}".format(indicesElementos))
print(90 * "-")

indices = []

for i in range(n_elementos):
    indices.append([])

for j in range(len(indicesElementos)):
    for i in indicesElementos[j]:
        for item in range(len(linha)):
            if linha[item] == i:
                indices[j].append(item)

print(90 * "-")
print("indices =", indices)

for k in range(len(lista)):
    for (newItem, i) in zip(indices[k], range(0, 4, 1)):
        for (item, j) in zip(indices[k], range(0, 4, 1)):
            listaGlobal[newItem][item] += lista[k][i][j]

for i in range(len(listaGlobal)):
    for j in range(len(listaGlobal)):
        listaGlobal[i][j] = float(format(float(listaGlobal[i][j]), ".1f"))

print(90 * "-")
print("listaGlobal: {}".format(listaGlobal))

myMatrix = np.array(listaGlobal)
print(90 * "-")
print("LaTeX code...")
print(90 * "-")
latexMatrix = a2l.to_ltx(myMatrix)
print(90 * "-")

# -----------------------------------------------------------------------------------
"""
if __name__ == '__main__':
    doc = Document()

    section = Section('Método dos elementos finitos: Treliça Plana')
    doc.append(section)

    subsection = Subsection('Introdução')

    subsection.append(NoEscape('O método dos elementos finitos faz uso das matrizes de rigidez como forma de análise '
                               'de um sistema. Dessa forma, tomando como base o sistema global de coordenadas é '
                               'possível saber por meio das características geométricas e propriedades materiais '
                               'dos elementos os esforços e deformações em cada um. Todavia, dentre as propriedades '
                               'mencionadas anteriormente existe o módulo de Young ({}), área da seção transversal das '
                               'barras {}, e o comprimento das barras {}. Como a análise feita é em relação ao sistema '
                               'global de coordenadas, calcula-se o ângulo {}'
                               .format('$E$', '$A$', '$L$', '$' + "\\" + 'theta' + '$')))

    newMatrix = [
        [NoEscape('{}cos^2{}'.format('\\', '\\theta')),
         NoEscape('{}cos{}{}sin{}'.format('\\', '\\theta', '\\', '\\theta')),
         NoEscape('-{}cos^2{}'.format('\\', '\\theta')),
         NoEscape('-{}cos{}{}sin{}'.format('\\', '\\theta', '\\', '\\theta'))
         ],
        [NoEscape('{}cos{}{}sin{}'.format('\\', '\\theta', '\\', '\\theta')),
         NoEscape('{}sin^2{}'.format('\\', '\\theta')),
         NoEscape('-{}cos{}{}sin{}'.format('\\', '\\theta', '\\', '\\theta')),
         NoEscape('-{}sin^2{}'.format('\\', '\\theta'))
         ],
        [NoEscape('-{}cos^2{}'.format('\\', '\\theta')),
         NoEscape('-{}cos{}{}sin{}'.format('\\', '\\theta', '\\', '\\theta')),
         NoEscape('{}cos^2{}'.format('\\', '\\theta')),
         NoEscape('{}cos{}{}sin{}'.format('\\', '\\theta', '\\', '\\theta'))
         ],
        [NoEscape('-{}cos{}{}sin{}'.format('\\', '\\theta', '\\', '\\theta')),
         NoEscape('-{}sin^2{}'.format('\\', '\\theta')),
         NoEscape('{}cos{}{}sin{}'.format('\\', '\\theta', '\\', '\\theta')),
         NoEscape('{}sin^2{}'.format('\\', '\\theta'))
         ]
    ]

    K = np.matrix(newMatrix)
    matrix = Matrix(K, mtype='b')
    math = Math(data=[NoEscape('k_{}={}dfrac{}{}'.format('{n}', '\\', '{EA}', '{L}')), matrix])
    subsection.append(math)

    section.append(subsection)

    subsection = Subsection('Componentes do sistema')

    subsection.append('Número de barras da estrutura: {}\n'.format(n_elementos))
    subsection.append('Número de nós da estrutura: {}'.format(coords))

    subsubsection = Subsubsection('Comprimento das barras')

    for i in range(len(Ls)):
        if i == (len(Ls) - 1):
            subsubsection.append('Barra {}: {} cm'.format(i + 1, Ls[i]))
        else:
            subsubsection.append('Barra {}: {} cm\n'.format(i + 1, Ls[i]))

    subsection.append(subsubsection)

    subsubsection = Subsubsection('Módulo de elasticidade das barras (MPa)')

    subsubsection.append('E = {} MPa'.format(Es[1] / 10 ** 6))

    subsection.append(subsubsection)

    section.append(subsection)

    subsection = Subsection('Matrizes de rigidez dos elementos')

    for i in range(len(lista)):
        myMatrix = [[NoEscape('k_{0}^{1}'.format('{11}', '{(' + str(i + 1) + ')}')),
                     NoEscape('k_{0}^{1}'.format('{12}', '{(' + str(i + 1) + ')}')),
                     NoEscape('k_{0}^{1}'.format('{13}', '{(' + str(i + 1) + ')}')),
                     NoEscape('k_{0}^{1}'.format('{14}', '{(' + str(i + 1) + ')}'))],
                    [NoEscape('k_{0}^{1}'.format('{21}', '{(' + str(i + 1) + ')}')),
                     NoEscape('k_{0}^{1}'.format('{22}', '{(' + str(i + 1) + ')}')),
                     NoEscape('k_{0}^{1}'.format('{23}', '{(' + str(i + 1) + ')}')),
                     NoEscape('k_{0}^{1}'.format('{24}', '{(' + str(i + 1) + ')}'))],
                    [NoEscape('k_{0}^{1}'.format('{31}', '{(' + str(i + 1) + ')}')),
                     NoEscape('k_{0}^{1}'.format('{32}', '{(' + str(i + 1) + ')}')),
                     NoEscape('k_{0}^{1}'.format('{33}', '{(' + str(i + 1) + ')}')),
                     NoEscape('k_{0}^{1}'.format('{34}', '{(' + str(i + 1) + ')}'))],
                    [NoEscape('k_{0}^{1}'.format('{41}', '{(' + str(i + 1) + ')}')),
                     NoEscape('k_{0}^{1}'.format('{42}', '{(' + str(i + 1) + ')}')),
                     NoEscape('k_{0}^{1}'.format('{43}', '{(' + str(i + 1) + ')}')),
                     NoEscape('k_{0}^{1}'.format('{44}', '{(' + str(i + 1) + ')}'))]
                    ]
        K = np.matrix(myMatrix)
        matrix = Matrix(K, mtype='b')
        math = Math(data=[NoEscape('k_{}='.format('{' + str(i + 1) + '}')), matrix])

        subsection.append(NoEscape('Elemento {}:'.format('{' + str(i + 1) + '}')))
        subsection.append(math)

        k = np.matrix(lista[i])
        matrix = Matrix(k, mtype='b')
        math = Math(data=[NoEscape('k_{}='.format('{' + str(i + 1) + '}')), matrix])
        subsection.append(math)

    section.append(subsection)

    subsection = Subsection('Matriz de rigidez Global')

    K = np.matrix(listaGlobal)
    matrix = Matrix(K, mtype='b')
    math = Math(data=['K=', matrix])
    subsection.append(math)

    section.append(subsection)

    doc.generate_pdf('numpy_ex_1', clean_tex=False)
"""

a = np.array(listaGlobal)

n_elementos_matriz_de_forcas = 2 * coords

forcas = []

for i in range(coords):
    print('Nó {}'.format(i + 1))
    resposta = input('Restrições: x [X], y [Y], x e y [XY], livre [L]: ')

    if resposta == 'X':
        novaResposta = float(input('Fy = '))
        forcas.append(['u{}:'.format(i + 1), 0])
        forcas.append(['v{}:'.format(i + 1), novaResposta])
    elif resposta == 'Y':
        novaResposta = float(input('Fx = '))
        forcas.append(['u{}:'.format(i + 1), novaResposta])
        forcas.append(['v{}:'.format(i + 1), 0])
    elif resposta == 'XY':
        for j in (['u', 'v']):
            forcas.append(['{}{}'.format(j, i + 1), 'R'])
    else:
        for (m, n) in zip(['x', 'y'], ['u', 'v']):
            novaResposta = float(input('F{} = '.format(m)))
            forcas.append(['{}{}'.format(n, i + 1), novaResposta])

forcasFiltradoComUeV = []
forcasFiltrado = []

for i in range(int(len(forcas))):
    if type(forcas[i][1]) == float:
        forcasFiltradoComUeV.append(forcas[i])

for i in range(int(len(forcas))):
    if type(forcas[i][1]) == float:
        forcasFiltrado.append(forcas[i][1])

print('forcas = {}'.format(forcas))
print('forcasFiltradoComUeV = {}'.format(forcasFiltradoComUeV))
print('forcasFiltrado = {}'.format(forcasFiltrado))

print(np.array(forcasFiltrado))

ccs = []

for item in forcas:
    if item[1] == 'R':
        ccs.append(forcas.index(item))

a = np.delete(a, ccs, axis=1)
a = np.delete(a, ccs, axis=0)

print(a)

numpyListInverse = np.linalg.inv(a)

deslocamentosNumpy = np.matmul(numpyListInverse, forcasFiltrado)
deslocamentosArray = deslocamentosNumpy.tolist()

print(deslocamentosArray)

deslocamentosComUeV = []


for i in range(len(forcasFiltradoComUeV)):
    deslocamentosComUeV.append(('{}'.format(forcasFiltradoComUeV[i][0]), deslocamentosArray[i]))

print(deslocamentosComUeV)
