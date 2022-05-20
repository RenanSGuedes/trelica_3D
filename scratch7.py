import copy

pontoNo, elementsComNos = [], []

elements = [
    [[0, 0, 0], [12, 8, 0]],
    [[12, 8, 0], [12, 0, 0]],
]
listaP = []

for item in elements:
    for j in range(2):
        listaP.append(item[j])

for i in range(len(listaP)):
    if listaP[i] not in pontoNo:
        pontoNo.append(listaP[i])

print("pontoNo", pontoNo)

pontoNoAgrupado = []

for i in range(len(pontoNo)):
    pontoNoAgrupado.append([pontoNo[i], i + 1])

print("pontoNoAgrupado", pontoNoAgrupado)

indicesElementos = copy.deepcopy(elements)

for i in range(len(elements)):
    for j in range(2):
        for k in range(len(pontoNoAgrupado)):
            if elements[i][j] == pontoNoAgrupado[k][0]:
                indicesElementos[i][j] = pontoNoAgrupado[k][1]

print("elements", elements)
print("indicesElementos", indicesElementos)
print("listaP", listaP)

print("elementsComNos", elementsComNos)

for i in range(len(indicesElementos)):
    elementsComNos.append([[indicesElementos[i][0], elements[i][0][0], elements[i][0][1]],
                           [indicesElementos[i][1], elements[i][1][0], elements[i][1][1]]])

print("elementsComNos", elementsComNos)
