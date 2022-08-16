import random

print(random.random())

linha = []

for i in range(100):
    linha.append([])

for i in range(100):
    for j in range(100):
        linha[j].append(random.random())
        
print(linha)