import streamlit as st
import pandas as pd

rows = []

xp1s, yp1s, zp1s, xp2s, yp2s, zp2s = [], [], [], [], [], []

points = []

Es, Ds, Ls, As = [], [], [], []
cteLs, cteMs, cteNs = [], [], []

elements = []

def lerArquivo():
	uploaded_file = st.file_uploader("Escolha um arquivo")

	@st.cache(suppress_st_warning=True)
	def readCSVFile(file):
		if file is not None:
			return pd.read_csv(file)
		else:
			return st.write('Hello')

	pandasToPythonList = readCSVFile(uploaded_file).values.tolist()

	for row in pandasToPythonList:
		rows.append(row)

	for i in range(int(len(rows))):
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

lerArquivo()

st.write(xp1s)
st.write(As)
st.write(cteLs)