import streamlit as st
from sympy import pi

rows = []

xp1s, yp1s, zp1s, xp2s, yp2s, zp2s = [], [], [], [], [], []

points = []

Es, Ds, Ls, As = [], [], [], []
cteLs, cteMs, cteNs = [], [], []

elements = []

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
      Ei = st.number_input(
          "E (MPa)",
          key='moduloE_{}'.format(i)
      )
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
  
lerInputsManuais()
st.write(elements)