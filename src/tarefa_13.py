# Detectar se é do bart ou do hommer as caracteristicas
from minisom import MiniSom
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pylab import pcolor, colorbar, plot

base = pd.read_csv(
    '/Users/es19237/Desktop/Python/Deep Learning/Mapas auto Organizaveis/files/personagens.csv')
X = base.iloc[:, 0:6].values
y = base.iloc[:, 6].values

normalizador = MinMaxScaler()
X = normalizador.fit_transform(X)

# montar o algoritimo
som = MiniSom(x=9, y=9, input_len=6, random_seed=0,
              learning_rate=0.5, sigma=1.0)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=500)

y[y == 'Bart'] = 0
y[y == 'Homer'] = 1

pcolor(som.distance_map().T)
colorbar()
# Bolinha Bart
# Quadrado Hommer
markers = ['o', 's']
color = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    # colocando o marcador
    plot(w[0]+0.5, w[1]+0.5, markers[y[i]], markerfacecolor='none',
         markersize=10, markeredgecolor=color[y[i]], markeredgewidth=2)

# monta o mapeamento e obtem os resultados para conferirmos se o que ta mostrando realmente confere com a base
mapeamento = som.win_map(X)
resultados = mapeamento[(7, 4)]
resultados = normalizador.inverse_transform(resultados)

classe = []
for i in range(len(base)):
    for j in range(len(resultados)):
        if ((base.iloc[i, 0] == resultados[j, 0]) and
           (base.iloc[i, 1] == resultados[j, 1]) and
           (base.iloc[i, 2] == resultados[j, 2]) and
           (base.iloc[i, 3] == resultados[j, 3]) and
           (base.iloc[i, 4] == resultados[j, 4]) and
           (base.iloc[i, 5] == resultados[j, 5])):
            classe.append(base.iloc[i, 6])
classe = np.asarray(classe)

resultados_final = np.column_stack((resultados, classe))
resultados_final = resultados_final[resultados_final[:, 4].argsort()]
