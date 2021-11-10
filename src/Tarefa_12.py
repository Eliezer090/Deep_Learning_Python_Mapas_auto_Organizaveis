from minisom import MiniSom
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pylab import pcolor, colorbar, plot

base = pd.read_csv(
    '/Users/es19237/Desktop/Python/Deep Learning/Mapas auto Organizaveis/files/entradas_breast.csv')
X = base.iloc[:, 0:30].values

saidas = pd.read_csv(
    '/Users/es19237/Desktop/Python/Deep Learning/Mapas auto Organizaveis/files/saidas_breast.csv')
y = saidas.iloc[:, 0].values

# Normalizando
normalizador = MinMaxScaler()
X = normalizador.fit_transform(X)
# Parametrizacao
som = MiniSom(x=11, y=11, input_len=30, sigma=5,
              learning_rate=0.1, random_seed=0)
som.random_weights_init(X)
som.train(data=X, num_iteration=10000)
# montando a viasualização
pcolor(som.distance_map().T)
colorbar()

markers = ['o', 's']
color = ['r', 'g']

# conseguirmos visualizar os melhores parametros aqui termos as informaçõs de linha e valores
for i, x in enumerate(X):
    # print(i)
    # print(X)
    w = som.winner(x)
    # print(w)
    # colocando o marcador
    plot(w[0]+0.5, w[1]+0.5, markers[y[i]], markerfacecolor='none',
         markersize=10, markeredgecolor=color[y[i]], markeredgewidth=2)
