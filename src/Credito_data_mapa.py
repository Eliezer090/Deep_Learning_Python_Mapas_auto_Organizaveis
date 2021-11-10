# Algotimo que verifica clientes que podem vir a tentar uma fraude.
# Coluna default valor 0 sao os que ganharam emprestimos
from minisom import MiniSom
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pylab import pcolor, colorbar, plot

base = pd.read_csv(
    '/Users/es19237/Desktop/Python/Deep Learning/Mapas auto Organizaveis/files/credit_data.csv')

base = base.dropna()
base.loc[base.age < 0, 'age'] = 40.80

X = base.iloc[:, 0:4].values

y = base.iloc[:, 4].values

normalizador = MinMaxScaler()
X = normalizador.fit_transform(X)

som = MiniSom(x=15, y=15, input_len=4, random_seed=0)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

# registros mais diferentes quer dizer que tem mais probabilidade de combater fraudes(Perto de 1)
pcolor(som.distance_map().T)
colorbar()

markers = ['o', 's']
color = ['r', 'g']
# Bolinha que tiveram o credito aprovado
# Quadrado que nao tiveram o credito aprovado
for i, x in enumerate(X):
    w = som.winner(x)
    # colocando o marcador
    plot(w[0]+0.5, w[1]+0.5, markers[y[i]], markerfacecolor='none',
         markersize=10, markeredgecolor=color[y[i]], markeredgewidth=2)

# Capturando os suspeitos para conseguirmos identifica-los
mapeamento = som.win_map(X)
# Aqui pega só 2 posições que estavam mais perto de 1 mas pode ser pego mais de 2
suspeitos = np.concatenate((mapeamento[(4, 5)], mapeamento[(6, 11)]), axis=0)
suspeitos = normalizador.inverse_transform(suspeitos)

# Pega os que tiveram o credito aprovado e verifica se os suspeitos estão entre eles
classe = []

for i in range(len(base)):
    for j in range(len(suspeitos)):
        # COnverte para inteiro
        if base.iloc[i, 0] == int(round(suspeitos[j, 0])):
            classe.append(base.iloc[i, 4])

classe = np.array(classe)

# concatena os 2
suspeito_final = np.column_stack((suspeitos, classe))
# Ordena para falicitar visulaização
suspeito_final = suspeito_final[suspeito_final[:, 4].argsort()]
