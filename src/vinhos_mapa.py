from minisom import MiniSom
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pylab import pcolor, colorbar, plot

base = pd.read_csv(
    '/Users/es19237/Desktop/Python/Deep Learning/Mapas auto Organizaveis/files/wines.csv')

x = base.iloc[:, 1:14].values
y = base.iloc[:, 0].values
# Normalizando para deixar todos os valores na mesma escala,
# essa escala pode ser definida pelo parametro feature_range que vai
# dentro do MinMaxScaler(feature_range = (0,1))
normalizador = MinMaxScaler()

x = normalizador.fit_transform(x)
'''
    o x é as linhas e o y é as colunas
    Para acharmos os valores de 8 ali é usado a formula: 5 raiz N
    onde N é a quantidade de linhas que temos nos dados: 5xraiz178 = 65,65 
    que fazendo 8x8 = 64 que da quase isso

    input_len =  é a quantidade de colunas.
    sigma = é o raio o alcance que vai ser feito a atualização dos neuronios.
    learning_rate = taxa de atualização dos pesos.
    random_seed = para termos sempre o mesmo resultado a cada execução.
'''
som = MiniSom(x=8, y=8, input_len=13, sigma=1,
              learning_rate=0.5, random_seed=2)
# Gerar os pontos aleatórios para posteior fazer o calculo da distancia
som.random_weights_init(x)
# realizar o treinamento
som.train(data=x, num_iteration=100000)

# monta um grafico para poder visualizar o mapa
# Aqui ele ta executando o mean inter neuros distance, ou seja quanto um neuronio é parecido dos sesu vizinhos
pcolor(som.distance_map().T)
# bara de cor, quanto mais escurro mais parecido com seus vizinhos ele é
colorbar()

## Mais visualizações dos dados ##

# Pegar o melhor parametro
w = som.winner(x[1])

# Criando os marcadores
markers = ['o', 's', 'D']
color = ['r', 'g', 'b']

# transforma para iniciar com 0 pois senao nao se acha nos marcadores criado acima
#y[y==1] = 0
#y[y==2] = 1
#y[y==3] = 2

# Acharmos os melhores parametros aqui termos as informaçõs de linha e valores
for i, X in enumerate(x):
    # print(i)
    # print(X)
    w = som.winner(X)
    # print(w)
    # colocando o marcador
    plot(w[0]+0.5, w[1]+0.5, markers[y[i]], markerfacecolor='none',
         markersize=10, markeredgecolor=color[y[i]], markeredgewidth=2)
