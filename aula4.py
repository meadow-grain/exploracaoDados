import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine, load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report, ConfusionMatrixDisplay

plt.rcParams.update({"figure.figsize": (7, 4.5), "axes.grid": True, "grid.alpha": 0.3, "font.size": 11})
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("Ambiente pronto!")

# Carrega o dataset Iris retornando apenas os atributos (I) e as classes (iy)
I, iy = load_iris(return_X_y=True)

# Recupera os nomes das espécies (setosa, versicolor, virginica) do dataset original
nomes = load_iris().target_names

# Define o tamanho da janela do gráfico para 7 polegadas de largura por 5 de altura
plt.figure(figsize=(7, 5))

# Percorre cada classe única (índice c e o nome da espécie) para plotar individualmente
for c, nome in enumerate(nomes):
    # Plota os pontos correspondentes à espécie atual usando a coluna 2 (comprimento) e 3 (largura) da pétala
    plt.scatter(I[iy == c, 2], I[iy == c, 3], label=nome, s=45)

# Define o rótulo do eixo horizontal (X) do gráfico
plt.xlabel("comprimento da pétala (cm)")

# Define o rótulo do eixo vertical (Y) do gráfico
plt.ylabel("largura da pétala (cm)")

# Define o título principal que descreve o gráfico
plt.title("Iris: as espécies no plano das pétalas")

# Insere a legenda com o nome das espécies no gráfico
plt.legend()

# Renderiza e exibe o gráfico finalizado na tela
plt.show()

# Carrega o dataset de vinhos retornando os atributos (W) e as classes correspondentes (wy)
W, wy = load_wine(return_X_y=True)

# Calcula a acurácia média por validação cruzada (com o 'cv' definido anteriormente)
# de um classificador k-NN com k=5 vizinhos, usando os dados originais (sem padronização)
sem = cross_val_score(KNeighborsClassifier(5), W, wy, cv=cv).mean()

# Cria um pipeline que primeiro padroniza os dados (StandardScaler) e depois aplica o k-NN (k=5),
# calculando em seguida a acurácia média por validação cruzada (com padronização)
com = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(5)), W, wy, cv=cv).mean()

# Exibe o resultado formatado com 3 casas decimais para o modelo sem padronização
print(f"k-NN sem padronização (CV): {sem:.3f}")

# Exibe o resultado formatado com 3 casas decimais para o modelo com padronização
print(f"k-NN com padronização (CV): {com:.3f}")