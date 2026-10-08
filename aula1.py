import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# dados = {'id_familia':[1,2,3,4,5,6,7,8],
#          'tipo_casa': ["apartamento","casa","casa","kitnet","sobrado","apartamento","apartamento","casa"],
#          'tamanho_casa': ["pequeno","médio","grande","pequeno","médio","grande","grande","pequeno"],
#          'n_cartoes': [4,6,6,7,8,7,8,10],
#          'tamanho_familia': [2,2,4,4,5,5,6,6],
#          'renda_familiar':[14,16,14,17,18,21,17,25],
#          'num_automoveis': [1,2,2,1,3,2,1,3],
#          'distanciacentro': [22.2,3,31,2,1,12,2,4]}#km

# df = pd.DataFrame(dados)
# print(df)
# print()

#----------------------------------------------------------------------

# df = pd.read_csv('reservashotel.csv')

# df["tembebês"] = df["bebês"].apply(lambda x: "sim" if x > 0 else "não")
# df["estadias"] = df["estadias_em_noites_finalsemana"] + df["estadias_em_noites_semana"]

# print("Média:",df[df['tembebês'] == 'sim']["estadias"].mean())
# print("Mediana:",df[df['tembebês'] == 'sim']["estadias"].median())
# print("Moda:",df[df['tembebês'] == 'sim']["estadias"].mode())

# print("Média:",df[df['tembebês'] == 'não']["estadias"].mean())
# print("Mediana:",df[df['tembebês'] == 'não']["estadias"].median())
# print("Moda:",df[df['tembebês'] == 'não']["estadias"].mode())

# print(df.groupby('espaços_de_estacionamento_de_carro_requerido')['estadias'].mean())

#-----------------------------------Hands On-----------------------------------

# dffifa = pd.read_csv('https://www.harve.com.br/praticas/fifaplayers_pt.csv')
# dffifa.head()
# def converter_valores(valor):
#     valor = valor.upper().replace("€", "").replace("$", "").replace("£", "")
#     if "K" in valor:
#         return float(valor.replace("K", "")) * 1000
#     elif "M" in valor:
#         return float(valor.replace("M", "")) * 1000000
#     return float(valor)

# dffifa['remuneracao_corrigida'] = dffifa['remuneração'].apply(converter_valores)

# print("A média da pontuação geral é:", dffifa['pontuação geral'].mean())
# print("A mediana da pontuação geral é:", dffifa['pontuação geral'].median())
# print()
# print("A média da remuneração é:", round(dffifa['remuneracao_corrigida'].mean(), 2))
# print("A mediana da remuneração é:", round(dffifa['remuneracao_corrigida'].median(), 2))
# print("A moda da remuneração é:", round(dffifa['remuneracao_corrigida'].mode()[0], 2))

# histograph = sns.histplot(data=dffifa, x='remuneracao_corrigida', bins=30)
# histograph.set_title('Distribuição da Remuneração Corrigida dos Jogadores de FIFA')
# histograph.set_xlabel('Remuneração Corrigida')
# histograph.set_ylabel('Frequência')
# plt.show(block=True)

#----------------------------------------------------------------------

# dados = {'id_familia':[1,2,3,4,5,6,7,8],
#          'tipo_casa': ["apartamento","casa","casa","kitnet","sobrado","apartamento","apartamento","casa"],
#          'tamanho_casa': ["pequeno","médio","grande","pequeno","médio","grande","grande","pequeno"],
#          'n_cartoes': [4,6,6,7,8,7,8,10],
#          'tamanho_familia': [2,2,4,4,5,5,6,6],
#          'renda_familiar':[14,16,14,17,18,21,17,25],
#          'num_automoveis': [1,2,2,1,3,2,1,3],
#          'distanciacentro': [22.2,3,31,2,1,12,2,4]}#km

# df = pd.DataFrame(dados)

# print(df['distanciacentro'].describe())
# boxplot = sns.boxplot(data=df, x='distanciacentro')
# boxplot.set_title('Boxplot da Distância ao Centro')
# boxplot.set_xlabel('Distância ao Centro (km)')
# boxplot.set_ylabel('Valores')
# plt.show(block=True)

#-----------------------------------Hands On-----------------------------------

# dffifa = pd.read_csv('https://www.harve.com.br/praticas/fifaplayers_pt.csv')

# def converter_valores(valor):
#     valor = valor.upper().replace("€", "").replace("$", "").replace("£", "")
#     if "K" in valor:
#         return float(valor.replace("K", "")) * 1000
#     elif "M" in valor:
#         return float(valor.replace("M", "")) * 1000000
#     return float(valor)

# dffifa['remuneracao_corrigida'] = dffifa['remuneração'].apply(converter_valores)

# print(dffifa['remuneracao_corrigida'].quantile([0, 0.25, 0.5, 0.75, 1.0]))
# sns.boxplot(data=dffifa, x='remuneracao_corrigida')
# plt.title('Boxplot da Remuneração Corrigida dos Jogadores de FIFA')
# plt.xlabel('Remuneração Corrigida')
# plt.ylabel('Valores')
# plt.show(block=True)

#----------------------------------------------------------------------

# dfhotel = pd.read_csv('reservashotel.csv')
# dfhotel["estadias"] = dfhotel["estadias_em_noites_finalsemana"] + dfhotel["estadias_em_noites_semana"]
# dfhotel["mean"] = dfhotel.groupby("data_de_chegada_mês")["estadias"].mean()
# dfhotel["std"] = dfhotel.groupby("data_de_chegada_mês")["estadias"].std()

# agrupado = dfhotel.groupby("data_de_chegada_mês")["estadias"].agg(['mean', 'std'])
# agrupado['cv'] = agrupado['std'] / agrupado['mean']
# print(agrupado)

#-----------------------------------Hands On-----------------------------------

# dffifa = pd.read_csv('https://www.harve.com.br/praticas/fifaplayers_pt.csv')

# def converter_valores(valor):
#     valor = valor.upper().replace("€", "").replace("$", "").replace("£", "")
#     if "K" in valor:
#         return float(valor.replace("K", "")) * 1000
#     elif "M" in valor:
#         return float(valor.replace("M", "")) * 1000000
#     return float(valor)

# dffifa['remuneracao_corrigida'] = dffifa['remuneração'].apply(converter_valores)

# amp = dffifa['remuneracao_corrigida'].max() - dffifa['remuneracao_corrigida'].min()
# print("Amplitude da remuneração:", amp)
# print("Desvio padrão da remuneração:", round(dffifa['remuneracao_corrigida'].std(), 2))

lista = np.array([1,1,1,1,2,2,2,2])
print(lista.std())