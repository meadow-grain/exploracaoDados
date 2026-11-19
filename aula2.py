import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris, load_digits, load_breast_cancer, make_classification
from sklearn.model_selection import (train_test_split, cross_val_score, cross_validate,
                                     KFold, StratifiedKFold, learning_curve)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils import resample
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,
                             accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, ConfusionMatrixDisplay)

plt.rcParams.update({"figure.figsize": (7, 4.5), "axes.grid": True, "grid.alpha": 0.3, "font.size": 11})
pd.set_option("display.precision", 3)
print("Ambiente pronto!")

# try:
#     titanic = sns.load_dataset("titanic")
# except Exception:
#     titanic = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")

# df = titanic[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]].copy()
# # print("Formato (linhas, colunas):", df.shape)
# # print(df.head())

# faltantes = df.isna().sum()
# faltantes = faltantes[faltantes > 0].sort_values()
# faltantes.plot(kind="barh", color="tab:red", title="Valores faltantes por coluna")
# plt.xlabel("quantidade de linhas")

# print((df.isna().mean() * 100).round(1)[lambda s: s > 0].to_string(), "  (% de linhas)")

# print("Idades acima de 100 anos:", (df.age > 100).sum())
# print(df.age.describe().round(1).to_string())

# fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
# sns.boxplot(x=df.age, ax=ax[0], color="lightsteelblue"); ax[0].set_title("Idade")
# sns.boxplot(x=df.fare, ax=ax[1], color="lightsalmon"); ax[1].set_title("Tarifa (fare): muitos valores altos")

# imoveis = pd.DataFrame({"imovel": list("ABCD"), "real": [200, 300, 250, 400], "previsto": [220, 280, 260, 350]})
# imoveis["erro"] = imoveis.real - imoveis.previsto
# imoveis["erro_abs"] = imoveis.erro.abs()
# imoveis["erro_quad"] = imoveis.erro ** 2
# imoveis["erro_%"] = (imoveis.erro_abs / imoveis.real * 100).round(1)
# print(imoveis)

# fig, ax = plt.subplots()
# x = np.arange(len(imoveis))
# ax.vlines(x, imoveis.previsto, imoveis.real, color="gray", linestyle="--", label="erro")
# ax.scatter(x, imoveis.real, s=90, color="tab:blue", label="real", zorder=3)
# ax.scatter(x, imoveis.previsto, s=90, color="tab:orange", marker="X", label="previsto", zorder=3)
# ax.set_xticks(x); ax.set_xticklabels(imoveis.imovel)
# ax.set_ylabel("preço (R$ mil)"); ax.set_title("Erro de cada previsão"); ax.legend()

# n = len(imoveis)
# mae  = imoveis.erro_abs.sum() / n
# mape = (imoveis.erro_abs / imoveis.real).sum() / n
# mse  = imoveis.erro_quad.sum() / n

# print(f"MAE  (na mão)  = {mae:.1f}   | sklearn: {mean_absolute_error(imoveis.real, imoveis.previsto):.1f}   -> erro médio de R$ {mae:.0f} mil")
# print(f"MAPE (na mão)  = {mape:.3f} | sklearn: {mean_absolute_percentage_error(imoveis.real, imoveis.previsto):.3f} -> erra {mape*100:.1f}% do valor real")
# print(f"MSE  (na mão)  = {mse:.1f} | sklearn: {mean_squared_error(imoveis.real, imoveis.previsto):.1f}  -> unidade em (R$ mil)²")
# print(f"RMSE = raiz do MSE = {mse**0.5:.1f}  -> volta para R${mse**0.5:.0f} mil")

# com_outlier = imoveis[["real", "previsto"]].copy()
# com_outlier.loc[3, "previsto"] = 200   # erro de 200 no imóvel D

# comparacao = pd.DataFrame({
#     "MAE":  [mean_absolute_error(imoveis.real, imoveis.previsto), mean_absolute_error(com_outlier.real, com_outlier.previsto)],
#     "MSE":  [mean_squared_error(imoveis.real, imoveis.previsto),  mean_squared_error(com_outlier.real, com_outlier.previsto)],
# }, index=["erro de 50 no imóvel D", "erro de 200 no imóvel D"])
# comparacao["MAE cresceu"] = (comparacao.MAE / comparacao.MAE.iloc[0]).round(1).astype(str) + "x"
# comparacao["MSE cresceu"] = (comparacao.MSE / comparacao.MSE.iloc[0]).round(1).astype(str) + "x"
# # print(comparacao)

# plt.show()

#--------------------------

# real = np.array([10, 20, 30, 40, 50])
# previsto = np.array([12, 18, 33, 35, 52])

# erro = real - previsto
# print(erro)
# print("MAE :", np.abs(erro).mean())
# print("MAPE:", round((np.abs(erro) / real).mean(), 4))
# print("MSE :", (erro ** 2).mean())
# print("RMSE :", (erro ** 2).mean() ** 0.5)

#--------------------------

# y_true = np.array([1] * 10 + [0] * 990)       # 1 = fraude, 0 = normal
# modelo_bobo = np.zeros(1000, dtype=int)         # sempre diz "normal"
# print("Acurácia:", accuracy_score(y_true, modelo_bobo))
# print("Recall  :", recall_score(y_true, modelo_bobo), " -> não detectou nenhuma fraude!")

# y_pred = np.array([1] * 8 + [0] * 2 + [1] * 20 + [0] * 970)

# # o scikit-learn devolve [[TN, FP], [FN, TP]]  (classe 0 primeiro)
# tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
# print(f"TP={tp}  FN={fn}  FP={fp}  TN={tn}")

# # desenhando no formato do slide 96 (classe 1 primeiro)
# matriz = np.array([[tp, fn], [fp, tn]])
# rotulos = np.array([[f"TP\n{tp}\n(fraude detectada)", f"FN\n{fn}\n(fraude que passou)"],
#                     [f"FP\n{fp}\n(alarme falso)",     f"TN\n{tn}\n(normal confirmada)"]])
# fig, ax = plt.subplots(figsize=(6.5, 4.5))
# cores = np.array([[1, -1], [-1, 1]])   # verde = acerto (TP, TN), vermelho = erro (FN, FP)
# sns.heatmap(cores, annot=rotulos, fmt="", cmap="RdYlGn", vmin=-2.2, vmax=2.2, cbar=False, ax=ax, linewidths=2, linecolor="white",
#             xticklabels=["Classe 1 (fraude)", "Classe 0 (normal)"], yticklabels=["Classe 1 (fraude)", "Classe 0 (normal)"])
# ax.set_xlabel("Classe PREVISTA"); ax.set_ylabel("Classe REAL"); ax.set_title("Matriz de confusão")
# ax.grid(False); plt.show()

# acuracia  = (tp + tn) / (tp + tn + fp + fn)
# precision = tp / (tp + fp)
# recall    = tp / (tp + fn)
# fpr       = fp / (fp + tn)
# f1        = 2 * precision * recall / (precision + recall)

# resultado = pd.DataFrame({
#     "na mão": [acuracia, precision, recall, fpr, f1],
#     "scikit-learn": [accuracy_score(y_true, y_pred), precision_score(y_true, y_pred),
#                      recall_score(y_true, y_pred), fp / (fp + tn), f1_score(y_true, y_pred)],
# }, index=["Acurácia", "Precision", "Recall", "FPR", "F1"])
# print(resultado)

# print(classification_report(y_true, y_pred, target_names=["normal", "fraude"]))

#--------------------------

# X_s, y_s = make_classification(n_samples=5000, n_features=10, n_informative=4, weights=[0.95, 0.05],
#                                flip_y=0.02, class_sep=0.8, random_state=42)
# Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(X_s, y_s, test_size=0.2, stratify=y_s, random_state=42)

# prob = LogisticRegression(max_iter=1000).fit(Xs_tr, ys_tr).predict_proba(Xs_te)[:, 1]
# limiares = np.arange(0.02, 0.32, 0.01)
# prec = [precision_score(ys_te, prob >= t, zero_division=0) for t in limiares]
# rec  = [recall_score(ys_te, prob >= t) for t in limiares]

# plt.plot(limiares, prec, label="precision", color="tab:blue")
# plt.plot(limiares, rec,  label="recall",    color="tab:red")
# plt.xlabel("limiar de decisão (probabilidade mínima para dizer 'positivo')"); plt.ylabel("valor")
# plt.title("Baixar o limiar: mais recall, menos precision"); plt.legend(); plt.show()

#--------------------------

TP, FN, FP, TN = 40, 10, 5, 45
acc = (TP + TN) / (TP + TN + FP + FN);
pr = TP / (TP + FP); rc = TP / (TP + FN); fp_rate = FP / (FP + TN)

print(f"acurácia={acc:.3f}  precision={pr:.3f}  recall={rc:.3f}  FPR={fp_rate:.3f}  F1={2*pr*rc/(pr+rc):.3f}")