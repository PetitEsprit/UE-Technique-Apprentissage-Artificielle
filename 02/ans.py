import numpy as np
np.set_printoptions(threshold=None)
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

""" I.2 """
warnings.filterwarnings('ignore')
data = pd.read_csv('./villes.csv', sep=';')
X = data.iloc[:, 1:13].values
labels = data.iloc[:, 0].values
pca = PCA()
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))

"""
Le premier axe du ACP content plus de 70% de l'information
"""
print("PC1 value: ", pca.explained_variance_ratio_[0])

"""
- Pour le 1er axe: les features ont tous une importance autour de 0.27
- Pour le 2ème axe: le 3ème, le 4ème, le 9ème et le 10ème ont très peu d'influence tandis
que les autres featues en ont beaucoup plus
"""

print("Variance des axes: \n", pca.components_[:2].round(2))

plt.scatter(X_pca[:, 0], X_pca[:, 1])
for label, x, y in zip(labels, X_pca[:, 0], X_pca[:, 1]):
	plt.annotate(label, xy=(x, y), xytext=(-0.2, 0.2), textcoords='offset points')
plt.show()

""" I.3 """

warnings.filterwarnings('ignore')
data = pd.read_csv('./crimes.csv', sep=';')
X = data.iloc[:, 1:13].values
labels = data.iloc[:, 0].values
pca = PCA()
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))

"""
Les 2 premiers axes du ACP content plus de 70% de l'information
"""
print("PC1 value: ", sum(pca.explained_variance_ratio_[:2]))

"""
- Pour le 1er axe: les features ont tous une importance autour de 0.35
- Pour le 2ème axe: le 3ème a très peu d'influence tandis
que le 1er en a énormément par rapport aux autres
"""
print("Variance des axes: \n", pca.components_[:2].round(2))

plt.scatter(X_pca[:, 0], X_pca[:, 1])
for label, x, y in zip(labels, X_pca[:, 0], X_pca[:, 1]):
	plt.annotate(label, xy=(x, y), xytext=(-0.2, 0.2), textcoords='offset points')
plt.show()


""" II. """