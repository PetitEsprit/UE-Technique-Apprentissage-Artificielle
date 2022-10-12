import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


np.set_printoptions(threshold=None)
warnings.filterwarnings('ignore')

""" Loading """
df = pd.read_csv('credit_scoring.csv', sep=';')
data = np.array_split(df.values, [-1, ], axis=1)

""" Showing """
#print("Taille d'échantillon: ", data[0].shape)
#plt.hist(data[1])
#plt.show()



""" Build data_train and data_test"""
# Question: should stratify ?
data_train,data_test,status_train,status_test = train_test_split(data[0], data[1])
data_train_norm,data_test_norm = StandardScaler().fit_transform(data_train), StandardScaler().fit_transform(data_test)
""" Knn test with k=5"""
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(data_train, status_train)
print("Accuracy Score Knn(k = 5): ", accuracy_score(neigh.predict(data_test), status_test))
""" CART Tree"""
cart = tree.DecisionTreeClassifier()
cart.fit(data_train, status_train)
print("Accuracy Score CART Tree: ", accuracy_score(cart.predict(data_test), status_test))
""" Knn test with k=5 normalized"""
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(data_train_norm, status_train)
print("Accuracy Score Knn(k = 5) normalized: ", accuracy_score(neigh.predict(data_test_norm), status_test))
""" CART Tree normalized"""
cart = tree.DecisionTreeClassifier()
cart.fit(data_train_norm, status_train)
print("Accuracy Score CART Tree normalized: ", accuracy_score(cart.predict(data_test_norm), status_test))


""" USE PCA """
#sur quel proportion appliquer le ACP(sur des data tests ou le tout) ?
# sur les datas normées, non normées ?
k = 5
pca = PCA()
pca.fit(data[0])
data_test_reduc = pca.transform(data_test_norm)
data_train_reduc = pca.transform(data_train_norm)
data_test_reduc = np.concatenate([data_test_reduc[:k], data_test])
....
print(data_test_reduc)


