# Principal Component Analisys

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from plot_dicision_region import plot_decision_regions

from sklearn.decomposition import PCA

data = load_wine()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)


# Извлечение данных методом главных компонент вручную

# Стандартизация данных
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.fit_transform(X_test)

# Построение ковариационной матрицы
cov_mat = np.cov(X_train_std.T)

# Получение собственных значений и собственных векторов 
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)


# Отрисовка долей объясненной дисперсии
# tot = sum(eigen_vals)
# var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
# cum_var_exp = np.cumsum(var_exp)

# plt.bar(range(1, 14), var_exp, align='center',
#         label='Individual explained variance')
# plt.step(range(1, 14), cum_var_exp, where='mid',
#          label='Cumulative explained variance')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal component index')
# plt.legend(loc='best')
# plt.tight_layout()
# # plt.savefig('figures/05_02.png', dpi=300)
# plt.show()


# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

w = np.hstack((eigen_pairs[0][1][:, np.newaxis], 
               eigen_pairs[1][1][:, np.newaxis], 
               eigen_pairs[2][1][:, np.newaxis],
               eigen_pairs[3][1][:, np.newaxis],
               eigen_pairs[4][1][:, np.newaxis],
               eigen_pairs[5][1][:, np.newaxis]))

X_train_manual_pca = X_train_std.dot(w)
X_test_manual_pca = X_test_std.dot(w)


# Обучение на базовом наборе и извлеченных вручную данных
model = LogisticRegression()
model.fit(X_train_std, y_train)
print(f"All {X_train_std.shape} data. Accuracy:", model.score(X_test_std, y_test))
model.fit(X_train_manual_pca, y_train)
print(f"Manual PCA {X_train_manual_pca.shape} data. Accuracy:", model.score(X_test_manual_pca, y_test))



pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

lr = LogisticRegression(random_state=1, solver='lbfgs')
lr = lr.fit(X_train_pca, y_train)
plot_decision_regions(X_test_pca, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


lr.fit(X_train_pca, y_train)
print(f"PCA {X_train_pca.shape} data. Accuracy:", lr.score(X_test_pca, y_test))

