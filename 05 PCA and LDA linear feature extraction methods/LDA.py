# Linear Discriminant Analisys

import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects
from plot_dicision_region import plot_decision_regions


df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

df_wine.head()


X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test =     train_test_split(X, y, test_size=0.3, 
                     stratify=y,
                     random_state=0)


sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


np.set_printoptions(precision=4)

mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    # print(f'MV {label}: {mean_vecs[label - 1]}\n')


# Матрица внутриклассового разброса
d = 13 # number of features
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))  # scatter matrix for each class
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)  # make column vectors
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter                          

# print('Within-class scatter matrix: '
#       f'{S_W.shape[0]}x{S_W.shape[1]}')


print('Class label distribution:',  
      np.bincount(y_train)[1:])



# Масштабированная матрица внутриклассового разброса
d = 13  
S_W = np.zeros((d, d))
for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter
    
# print('Scaled within-class scatter matrix: '
#       f'{S_W.shape[0]}x{S_W.shape[1]}')



# Матрица внешнеклассового разброса
mean_overall = np.mean(X_train_std, axis=0)
mean_overall = mean_overall.reshape(d, 1)  # make column vector

d = 13  
S_B = np.zeros((d, d))

for i, mean_vec in enumerate(mean_vecs):
    n = X_train_std[y_train == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)  # make column vector
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

# print('Between-class scatter matrix: '
#       f'{S_B.shape[0]}x{S_B.shape[1]}')


# Нахождение собственных значений и собственных векторов
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# Объединение и сортировка по собственным значениям
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)


print('Eigenvalues in descending order:\n')
for eigen_val in eigen_pairs:
    print(round(eigen_val[0], 3))


tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)

# plt.bar(range(1, 14), discr, align='center', label='Individual discriminability')
# plt.step(range(1, 14), cum_discr, where='mid', label='Cumulative discriminability')
# plt.ylabel('Discriminability ratio')
# plt.xlabel('Linear discriminants')
# plt.ylim([-0.1, 1.1])
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()


# Формирования матрицы преобразования
w = np.hstack((eigen_pairs[0][1][:, np.newaxis].real,
              eigen_pairs[1][1][:, np.newaxis].real))
print('Matrix W:\n', w)


X_train_lda = X_train_std.dot(w)
X_test_lda = X_test_std.dot(w)
colors = ['r', 'b', 'g']
markers = ['o', 's', '^']

# for l, c, m in zip(np.unique(y_train), colors, markers):
#     plt.scatter(X_train_lda[y_train == l, 0],
#                 X_train_lda[y_train == l, 1] * (-1),
#                 c=c, label=f'Class {l}', marker=m)
# plt.xlabel('LD 1')
# plt.ylabel('LD 2')
# plt.legend(loc='lower right')
# plt.tight_layout()
# plt.show()


lr = LogisticRegression(random_state=1, solver='lbfgs')
lr.fit(X_train_lda, y_train)
print(f"\nLDA {X_train_lda.shape} data. Accuracy:", lr.score(X_test_lda, y_test))



# LDA via scikit-learn

lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)


lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
lr = lr.fit(X_train_lda, y_train)

plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()




X_test_lda = lda.transform(X_test_std)

plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()




# digits = load_digits()

# fig, ax = plt.subplots(1, 4)

# for i in range(4):
#     ax[i].imshow(digits.images[i], cmap='Greys')
# plt.show() 


# digits.data.shape


# y_digits = digits.target
# X_digits = digits.data



# tsne = TSNE(n_components=2,
#             init='pca',
#             random_state=123)
# X_digits_tsne = tsne.fit_transform(X_digits)



# def plot_projection(x, colors):
    
#     f = plt.figure(figsize=(8, 8))
#     ax = plt.subplot(aspect='equal')
#     for i in range(10):
#         plt.scatter(x[colors == i, 0],
#                     x[colors == i, 1])

#     for i in range(10):

#         xtext, ytext = np.median(x[colors == i, :], axis=0)
#         txt = ax.text(xtext, ytext, str(i), fontsize=24)
#         txt.set_path_effects([
#             PathEffects.Stroke(linewidth=5, foreground="w"),
#             PathEffects.Normal()])
        
# plot_projection(X_digits_tsne, y_digits)
# plt.show()
