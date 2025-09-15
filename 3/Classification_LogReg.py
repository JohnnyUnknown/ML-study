from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from plot_decision_regions import plot_decision_regions

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Функция для вычисления стреднего (sc.mean_) и стандартого отклонения (sc.transform()) выборки
sc = StandardScaler()
sc.fit(X_train)

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

lr = LogisticRegression(C=100.0, solver='lbfgs', multi_class='multinomial')
lr.fit(X_train_std, y_train)

y_pred = lr.predict(X_test_std)
print("Ошибочно классифицированы: %d" % (y_test != y_pred).sum())

# print('Accuracy LogReg: %.3f' % accuracy_score(y_test, y_pred))
print("Accuracy LogReg: %.3f" % lr.score(X_test_std, y_test))

# Отрисовка областей решений модели
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
# plot_decision_regions(X=X_combined_std, y=y_combined,
#                     classifier=lr, test_idx=range(105, 150))


# Возвращаемые оценки для всех классов упорядочены по меткам классов.
print(lr.predict_proba(X_test_std[:3, :]))

from sklearn.metrics import f1_score
print(f"F1_score LogReg: {f1_score(y_test, y_pred, average=None)}")