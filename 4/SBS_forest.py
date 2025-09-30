import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import StringIO
from itertools import combinations
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_wine


dataset = load_wine(as_frame=True)
data = dataset.data
target = dataset.target


X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=0, stratify=target)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_std = scaler.transform(X_train)
X_test_std = scaler.transform(X_test)


feat_labels = data.columns

forest = RandomForestClassifier(n_estimators=500, random_state=1)

forest.fit(X_train, y_train)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

# for f in range(X_train.shape[1]):
#     print("%2d) %-*s %f" % (f + 1, 30, 
#                             feat_labels[indices[f]], 
#                             importances[indices[f]]))

# plt.title('Feature importance')
# plt.bar(range(X_train.shape[1]), 
#         importances[indices],
#         align='center')

# plt.xticks(range(X_train.shape[1]), 
#            feat_labels[indices], rotation=90)
# plt.xlim([-1, X_train.shape[1]])
# plt.tight_layout()
# plt.show()

# Выбор наиболее релевантных параметров
sfm = SelectFromModel(forest, threshold=0.1, prefit=True)
X_selected = sfm.transform(X_train)
X_selected_test = sfm.transform(X_test)

# print('Number of features that meet this threshold criterion:', 
#       X_selected.shape[1])

# for f in range(X_selected.shape[1]):
#     print("%2d) %-*s %f" % (f + 1, 30, 
#                             feat_labels[indices[f]], 
#                             importances[indices[f]]))
    

scaler.fit(X_selected)
X_train_select = pd.DataFrame(scaler.transform(X_selected), columns=feat_labels[indices[:5]])
X_test_select = pd.DataFrame(scaler.transform(X_selected_test), columns=feat_labels[indices[:5]])


model = LogisticRegression(solver="lbfgs", max_iter=100)

model.fit(X_train_std, y_train)
print(f"All {X_train_std.shape} data. Accuracy:", model.score(X_test_std, y_test))

model.fit(X_train_select, y_train)
print(f"Select {X_train_select.shape} data. Accuracy:", model.score(X_test_select, y_test))