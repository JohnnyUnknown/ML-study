import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC



df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases'
                 '/breast-cancer-wisconsin/wdbc.data', header=None)
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)
le.classes_


le.transform(['M', 'B'])


X_train, X_test, y_train, y_test =     train_test_split(X, y, 
                     test_size=0.20,
                     stratify=y,
                     random_state=1)

pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state=1))

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'svc__C': param_range, 
               'svc__kernel': ['linear']},
              {'svc__C': param_range, 
               'svc__gamma': param_range, 
               'svc__kernel': ['rbf']}]

rs = RandomizedSearchCV(estimator=pipe_svc, 
                  param_distributions=param_grid, 
                  scoring='accuracy', 
                  refit=True,
                  n_iter=20,
                  cv=10,
                  random_state=1)
rs = rs.fit(X_train, y_train)
print(rs.best_score_)
print(rs.best_params_)