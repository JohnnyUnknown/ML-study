import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score
# from matplotlib import pyplot as plt
# from ucimlrepo import fetch_ucirepo 
  
# spambase = fetch_ucirepo(id=94) 
# X = spambase.data.features 
# y = spambase.data.targets.values

columns = ["word_freq_make", 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our',
    'word_freq_over', 'word_freq_remove', "word_freq_internet", "word_freq_order", "word_freq_mail",
    'word_freq_receive', 'word_freq_will', "word_freq_people", "word_freq_report", "word_freq_addresses", 
    "word_freq_free", "word_freq_business", "word_freq_email", "word_freq_you", "word_freq_credit",
    "word_freq_your", "word_freq_font", "word_freq_000", "word_freq_money", "word_freq_hp", "word_freq_hpl",
    "word_freq_george", "word_freq_650", "word_freq_lab", "word_freq_labs", "word_freq_telnet", "word_freq_857", "word_freq_data",
    "word_freq_415", "word_freq_85", "word_freq_technology", "word_freq_1999", "word_freq_parts", "word_freq_pm",
    "word_freq_direct", "word_freq_cs", "word_freq_meeting", "word_freq_original", "word_freq_project", 
    "word_freq_re", "word_freq_edu", "word_freq_table", "word_freq_conference", "char_freq_;", "char_freq_(", 
    "char_freq_[", "char_freq_!", "char_freq_$", "char_freq_#", "capital_run_length_average", 
    "capital_run_length_longest", "capital_run_length_total", "spam"]
data = pd.read_csv("spambase\\spambase.data", names=columns)
X = data.loc[:, :"capital_run_length_total"]
y = data["spam"]

# print(pd.isnull(X).sum())

scaler = StandardScaler()
scaler.fit(X)
X_std = pd.DataFrame(scaler.transform(X), columns=columns[:-1]).round(decimals=3)


# print(X.describe().transpose().round(2))
# print(X_std.describe().transpose().round(2))

# print(y.value_counts())
# # 0        2788
# # 1        1813

# # Отсеивание параметров с малой разностью средних по таргету (маленькая разность = слабое влияние) (не улучшило модель)
# mean_0 = X_std.loc[y == 0].mean()
# mean_1 = X_std.loc[y == 1].mean()
# groupped = pd.DataFrame({
#     'mean 0': mean_0[1:],
#     'mean 1': mean_1[1:],
#     'diff': abs(mean_0 - mean_1),
# }).sort_values(by='diff', ascending=False)
# good_names = groupped.index[1:-5]
# X_std = X_std.loc[:, good_names]
# print(groupped)



X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=1, stratify=y)

forest = RandomForestClassifier(n_estimators=120, criterion="entropy", random_state=1)
log_reg = LogisticRegression(penalty="l2", C=1.0, solver="liblinear", random_state=1)
svc = SVC(C=1.0, kernel="rbf", random_state=1)
knn = KNeighborsClassifier(n_neighbors=3, metric="minkowski")


# Подбор наиболее значимых параметров
# forest = RandomForestClassifier(n_estimators=500, random_state=1)
# forest.fit(X_train, y_train)
# importances = forest.feature_importances_
# indices = np.argsort(importances)[::-1]

# X_train = X_train.loc[:, data.columns[indices[:24]]]
# X_test = X_test.loc[:, data.columns[indices[:24]]]



forest.fit(X_train, y_train)
forest_pred = forest.predict(X_test)
print(
    f'\naccuracy forest = {accuracy_score(y_test, forest_pred):.2f}'
    f'\nrecall forest = {recall_score(y_test, forest_pred):.2f}'
    f'\nf1-score forest = {f1_score(y_test, forest_pred):.2f}\n'
)

log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)
print(
    f'accuracy log_reg = {accuracy_score(y_test, log_reg_pred):.2f}'
    f'\nrecall log_reg = {recall_score(y_test, log_reg_pred):.2f}'
    f'\nf1-score log_reg = {f1_score(y_test, log_reg_pred):.2f}\n'
)
# print(log_reg.intercept_)
# print(log_reg.coef_)

svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)
print(
    f'accuracy svc = {accuracy_score(y_test, svc_pred):.2f}'
    f'\nrecall svc = {recall_score(y_test, svc_pred):.2f}'
    f'\nf1-score svc = {f1_score(y_test, svc_pred):.2f}\n'
)

knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
print(
    f'accuracy knn = {accuracy_score(y_test, knn_pred):.2f}'
    f'\nrecall knn = {recall_score(y_test, knn_pred):.2f}'
    f'\nf1-score knn = {f1_score(y_test, knn_pred):.2f}\n'
)


# Исходные данные ----------------------------------------------------

# accuracy forest = 0.95
# recall forest = 0.93
# f1-score forest = 0.94

# accuracy log_reg = 0.93
# recall log_reg = 0.91
# f1-score log_reg = 0.91

# accuracy svc = 0.70
# recall svc = 0.47
# f1-score svc = 0.55

# accuracy knn = 0.81
# recall knn = 0.75
# f1-score knn = 0.76


# После стандартизации ----------------------------------------------------

# accuracy forest = 0.96
# recall forest = 0.93
# f1-score forest = 0.94

# accuracy log_reg = 0.93
# recall log_reg = 0.90
# f1-score log_reg = 0.91

# accuracy svc = 0.94
# recall svc = 0.90
# f1-score svc = 0.92

# accuracy knn = 0.92
# recall knn = 0.89
# f1-score knn = 0.89

# -----------------------------------------------------