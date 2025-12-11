from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score

data = load_iris()

# scaler = StandardScaler()
# scaler.fit(data.data)
# data_std = scaler.transform(data.data)

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=1)

model_tree = DecisionTreeClassifier(criterion="gini", max_depth=4)
model_tree.fit(X_train, y_train)
# feature_names = ['Sepal length', 'Sepal width',
#                  'Petal length', 'Petal width']
# plot_tree(model_tree, feature_names=feature_names, filled=True)

y_pred = model_tree.predict(X_test)
plt.show()

print("Ошибочно классифицированы: %d" % (y_test != y_pred).sum())
print(f"Accuracy tree: {model_tree.score(X_test, y_test):.3f}")
print(f"F1_score tree: {f1_score(y_test, y_pred, average=None)}")



forest = RandomForestClassifier(n_estimators=25, criterion="gini", random_state=1)
forest.fit(X_train, y_train)

y_pred_forest = forest.predict(X_test)

print("Ошибочно классифицированы: %d" % (y_test != y_pred_forest).sum())
print(f"Accuracy forest: {forest.score(X_test, y_test):.3f}")
print(f"F1_score forest: {f1_score(y_test, y_pred_forest, average=None)}")