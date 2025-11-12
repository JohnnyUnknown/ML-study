import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Исходные данные
X = np.array([[1], [2], [3], [4], [5], [6]]).astype(float)  # признаки
y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])                # таргет

# Шаг 0: Инициализация F0 = среднее(y)
F_prev = np.full_like(y, np.mean(y))  # [3.5, 3.5, 3.5, 3.5, 3.5, 3.5]
print("F0 =", F_prev[0])

# Шаг 1: Вычисляем остатки (антиградиенты для MSE)
residuals = y - F_prev
print("Остатки (r):", residuals)

# Шаг 2: Обучаем дерево на остатках (ограничим глубину)
tree = DecisionTreeRegressor(max_depth=2, random_state=0)
tree.fit(X, residuals)

# Шаг 3: Получаем предсказания дерева (это НЕ γ!)
# tree.predict(X) возвращает средние остатки в листьях — то есть совпадает с γ при MSE.
# Но в общем случае (например, при LogLoss) tree.predict — лишь приближение, а γ нужно вычислять отдельно через оптимизацию.
raw_tree_pred = tree.predict(X)
print("Сырые предсказания дерева (не γ!):", raw_tree_pred)

# Шаг 4: Получаем номера листьев для каждого объекта
leaf_ids = tree.apply(X)  # индексы листьев
print("Лист для каждого объекта:", leaf_ids)

# Шаг 5: Для каждого уникального листа вычисляем оптимальное γ (при MSE = среднее остатков в листе)
gamma = {}
for leaf_id in np.unique(leaf_ids):
    # какие объекты попали в этот лист?
    mask = (leaf_ids == leaf_id)
    # γ = среднее значение остатков в этом листе
    gamma[leaf_id] = residuals[mask].mean()
    print(f"Лист {leaf_id}: объекты {np.where(mask)[0]}, остатки {residuals[mask]}, γ = {gamma[leaf_id]:.3f}")

# Шаг 6: Применяем γ (а не сырые предсказания дерева!)
gamma_applied = np.array([gamma[leaf_id] for leaf_id in leaf_ids])
print("Поправки γ для каждого объекта:", gamma_applied)

# Шаг 7: Обновляем модель (пусть learning_rate = 1.0 для простоты)
learning_rate = 1.0
F_new = F_prev + learning_rate * gamma_applied
print("Новое предсказание F1:", F_new)

# Проверка: ошибка уменьшилась?
print("MSE до:", np.mean((y - F_prev)**2))
print("MSE после:", np.mean((y - F_new)**2))