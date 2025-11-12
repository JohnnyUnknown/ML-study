import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Исходные данные
X = np.array([[1], [2], [3], [4], [5], [6]]).astype(float)
y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

# 1. Инициализация F0 — глобальное среднее
F0 = np.mean(y)
F_ordered = np.full_like(y, F0)  # текущие предсказания (будут обновляться)

# 2. Фиксируем перестановку (в реальном CatBoost — случайная)
permutation = np.arange(len(y))  # [0, 1, 2, 3, 4, 5] — исходный порядок
# Можно перемешать: np.random.permutation(len(y))

# 3. Массив для "честных" остатков
ordered_residuals = np.zeros_like(y)

# 4. Инициализируем "накопленную" модель
# В простейшем случае — это просто среднее по уже увиденным y
# (аналог F^{(i)} в CatBoost)

print("Вычисление честных остатков (ordered residuals):\n")

for i in range(len(permutation)):
    idx = permutation[i]  # настоящий индекс объекта в исходных данных
    
    if i == 0:
        # Нет объектов до первого → используем глобальное среднее
        pred = F0
    else:
        # Берём объекты ДО текущего в перестановке
        prev_indices = permutation[:i]  # индексы уже обработанных объектов
        
        # В простейшем случае: предсказание = среднее y по предыдущим
        # (это аналог "промежуточной модели" F^{(i)})
        pred = np.mean(y[prev_indices])
    
    # Сохраняем предсказание и остаток
    F_ordered[idx] = pred
    ordered_residuals[idx] = y[idx] - pred
    
    print(f"Объект {idx} (y={y[idx]}): "
          f"предсказание по предыдущим = {pred:.3f}, "
          f"остаток = {ordered_residuals[idx]:.3f}")

print("\nИтог:")
print("Честные остатки:", ordered_residuals.round(3))
print("Обычные остатки (без порядка):", (y - F0).round(3))