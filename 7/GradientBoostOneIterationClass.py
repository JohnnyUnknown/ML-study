import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Исходные данные
X = np.array([[1], [2], [3], [4], [5], [6]]).astype(float)
y = np.array([0, 0, 0, 1, 1, 1])  # бинарные метки

# --- Шаг 0: Инициализация F0 как логит доли класса 1 ---
p0 = np.mean(y)  # доля положительного класса
# Защита от p0 = 0 или 1
eps = 1e-10
p0 = np.clip(p0, eps, 1 - eps)
F_prev = np.full_like(y, np.log(p0 / (1 - p0)), dtype=float)
print(f"F0 (логит) = {F_prev[0]:.4f}")
print(f"Начальная вероятность = {p0:.4f}")

# --- Шаг 1: Вычисляем антиградиенты (остатки) для LogLoss ---
# Производная LogLoss по F: dL/dF = σ(F) - y
# Антиградиент = -dL/dF = y - σ(F)
p_prev = 1.0 / (1.0 + np.exp(-F_prev))  # сигмоида от F_prev
residuals = y - p_prev
print("Остатки (антиградиенты):", residuals.round(4))

# --- Шаг 2: Обучаем дерево на остатках ---
tree = DecisionTreeRegressor(max_depth=2, random_state=0)
tree.fit(X, residuals)

# --- Шаг 3: Получаем листья ---
leaf_ids = tree.apply(X)

# --- Шаг 4: Для каждого листа вычисляем оптимальное γ ---
# Для LogLoss точное γ требует численной оптимизации.
# Но в классическом Gradient Boosting (Friedman, 2001) используют ПРИБЛИЖЕНИЕ:
# γ ≈ sum(residuals) / sum(p * (1 - p))  <-- это из разложения второго порядка
# Это приближение используется, например, в sklearn.

gamma = {}
for leaf_id in np.unique(leaf_ids):
    mask = (leaf_ids == leaf_id)
    r_sum = residuals[mask].sum()
    # p*(1-p) — это "вторая производная" (гессиан) LogLoss
    hessian = (p_prev[mask] * (1 - p_prev[mask])).sum()
    if hessian == 0:
        gamma_leaf = 0.0
    else:
        gamma_leaf = r_sum / hessian  # приближение γ
    gamma[leaf_id] = gamma_leaf
    print(f"Лист {leaf_id}: "
          f"sum(r)={r_sum:.4f}, sum(h)={hessian:.4f}, γ={gamma_leaf:.4f}")

# --- Шаг 5: Применяем γ ---
gamma_applied = np.array([gamma[leaf_id] for leaf_id in leaf_ids])
print("Поправки γ:", gamma_applied.round(4))

# --- Шаг 6: Обновляем модель (learning_rate = 1.0) ---
learning_rate = 1.0
F_new = F_prev + learning_rate * gamma_applied

# --- Проверка: LogLoss до и после ---
def logloss(y, F):
    # Устойчивая реализация
    F = np.clip(F, -100, 100)  # защита от overflow
    return np.mean(np.log(1 + np.exp(-F)) + y * F)

print("\nLogLoss до:", logloss(y, F_prev))
print("LogLoss после:", logloss(y, F_new))

# --- Вероятности после обновления ---
p_new = 1.0 / (1.0 + np.exp(-F_new))
print("Вероятности после:", p_new.round(4))