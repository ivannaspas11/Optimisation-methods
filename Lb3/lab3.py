import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def f(x):
    return (x + 1)**3 * (x + 2.5) * (x - 3.3) + 3.5

# Початкові дані
a, b = 2.3600, 2.4200
eps = 1e-4

# tau — стала, рахуємо один раз
tau = (5 ** 0.5 - 1) / 2

# Ініціалізація
x1 = a + (1 - tau) * (b - a)
x2 = a + tau * (b - a)
f1, f2 = f(x1), f(x2)  # 2 обчислення функції

history = []
k = 0
L_k = b - a
N_f = 2
trial_points = [(x1, f1), (x2, f2)]

# Запис початкового стану
history.append({
    'k': k, 'a_k': a, 'b_k': b, 'x1': x1, 'x2': x2, 
    'f(x1)': f1, 'f(x2)': f2, 'L_k': L_k, 'L_k/L_{k-1}': '-', 'Рішення': '-'
})

while (b - a) > eps:
    k += 1
    L_prev = b - a
    
    if f1 <= f2:
        decision = "f1 <= f2: зсув вправо (b=x2)"
        b = x2
        x2, f2 = x1, f1  # успадкували
        x1 = a + (1 - tau) * (b - a)
        f1 = f(x1)       # 1 обчислення
    else:
        decision = "f1 > f2: зсув вліво (a=x1)"
        a = x1
        x1, f1 = x2, f2  # успадкували
        x2 = a + tau * (b - a)
        f2 = f(x2)       # 1 обчислення
        
    N_f += 1
    trial_points.append((x1, f1) if f1 <= f2 else (x2, f2))
    L_k = b - a
    
    history.append({
        'k': k, 'a_k': a, 'b_k': b, 'x1': x1, 'x2': x2, 
        'f(x1)': f1, 'f(x2)': f2, 'L_k': L_k, 
        'L_k/L_{k-1}': L_k / L_prev, 'Рішення': decision
    })

x_star = (a + b) / 2
f_star = f(x_star)

# Виведення таблиці
df = pd.DataFrame(history)
pd.set_option('display.float_format', lambda x: '%.6f' % x if isinstance(x, float) else x)
print(df.to_string(index=False))
print(f"\nЗнайдений мінімум: x* = {x_star:.6f}, f(x*) = {f_star:.6f}")
print(f"Кількість ітерацій: {k}")
print(f"Кількість обчислень функції N_f: {N_f}")

# Побудова графіка
x_vals = np.linspace(2.35, 2.43, 400)
y_vals = f(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label='f(x)', color='blue')

# Нанесення пробних точок
tp_x = [pt[0] for pt in trial_points]
tp_y = [pt[1] for pt in trial_points]
plt.scatter(tp_x, tp_y, color='red', zorder=5, label='Пробні точки (x_k, f_k)')
plt.axvline(x_star, color='green', linestyle='--', label=f'x* = {x_star:.4f}')

plt.title('Метод золотого перерізу')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()