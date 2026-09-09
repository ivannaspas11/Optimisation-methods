import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


func_evals = 0

def f(x):
    global func_evals
    func_evals += 1
    return (x + 1)**3 * (x + 2.5) * (x - 3.3) + 3.5

def dichotomy_method(a, b, sigma, epsilon):
    iterations = []
    k = 1
    
    # зупин якщо довж інтерв > задан точн σ
    while (b - a) > sigma:
        x1 = (a + b) / 2 - epsilon / 2
        x2 = x1 + epsilon
        
        f1 = f(x1)
        f2 = f(x2)
        
        iterations.append({
            "k": k, 
            "a_k": a, 
            "b_k": b, 
            "x1": x1, 
            "x2": x2, 
            "f(x1)": f1, 
            "f(x2)": f2, 
            "b_k - a_k": b - a
        })
        
        # звуж інтерв
        if f1 < f2:
            b = x2
        else:
            a = x1
            
        k += 1
        
    x_opt = (a + b) / 2
    df = pd.DataFrame(iterations)
    return x_opt, [a, b], df

a_initial, b_initial = 2.36, 2.42
sigma_val = 0.001
epsilon_val = 0.0001

func_evals = 0  # скид ліч 
x_opt, final_interval, df_iterations = dichotomy_method(a_initial, b_initial, sigma_val, epsilon_val)
evals_for_algo = func_evals  # фікс кільк викликів до графіка

print("Таблиця ітерацій методу дихотомії:")
print(df_iterations.to_string(index=False))
print(f"\nКількість ітерацій: {len(df_iterations)}")
print(f"Кількість обчислень f(x): {evals_for_algo}")
print(f"Точка мінімуму: x* = {x_opt:.5f}")
print(f"Мінімальне значення функції: f(x*) = {f(x_opt):.5f}")

x_vals = np.linspace(2.35, 2.43, 400)
y_vals = f(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label="f(x) = (x+1)³(x+2.5)(x-3.3)+3.5", color="purple", linewidth=2)

# показ поч і кінцев інтерв
plt.axvline(a_initial, color="grey", linestyle=":", label=f"Початкове a0 = {a_initial}")
plt.axvline(b_initial, color="grey", linestyle=":", label=f"Початкове b0 = {b_initial}")

plt.axvline(final_interval[0], color="red", linestyle="--", label=f"Кінцеве a = {final_interval[0]:.4f}")
plt.axvline(final_interval[1], color="green", linestyle="--", label=f"Кінцеве b = {final_interval[1]:.4f}")
plt.axvspan(final_interval[0], final_interval[1], color='yellow', alpha=0.4, label="Фінальний інтервал")

# т мін
plt.scatter(x_opt, f(x_opt), color="red", zorder=5, label=f"Мінімум x* = {x_opt:.5f}")

plt.title("Метод дихотомії: Звуження інтервалу")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)

plt.savefig("../dichotomy_plot.png", dpi=300, bbox_inches='tight')
print("\nГрафік успішно збережено у '../dichotomy_plot.png'")
plt.show()