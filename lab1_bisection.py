import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# папка для збереження графіка
os.makedirs("images", exist_ok=True)

func_evals = 0

def f(x):
    global func_evals
    func_evals += 1
    return (x + 1)**3 * (x + 2.5) * (x - 3.3) + 3.5

def bisection_method(a, b, sigma):
    iterations = []
    k = 1
    
    # перша сер. т.
    xm = (a + b) / 2
    fm = f(xm)
    
    # ум зуп. довж інтерв більша за задану точн σ
    while (b - a) > sigma:
        L = b - a
        x1 = a + L / 4
        x2 = b - L / 4
        
        f1 = f(x1)
        f2 = f(x2)
        
        iterations.append({
            "k": k, 
            "a": a, 
            "b": b, 
            "x1": x1, 
            "xm": xm, 
            "x2": x2,
            "f(x1)": f1, 
            "f(xm)": fm, 
            "f(x2)": f2, 
            "b - a": L
        })
        
        # прав звуж інтерв для м діл навп
        if f1 < fm:
            b = xm
            xm = x1
            fm = f1
        elif f2 < fm:
            a = xm
            xm = x2
            fm = f2
        else:
            a = x1
            b = x2
            
        k += 1
        
    # т мін серед ост інтерв
    x_opt = (a + b) / 2
    df = pd.DataFrame(iterations)
    return x_opt, [a, b], df

# поч парам (з м Свенна)
a_initial, b_initial = 2.36, 2.42
sigma_val = 0.001

func_evals = 0  
# виконання м
x_opt, final_interval, df_iterations = bisection_method(a_initial, b_initial, sigma_val)
evals_for_algo = func_evals  

# вивід результ
print("Таблиця ітерацій методу половинного поділу:")
print(df_iterations.to_string(index=False))
print(f"\nКількість ітерацій: {len(df_iterations)}")
print(f"Кількість обчислень f(x): {evals_for_algo}")
print(f"Точка мінімуму: x* = {x_opt:.5f}")
print(f"Мінімальне значення функції: f(x*) = {f(x_opt):.5f}")

# будув граф
x_vals = np.linspace(2.35, 2.43, 400)
y_vals = f(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label="f(x) = (x+1)³(x+2.5)(x-3.3)+3.5", color="teal", linewidth=2)

# показ поч та кінц інтерв
plt.axvline(a_initial, color="grey", linestyle=":", label=f"Початкове a0 = {a_initial}")
plt.axvline(b_initial, color="grey", linestyle=":", label=f"Початкове b0 = {b_initial}")

plt.axvline(final_interval[0], color="red", linestyle="--", label=f"Кінцеве a = {final_interval[0]:.4f}")
plt.axvline(final_interval[1], color="green", linestyle="--", label=f"Кінцеве b = {final_interval[1]:.4f}")
plt.axvspan(final_interval[0], final_interval[1], color='yellow', alpha=0.4, label="Фінальний інтервал")

# т мін
plt.scatter(x_opt, f(x_opt), color="red", zorder=5, label=f"Мінімум x* = {x_opt:.5f}")

plt.title("Метод половинного поділу (ділення відрізка навпіл)")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)

plt.savefig("images/bisection_plot.png", dpi=300, bbox_inches='tight')
print("\nГрафік успішно збережено у 'images/bisection_plot.png'")
plt.show()