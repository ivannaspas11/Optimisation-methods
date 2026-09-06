import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# папка для збереження графіка
os.makedirs("images", exist_ok=True)

func_evals = 0

# цільова функція 20 варіант
def f(x):
    global func_evals
    func_evals += 1
    return (x + 1)**3 * (x + 2.5) * (x - 3.3) + 3.5

# поч парам
x0 = 2.35
delta = 0.01

def swann_method(x0, delta):
    iterations = []
    
    # визнач напрям
    f_minus = f(x0 - delta)
    f_center = f(x0)
    f_plus = f(x0 + delta)
    
    if f_minus >= f_center >= f_plus:
        step = delta
        a = x0 - delta
        direction = "Праворуч (+)"
    elif f_minus <= f_center <= f_plus:
        step = -delta
        b = x0 + delta
        direction = "Ліворуч (-)"
    else:
        # якщо фун формує западину
        return [x0 - delta, x0 + delta], pd.DataFrame()
        
    print(f"Початковий напрямок руху: {direction}\n")
    
    k = 1
    x_prev = x0
    x_curr = x0 + step
    
    iterations.append({
        "k": 0, "x_k": x0, "f(x_k)": f_center, 
        "Крок (h)": step, "Умова спадання": "-"
    })
    
    # ітерац
    while True:
        f_prev = f(x_prev)
        f_curr = f(x_curr)
        
        condition = f_curr < f_prev
        
        iterations.append({
            "k": k, 
            "x_k": x_curr, 
            "f(x_k)": f_curr, 
            "Крок (h)": step * (2**k), 
            "Умова спадання": "Так" if condition else "Ні (Зупинка)"
        })
        
        if not condition:
            # мін між {k-1} і {k+1}
            if step > 0:
                interval = [x_prev_prev, x_curr]
            else:
                interval = [x_curr, x_prev_prev]
            break
            
        x_prev_prev = x_prev
        x_prev = x_curr
        x_curr = x_curr + step * (2**k)
        k += 1
        
    df = pd.DataFrame(iterations)
    return interval, df

func_evals = 0  # скид ліч перед стартом алгор
interval, df_iterations = swann_method(x0, delta)

print("Таблиця ітерацій:")
print(df_iterations.to_string(index=False))
print(f"\nЗнайдений інтервал невизначеності [a0, b0]: [{interval[0]:.4f}, {interval[1]:.4f}]")
print(f"Кількість обчислень f(x): {func_evals}")

# графік

# т для X (діап ширше за знайд інтерв для прикл)
x_vals = np.linspace(2.0, 2.8, 400)
y_vals = f(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label="f(x) = (x+1)³(x+2.5)(x-3.3)+3.5", color="blue", linewidth=2)

# лінія і зона інтерв
plt.axvline(interval[0], color="red", linestyle="--", label=f"Ліва межа a0 = {interval[0]:.4f}")
plt.axvline(interval[1], color="green", linestyle="--", label=f"Права межа b0 = {interval[1]:.4f}")
plt.axvspan(interval[0], interval[1], color='yellow', alpha=0.3, label="Відрізок невизначеності")

plt.title("Метод Свенна: Пошук початкового інтервалу")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.grid(True)

plt.savefig("images/swann_plot.png", dpi=300, bbox_inches='tight')
print("\nГрафік успішно збережено у 'images/swann_plot.png'")

plt.show()