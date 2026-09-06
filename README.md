# Методи одновимірної оптимізації

Реалізація чисельних методів оптимізації для пошуку локального мінімуму функції:
$$f(x) = (x+1)^3(x+2.5)(x-3.3)+3.5$$

## Реалізовані алгоритми

1. **Метод Свенна** — знаходження початкового інтервалу невизначеності $[a, b]$, що гарантовано містить локальний мінімум.
2. **Метод дихотомії** — звуження інтервалу невизначеності з параметром зміщення $\varepsilon = 0.0001$ та точністю $\sigma = 0.001$.
3. **Метод половинного поділу (Bisection)** — звуження інтервалу шляхом його поділу на 4 рівні частини та аналізу значень у 3 внутрішніх точках.

---

## Структура репозиторію

```text
├── images/
│   ├── swann_plot.png
│   ├── swann_flow.png
│   ├── dichotomy_plot.png
│   ├── dichotomy_flow.png
│   ├── bisection_plot.png
│   └── bisection_flow.png
├── lab1_swann.py
├── lab1_dichotomy.py
├── lab1_bisection.py
├── requirements.txt
├── .gitignore
└── README.md
```
## Блок-схеми алгоритмів (ДСТУ ISO 5807:2016)

### Метод Свенна
![Блок-схема Свенна](images/swann_flow.png)

### Метод дихотомії
![Блок-схема дихотомії](images/dichotomy_flow.png)

### Метод половинного поділу
![Блок-схема половинного поділу](images/bisection_flow.png)

---

## Результати та візуалізація

### Метод Свенна
![Графік Свенна](images/swann_plot.png)

### Метод дихотомії
![Графік дихотомії](images/dichotomy_plot.png)

### Метод половинного поділу
![Графік половинного поділу](images/bisection_plot.png)

---

## Встановлення та запуск
1. Клонувати репозиторій:
```bash
git clone https://github.com/ivannaspas11/Optimisation-methods.git
cd Optimisation-methods
pip install -r requirements.txt
python lab1_swann.py
python lab1_dichotomy.py
python lab1_bisection.py
```