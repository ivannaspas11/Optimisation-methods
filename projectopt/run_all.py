"""Запускає всі частини проєкту однією командою:  python run_all.py"""
import runpy
import time

SCRIPTS = [
    ("Частина 0", "main_part0.py"),
    ("Лабораторна 1", "main_lab1.py"),
    ("Лабораторна 2", "main_lab2.py"),
    ("Лабораторна 3", "main_lab3.py"),
    ("Лабораторна 4", "main_lab4.py"),
    ("Лабораторна 5", "main_lab5.py"),
]

for title, script in SCRIPTS:
    print("\n" + "#" * 70)
    print(f"# {title}  ({script})")
    print("#" * 70)
    start = time.time()
    runpy.run_path(script, run_name="__main__")
    print(f"\n[{title}: {time.time() - start:.1f} с]")