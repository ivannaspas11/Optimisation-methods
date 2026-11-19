"""Блок-схема алгоритму Флетчера-Рівса -> lr9_flowchart.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

fig, ax = plt.subplots(figsize=(11, 15))
ax.set_xlim(-7, 10.5); ax.set_ylim(-1.2, 17.6); ax.axis("off")
EC, FC = "#1f3b5c", "#eef4fb"

def box(x, y, w, h, text, fs=10, rounded=False):
    st = "round,pad=0.02,rounding_size=0.4" if rounded else "square,pad=0.02"
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle=st, fc=FC, ec=EC, lw=1.6))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)

def diamond(x, y, w, h, text, fs=10):
    ax.add_patch(Polygon([(x, y + h/2), (x + w/2, y), (x, y - h/2), (x - w/2, y)], closed=True, fc="#fff6e0", ec=EC, lw=1.6))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)

def arrow(pts, label=None, lpos=None):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=EC, lw=1.5)
    ax.annotate("", xy=pts[-1], xytext=pts[-2], arrowprops=dict(arrowstyle="-|>", color=EC, lw=1.5))
    if label:
        ax.text(lpos[0], lpos[1], label, fontsize=10, fontweight="bold", color="#8a1c1c", ha="center", va="center")

box(0, 16.4, 5.4, 1.0, "Початок\nx⁽⁰⁾, ε, n", rounded=True)                                   # 1
box(0, 14.7, 6.6, 1.2, "k = 0;  g = ∇F(x⁽⁰⁾)\nd = −g   (пам'яті ще немає)")                    # 2
diamond(0, 12.6, 5.4, 2.0, "‖g‖₂ ≤ ε ?")                                                      # 3
box(6.2, 12.6, 4.2, 1.0, "x* = x\nКінець", rounded=True)
box(0, 10.4, 7.2, 1.3, "Точний лінійний пошук:\nh = arg min φ(h), φ(h) = F(x + h·d)\n(Свенн → відрізок → золотий переріз)")   # 4
box(0, 8.6, 5.4, 1.0, "x = x + h·d")                                                           # 5
box(0, 7.0, 7.0, 1.3, "g_new = ∇F(x)\nβ = ‖g_new‖² / ‖g‖²")                                   # 6
diamond(0, 4.9, 6.0, 2.1, "(k+1) mod n = 0 ?\n(перезапуск)")
box(6.4, 4.9, 3.0, 1.0, "β = 0")
box(0, 2.6, 7.0, 1.3, "d = −g_new + β·d;  g = g_new\nk = k + 1")                               # 7

arrow([(0, 15.9), (0, 15.3)])
arrow([(0, 14.1), (0, 13.6)])
arrow([(2.7, 12.6), (4.1, 12.6)], "так", (3.4, 12.95))
arrow([(0, 11.6), (0, 11.05)], "ні", (0.35, 11.3))
arrow([(0, 9.75), (0, 9.1)])
arrow([(0, 8.1), (0, 7.65)])
arrow([(0, 6.35), (0, 5.95)])
arrow([(3.0, 4.9), (4.9, 4.9)], "так", (3.95, 5.25))
arrow([(0, 3.85), (0, 3.25)], "ні", (0.35, 3.55))
arrow([(6.4, 4.4), (6.4, 2.6), (3.5, 2.6)])                                                   # β=0 -> d
arrow([(-3.5, 2.6), (-5.0, 2.6), (-5.0, 12.6), (-2.7, 12.6)])                                 # назад до кроку 3
ax.text(-6.8, 17.3, "Блок-схема алгоритму Флетчера–Рівса", fontsize=13, fontweight="bold", va="center")
    