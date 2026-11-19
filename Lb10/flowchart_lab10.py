"""Блок-схема алгоритму Хука-Дживса (ГОСТ 19.701-90 / ДСТУ ISO 5807) -> lr10_flowchart.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

fig, ax = plt.subplots(figsize=(11, 13))
ax.set_xlim(-6.5, 10.5); ax.set_ylim(-3.6, 13.3); ax.axis("off")
EC, FC = "#1f3b5c", "#eef4fb"

def box(x, y, w, h, text, fs=10, rounded=False, fc=FC):
    st = "round,pad=0.02,rounding_size=0.4" if rounded else "square,pad=0.02"
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle=st, fc=fc, ec=EC, lw=1.6))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)

def diamond(x, y, w, h, text, fs=10):
    ax.add_patch(Polygon([(x, y + h/2), (x + w/2, y), (x, y - h/2), (x - w/2, y)], closed=True, fc="#fff6e0", ec=EC, lw=1.6))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)

def arrow(pts, label=None, lpos=None):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=EC, lw=1.5)
    a, b = pts[-2], pts[-1]
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=EC, lw=1.5))
    if label:
        ax.text(lpos[0], lpos[1], label, fontsize=10, fontweight="bold", color="#8a1c1c", ha="center", va="center")

# вузли
box(0, 12, 5.2, 1.0, "Початок\nx⁽⁰⁾, Δ₀, ε", rounded=True)                        # 1
box(0, 10.3, 6.4, 1.2, "k = 0;  Δ = Δ₀;  F₀ = F(x⁽⁰⁾)\ny = x⁽⁰⁾  (звідки шукати)")       # 2
box(0, 8.4, 6.4, 1.2, "Дослідний пошук із y кроком Δ\nпо кожній координаті → z, F(z)")    # 3
diamond(0, 6.2, 5.8, 2.0, "F(z) < F(x⁽ᵏ⁾) ?\n(порівняння з базою)")                      # 4
box(6.2, 6.2, 5.4, 1.5, "x⁽ᵏ⁺¹⁾ = z\nзразок: y = z + (z − x⁽ᵏ⁾)\nk = k + 1")             # 5
diamond(0, 3.5, 5.8, 2.0, "y ≠ x⁽ᵏ⁾ ?\n(шукали зі зразка?)")                            # 6
box(6.2, 3.5, 4.2, 1.0, "y = x⁽ᵏ⁾\n(повернення в базу)")                                  # 7
box(0, 1.2, 4.8, 1.0, "Δ = Δ / 2")                                                      # 8
diamond(0, -0.9, 5.0, 1.8, "Δ < ε ?")                                                   # 9
box(6.2, -0.9, 4.4, 1.0, "x* = x⁽ᵏ⁾\nКінець", rounded=True)

# стрілки
arrow([(0, 11.5), (0, 10.9)])
arrow([(0, 9.7), (0, 9.0)])
arrow([(0, 7.8), (0, 7.2)])
arrow([(2.9, 6.2), (3.5, 6.2)], "так", (3.2, 6.55))
arrow([(0, 5.2), (0, 4.5)], "ні", (0.35, 4.85))
arrow([(2.9, 3.5), (4.1, 3.5)], "так", (3.5, 3.85))
arrow([(0, 2.5), (0, 1.7)], "ні", (0.35, 2.1))
arrow([(0, 0.7), (0, 0.0)])
arrow([(2.5, -0.9), (4.0, -0.9)], "так", (3.25, -0.55))
# повернення до кроку 3
arrow([(8.9, 6.2), (9.6, 6.2), (9.6, 8.55), (3.2, 8.55)])      # з кроку 5
arrow([(8.3, 3.5), (8.8, 3.5), (8.8, 8.25), (3.2, 8.25)])      # з кроку 7
arrow([(0, -1.8), (0, -2.6), (-4.8, -2.6), (-4.8, 8.4), (-3.2, 8.4)], "ні", (0.35, -2.2))  # Δ ≥ ε

ax.text(-6.3, 12.9, "Блок-схема алгоритму Хука–Дживса", fontsize=13, fontweight="bold", va="center")
fig.savefig("lr10_flowchart.png", dpi=130, bbox_inches="tight")