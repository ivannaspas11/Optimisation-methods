"""ЛР 11, частина 1. Метод зовнішніх штрафних функцій.
F = x1^2 + 2x2^2 + 2x1x2 - 9x1 - 13x2,  x0 = (-3; -2)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

X0 = np.array([-3.0, -2.0])
HESS = np.array([[2.0, 2.0], [2.0, 4.0]])      # F = 1/2 x'Hx - b'x
BVEC = np.array([9.0, 13.0])
EPS = 0.01


# ======================= функції F, ∇F та обмеження =======================
f_main = lambda x: x[0]**2 + 2*x[1]**2 + 2*x[0]*x[1] - 9*x[0] - 13*x[1]
g_main = lambda x: np.array([2*x[0] + 2*x[1] - 9, 2*x[0] + 4*x[1] - 13])

class Lin:
    """Лінійне обмеження g(x) = a·x - beta <= 0."""
    def __init__(self, a, beta, name):
        self.a, self.beta, self.name = np.array(a, float), float(beta), name
    def g(self, x):  return float(self.a @ x - self.beta)
    def grad(self, x): return self.a

# як у завданні: x1 - 2x2 <= 35
CON_A = Lin([1, -2], 35, "x1 - 2x2 <= 35   (g = x1 - 2x2 - 35)")
# контрольний варіант (знак обернено): x1 - 2x2 >= 35  ->  g = 35 - x1 + 2x2 <= 0
CON_B = Lin([-1, 2], -35, "x1 - 2x2 >= 35   (g = 35 - x1 + 2x2)")


# ======================= лічильники =======================
class Obj:
    """Обгортка: рахує звернення до функції (nf) та до градієнта (ng)."""
    def __init__(self, f, g):
        self._f, self._g, self.nf, self.ng = f, g, 0, 0
    def F(self, x):
        self.nf += 1
        return self._f(np.asarray(x, float))
    def grad(self, x):
        self.ng += 1
        return self._g(np.asarray(x, float))


# ======================= штрафна функція Φr та її градієнт =======================
def penalty(F, gradF, cons, r):
    """Φr = F + r·Σ max(0, g)^2;  ∇Φr = ∇F + 2r·Σ max(0, g)·∇g."""
    def Phi(x):
        return F(x) + r * sum(max(0.0, c.g(x))**2 for c in cons)
    def gradPhi(x):
        v = np.array(gradF(x), float)
        for c in cons:
            v = v + 2*r*max(0.0, c.g(x))*c.grad(x)       # «пружина» до межі
        return v
    return Phi, gradPhi

def P_of(cons, x): return sum(max(0.0, c.g(x))**2 for c in cons)


# ======================= ЛР 9: Свенн + золотий + Флетчер-Рівс =======================
def svenn(phi, delta0=0.01):
    pts, vals = [0.0], [phi(0.0)]; h, step = 0.0, delta0
    while True:
        h += step; v = phi(h); pts.append(h); vals.append(v)
        if v > vals[-2]: break
        step *= 2
    return (pts[-3] if len(pts) >= 3 else pts[0]), pts[-1]

def golden(phi, a, b, eps=1e-6):
    t = (np.sqrt(5) - 1) / 2
    x1, x2 = b - t*(b - a), a + t*(b - a); f1, f2 = phi(x1), phi(x2)
    while b - a > eps:
        if f1 < f2: b, x2, f2 = x2, x1, f1; x1 = b - t*(b - a); f1 = phi(x1)
        else:       a, x1, f1 = x1, x2, f2; x2 = a + t*(b - a); f2 = phi(x2)
    return (a + b) / 2

def fletcher_reeves(obj, x0, eps=EPS, restart=2, max_iter=5000, log=False, ls_eps=1e-6):
    x = np.array(x0, float); g = obj.grad(x); d = -g
    traj, rows, k = [x.copy()], [], 0
    while np.linalg.norm(g) > eps and k < max_iter:
        phi = lambda h: obj.F(x + h*d)
        a, b = svenn(phi); h = golden(phi, a, b, ls_eps)
        x_new = x + h*d; g_new = obj.grad(x_new)
        beta = (g_new @ g_new) / (g @ g)
        if (k + 1) % restart == 0: beta = 0.0
        if log: rows.append((k, x.copy(), obj._f(x), g.copy(), np.linalg.norm(g), d.copy(), h, beta, x_new.copy()))
        d = -g_new + beta*d; x, g = x_new, g_new; traj.append(x.copy()); k += 1
    return x, k, np.array(traj), rows, np.linalg.norm(g)


# ======================= SUMT: зовнішній цикл за r =======================
def sumt(F, gradF, cons, x0, r0=1.0, c=10.0, eps=EPS, inner_eps=EPS, r_list=None, max_outer=40, log_first=False, ls_eps=1e-6):
    """r_list - примусовий перелік r (без перевірки зупинки); інакше r *= c, стоп при r·P <= eps."""
    x = np.array(x0, float); r = r0 if r_list is None else r_list[0]
    out, inner_log, paths = [], None, []
    nphi_tot = 0
    for k in range(1, max_outer + 1):
        Phi, gradPhi = penalty(F, gradF, cons, r)
        o = Obj(Phi, gradPhi)
        xs = x.copy()
        x, it, traj, rows, gn = fletcher_reeves(o, x, eps=inner_eps, log=(log_first and k == 1), ls_eps=ls_eps)
        if log_first and k == 1: inner_log = rows
        paths.append(traj)
        nphi_tot += o.nf
        P = P_of(cons, x); gmax = max(cn.g(x) for cn in cons)
        lam = [2*r*max(0.0, cn.g(x)) for cn in cons]
        out.append(dict(k=k, r=r, start=xs, it=it, nphi=o.nf, ngphi=o.ng, x=x.copy(), F=F(x), g=gmax, P=P,
                        rP=r*P, lam=lam, nphi_tot=nphi_tot, gnorm=gn))
        if r_list is None:
            if r*P <= eps: break
            r *= c
        else:
            if k >= len(r_list): break
            r = r_list[k]
    return x, out, inner_log, paths


# ======================= аналітика =======================
def kkt_boundary(a, beta):
    """min F при a·x = beta:  [H a; a' 0][x; λ] = [b; beta]  ->  x_b, λ (∇F = -λ a)."""
    M = np.block([[HESS, a.reshape(2, 1)], [a.reshape(1, 2), np.zeros((1, 1))]])
    sol = np.linalg.solve(M, np.append(BVEC, beta))
    return sol[:2], float(sol[2])

def analytic_xr(con, r):
    """На порушеній стороні Φr квадратична: (H + 2r aa')x = b + 2r·beta·a. Повертає (x, g(x), чи справді g>0)."""
    a, be = con.a, con.beta
    x = np.linalg.solve(HESS + 2*r*np.outer(a, a), BVEC + 2*r*be*a)
    return x, con.g(x), con.g(x) > 0


# ======================= форматування =======================
def num(v, nd=6):
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s
def pt(v, nd=6): return "(" + "; ".join(num(t, nd) for t in v) + ")"
def sci(v): return f"{v:.3e}"
def rfmt(r): return f"{r:g}"

def print_outer(rows):
    hdr = (f"{'k':>2}{'r':>9} {'старт внутрішньої':<22}{'іт.ФР':>6}{'NΦ':>6}{'N∇Φ':>5}  {'x(r)':<24}"
           f"{'F(x(r))':>12}{'g(x(r))':>12}{'r·P':>11}{'2r·g':>10}{'NΦ разом':>9}")
    print(hdr); print("-" * len(hdr))
    for w in rows:
        print(f"{w['k']:>2}{rfmt(w['r']):>9} {pt(w['start'], 4):<22}{w['it']:>6}{w['nphi']:>6}{w['ngphi']:>5}  "
              f"{pt(w['x'], 6):<24}{w['F']:>12.5f}{w['g']:>12.6f}{w['rP']:>11.3e}{w['lam'][0]:>10.4f}{w['nphi_tot']:>9}")


def main():
    SEP = "=" * 118
    # ---------- П.1 ----------
    print(SEP)
    print("П.1 Постановка та аналітика (для контролю)")
    print("F(x1,x2) = x1^2 + 2x2^2 + 2x1x2 - 9x1 - 13x2,  x0 = (-3; -2),  F(x0) = 82")
    print("∇F = (2x1+2x2-9; 2x1+4x2-13),  H = [[2, 2], [2, 4]],  безумовний мінімум x_безум = (2.5; 2), F = -24.25")
    xu = np.array([2.5, 2.0])
    for name, con in (("A", CON_A), ("B (контрольний, знак обернено)", CON_B)):
        xb, lam = kkt_boundary(con.a, con.beta)
        print(f"\nВаріант {name}: {con.name}")
        print(f"   g(старт) = {num(con.g(X0), 4)},  g(безумовний мінімум) = {num(con.g(xu), 4)}"
              f"  -> безумовний мінімум {'ПОРУШУЄ' if con.g(xu) > 0 else 'НЕ порушує'} обмеження")
        print(f"   мінімум F на прямій-межі: x_межі = {pt(xb, 4)}, F = {num(f_main(xb), 5)}, "
              f"множник λ (∇F = -λ∇g) = {num(lam, 4)}   ∇F(x_межі) = {pt(g_main(xb), 4)}, ∇g = {pt(con.a, 1)}")
        if con.g(xu) > 0:
            print(f"   => обмеження АКТИВНЕ: x* = {pt(xb, 4)}, F* = {num(f_main(xb), 5)}, λ = {num(lam, 4)} >= 0")
        else:
            print(f"   => обмеження НЕАКТИВНЕ: x* = (2.5; 2), F* = -24.25, λ = 0. На межі λ = {num(lam, 3)} < 0, "
                  f"тобто «притискати до межі» F не вигідно (F на межі = {num(f_main(xb), 3)} > -24.25)")

    # ---------- П.2 самоперевірка на прикладі зі слайдів ----------
    print("\n" + SEP)
    print("П.2 Самоперевірка реалізації на прикладі зі слайдів 11-15: F = 2x1^2+3x2^2+x1x2, g = 4 - x1 - x2, старт (5; -3)")
    fs = lambda x: 2*x[0]**2 + 3*x[1]**2 + x[0]*x[1]
    gs = lambda x: np.array([4*x[0] + x[1], x[0] + 6*x[1]])
    con_s = Lin([-1, -1], -4, "4 - x1 - x2 <= 0")
    xs_, rows_s, inner_s, _ = sumt(fs, gs, [con_s], [5, -3], log_first=True)
    print_outer(rows_s)
    print(f"За слайдами: ітерацій ФР 2,2,4,4,2; NΦ = 68,66,122,101,48 (разом 405), N∇Φ = 19; x(1e4) = (2.498995; 1.500430)")
    tot_g = sum(w['ngphi'] for w in rows_s)
    ok = ([w['it'] for w in rows_s] == [2, 2, 4, 4, 2] and rows_s[-1]['nphi_tot'] == 405 and tot_g == 19)
    print(f"Збіг зі слайдами: {'ТАК' if ok else 'НІ'};  перший крок внутрішньої задачі h = {inner_s[0][6]:.6f} "
          f"(слайд: 0.229),  точка {pt(inner_s[0][8], 3)}")

    # ---------- П.3 варіант A ----------
    print("\n" + SEP)
    print("П.3 ВАРІАНТ A: x1 - 2x2 <= 35; обмеження неактивне")
    xA, rowsA, innerA, pathsA = sumt(f_main, g_main, [CON_A], X0, log_first=True)
    print("Звичайний прогін (r0 = 1, c = 10, ε = 0.01): зупинка настає одразу, бо r·P = 0 уже при r = 1")
    print_outer(rowsA)
    print("\nПримусовий прогін r = 1, 10, 100, 1000, 10^4 (щоб показати, що r нічого не змінює):")
    xA2, rowsA2, _, _ = sumt(f_main, g_main, [CON_A], X0, r_list=[1, 10, 100, 1e3, 1e4])
    print_outer(rowsA2)
    print(f"Результат: x = {pt(xA2, 6)}, F = {num(f_main(xA2), 6)}, max g = {num(rowsA2[-1]['g'], 4)} < 0, штраф P = 0, оцінка 2r·max(0,g) = 0")
    xb, lamb = kkt_boundary(CON_A.a, CON_A.beta)
    print(f"Порівняння з мінімумом на межі: x_межі = {pt(xb, 4)}, F = {num(f_main(xb), 4)}  (≫ {num(f_main(xA2), 4)}), "
          f"λ = {num(lamb, 3)} < 0 -> межу не «чіпаємо».")

    print("\nВнутрішня задача r = 1 (Флетчер-Рівс, ‖∇Φ‖ ≤ 0.01), старт (-3; -2); Φ1 = F, бо g < 0 уздовж усього шляху:")
    print(f"{'k':>2} {'x(k)':<26}{'Φ1(x(k))':>11} {'∇Φ1':<26}{'‖∇Φ1‖':>9} {'d(k)':<26}{'h_k':>10}{'β':>9}{'g(x(k))':>10}")
    for (k, x, Phi_, g_, ng, d, h, beta, xn) in innerA:
        print(f"{k:>2} {pt(x, 5):<26}{Phi_:>11.4f} {pt(g_, 5):<26}{ng:>9.4f} {pt(d, 5):<26}{h:>10.6f}{beta:>9.6f}{CON_A.g(x):>10.3f}")
    print(f"   -> x(1) = {pt(innerA[-1][8], 6)},  g = {CON_A.g(innerA[-1][8]):.4f} < 0,  NΦ = {rowsA[0]['nphi']},  N∇Φ = {rowsA[0]['ngphi']}   (той самий прогін, що ЛР 9)")
    print("\nПеревірка аналітикою (слайд 10). Система (H + 2r·aa')x = b + 2rβ·a правильна лише якщо її розв'язок на порушеній стороні (g > 0):")
    print(f"{'r':>9} {'розв\'язок системи':<26}{'g(розв\'язку)':>15}  висновок")
    for rr_ in (1, 10, 100, 1e3, 1e4):
        xa, ga, side = analytic_xr(CON_A, rr_)
        print(f"{rfmt(rr_):>9} {pt(xa, 5):<26}{ga:>15.4f}  " + ("g > 0: зрізка діє" if side else "g < 0: розв'язок УСЕРЕДИНІ Ω, зрізка нуль -> насправді x(r) = безумовний мінімум (2.5; 2)"))
    print(f"Чисельне x(r) для всіх r: {pt(rowsA2[-1]['x'], 6)}  (аналітично (2.5; 2)), ‖Δ‖ = {np.linalg.norm(rowsA2[-1]['x'] - xu):.2e}")
    print("Оцінка множника 2r·max(0, g) = 0 при всіх r; порушення max(0, g) = 0, тож «порушення ≈ λ/(2r)» дає 0 = 0.")
    print("\nДошліфовування проєкцією (слайд 16) не потрібне: порушення нуль, x̂ = x. Великі r зі старту (ls_eps = 1e-6, ліміт 5000 ітер.):")
    print(f"{'сценарій':<42}{'x':<24}{'F':>11}{'NΦ':>8}{'ітер.':>7}")
    for lab_, rl_ in (("r = 1 (базовий)", [1]), ("одразу r = 1e4 зі старту", [1e4]), ("одразу r = 1e6 зі старту", [1e6]),
                      ("r = 1 → 10 → … → 1e4 (теплий старт)", [1, 10, 100, 1e3, 1e4])):
        xo_, ro_, _, _ = sumt(f_main, g_main, [CON_A], X0, r_list=rl_)
        print(f"{lab_:<42}{pt(ro_[-1]['x'], 4):<24}{ro_[-1]['F']:>11.5f}{sum(q['nphi'] for q in ro_):>8}{sum(q['it'] for q in ro_):>7}")
    print("Штраф неактивний, тому великий r нічого не псує (Φr = F): гесіан HΦ = H, обумовленість ≈ 6.85 при будь-якому r.")

    # ---------- П.4 варіант B ----------
    print("\n" + SEP)
    print("Д.1 ДОДАТОК. ВАРІАНТ B (контрольний, знак обернено): x1 - 2x2 >= 35, g = 35 - x1 + 2x2 <= 0; ∇g = (-1; 2)")
    xs_b, lam_b = kkt_boundary(CON_B.a, CON_B.beta)
    Fs_b = f_main(xs_b)
    print(f"Аналітично: x* = {pt(xs_b, 4)}, F* = {num(Fs_b, 5)}, λ = {num(lam_b, 4)}")
    xB, rowsB, innerB, pathsB = sumt(f_main, g_main, [CON_B], X0, log_first=True)

    # внутрішня задача r = 1: повний протокол
    print("\nВнутрішня задача r = 1 (Флетчер-Рівс, ‖∇Φ‖ ≤ 0.01), старт (-3; -2):")
    print(f"{'k':>2} {'x(k)':<26}{'Φ1(x(k))':>11} {'∇Φ1':<26}{'‖∇Φ1‖':>9} {'d(k)':<26}{'h_k':>10}{'β':>9}")
    for (k, x, Phi_, g_, ng, d, h, beta, xn) in innerB:
        print(f"{k:>2} {pt(x, 5):<26}{Phi_:>11.4f} {pt(g_, 5):<26}{ng:>9.4f} {pt(d, 5):<26}{h:>10.6f}{beta:>9.6f}")
    print(f"   -> x(1) = {pt(innerB[-1][8], 6)},  g = {rowsB[0]['g']:.6f},  NΦ = {rowsB[0]['nphi']},  N∇Φ = {rowsB[0]['ngphi']}")
    # перевірка першого кроку аналітично
    r = 1.0
    HPhi = HESS + 2*r*np.outer(CON_B.a, CON_B.a)
    k0, x0_, Phi0, g0, ng0, d0, h0, b0, x1_ = innerB[0]
    h_an = (g0 @ g0) / (d0 @ HPhi @ d0)
    print(f"Перевірка 1-го кроку: HΦ = H + 2r·a·a' = {HPhi.tolist()}; h = ‖∇Φ‖²/(d'HΦ d) = {ng0**2:.4f}/{d0 @ HPhi @ d0:.4f} = {h_an:.6f}"
          f"  (чисельно {h0:.6f});  g на кроці = {num(CON_B.g(x0_ + h_an*d0), 4)} > 0 (зрізка не спрацьовує)")
    xan1, gan1, side = analytic_xr(CON_B, 1.0)
    print(f"Аналітична x(1) з системи (H+2r·aa')x = b + 2rβ·a: {pt(xan1, 6)}, g = {num(gan1, 6)}; чисельна: {pt(rowsB[0]['x'], 6)}")

    print("\nПротокол зовнішніх ітерацій (r0 = 1, c = 10, ε = 0.01, стоп при r·P ≤ ε):")
    print_outer(rowsB)
    print(f"Результат: x ≈ {pt(xB, 6)}, F ≈ {num(f_main(xB), 5)} (точно {num(Fs_b, 5)}), порушення g = {rowsB[-1]['g']:.3e}, "
          f"λ ≈ {num(rowsB[-1]['lam'][0], 4)} (точно {num(lam_b, 4)});  NΦ = {rowsB[-1]['nphi_tot']}, N∇Φ = {sum(w['ngphi'] for w in rowsB)}")

    # перевірка аналітикою для кожного r
    print("\nПеревірка аналітикою (слайди 10, 13): x(r) з системи 2x2; 2r·g → λ; порушення ≈ λ/(2r); F* - F ≈ λ²/(2r)")
    print(f"{'r':>9} {'x(r) чисельне':<26}{'x(r) аналітичне':<26}{'‖Δx‖':>10}{'g чисельне':>13}{'g аналіт.':>12}{'r·g':>9}{'2r·g':>9}{'r·(F*-F)':>10}")
    for w in rowsB:
        xa, ga, side = analytic_xr(CON_B, w['r'])
        print(f"{rfmt(w['r']):>9} {pt(w['x'], 5):<26}{pt(xa, 5):<26}{np.linalg.norm(w['x'] - xa):>10.2e}{w['g']:>13.6f}{ga:>12.6f}"
              f"{w['r']*w['g']:>9.4f}{2*w['r']*w['g']:>9.4f}{w['r']*(Fs_b - w['F']):>10.4f}")
    print(f"   Теорія: r·g → λ/2 = {lam_b/2:.4f};  2r·g → λ = {lam_b:.4f};  r·(F*-F) → λ²/2 = {lam_b**2/2:.4f}")
    print("Властивості (слайд 8): F(x(r)) зростає до F* знизу; g(x(r)) > 0 і спадає; r·P ≥ 0 спадає:")
    print("   F(x(r)): " + " < ".join(num(w['F'], 4) for w in rowsB) + f"  < F* = {num(Fs_b, 4)}")

    # ---------- П.5 дошліфовування і порівняння з r = 1e6 ----------
    print("\n" + SEP)
    print("Д.2 ДОДАТОК. Варіант B: дошліфовування проєкцією на межу та порівняння з великим r одразу зі старту")
    last = rowsB[-1]
    xr = last['x']; gr = CON_B.g(xr)
    xh = xr - gr*CON_B.a/(CON_B.a @ CON_B.a)
    print(f"x(r={rfmt(last['r'])}) = {pt(xr, 6)}, g = {gr:.3e}, F = {num(f_main(xr), 6)}")
    print(f"x̂ = x - g·∇g/‖∇g‖² = {pt(xh, 6)}, g(x̂) = {CON_B.g(xh):.1e}, F(x̂) = {num(f_main(xh), 6)}, ‖x̂ - x*‖ = {np.linalg.norm(xh - xs_b):.3e}")
    print(f"{'що зробили':<46}{'x':<26}{'g':>12}{'F':>12}{'‖x-x*‖':>11}{'NΦ':>9}{'ітер.ФР':>9}")
    print(f"{'зупинка при r = ' + rfmt(last['r']):<46}{pt(xr, 5):<26}{gr:>12.3e}{f_main(xr):>12.5f}{np.linalg.norm(xr - xs_b):>11.2e}{last['nphi_tot']:>9}{sum(w['it'] for w in rowsB):>9}")
    print(f"{'+ проєкція на межу':<46}{pt(xh, 5):<26}{CON_B.g(xh):>12.1e}{f_main(xh):>12.5f}{np.linalg.norm(xh - xs_b):>11.2e}{last['nphi_tot']:>9}{'+0':>9}")
    print(f"\nПорівняння з великим r (внутрішній ФР: ‖∇Φ‖ ≤ 0.01, ліміт 5000 ітерацій; ls_eps - точність золотого перерізу по h):")
    print(f"{'сценарій':<52}{'x':<24}{'g':>11}{'F':>11}{'‖x-x*‖':>10}{'NΦ':>8}{'ітер.':>7}  збіжність")
    cfgs = [("одразу r = 1e4 зі старту, ls_eps = 1e-6", [1e4], 1e-6),
            ("r = 1 → 10 → … → 1e5 (теплий старт), ls_eps = 1e-6", [1, 10, 100, 1e3, 1e4, 1e5], 1e-6),
            ("одразу r = 1e6 зі старту, ls_eps = 1e-6", [1e6], 1e-6),
            ("r = 1 → … → 1e6 (теплий старт), ls_eps = 1e-6", [1, 10, 100, 1e3, 1e4, 1e5, 1e6], 1e-6),
            ("одразу r = 1e6 зі старту, ls_eps = 1e-10", [1e6], 1e-10),
            ("r = 1 → … → 1e6 (теплий старт), ls_eps = 1e-10", [1, 10, 100, 1e3, 1e4, 1e5, 1e6], 1e-10)]
    cmp_rows = {}
    for lab, rl, le in cfgs:
        xo, ro, _, _ = sumt(f_main, g_main, [CON_B], X0, r_list=rl, ls_eps=le)
        w = ro[-1]; conv = all(q['gnorm'] <= EPS for q in ro)
        nphi = sum(q['nphi'] for q in ro); its = sum(q['it'] for q in ro)
        cmp_rows[lab] = (nphi, its, conv, w)
        print(f"{lab:<52}{pt(w['x'], 4):<24}{w['g']:>11.2e}{w['F']:>11.4f}{np.linalg.norm(w['x'] - xs_b):>10.2e}{nphi:>8}{its:>7}  "
              + ("збіглась" if conv else f"НЕ збіглась на r = {rfmt(w['r'])}: ‖∇Φ‖ = {w['gnorm']:.1f} після {w['it']} ітер."))
    print(f"Обумовленість H_Φ = H + 2r·aa': " + ", ".join(
        f"r={rfmt(q)}: {np.linalg.cond(HESS + 2*q*np.outer(CON_B.a, CON_B.a)):.2e}" for q in (1, 10, 100, 1e3, 1e4, 1e6)))
    print("Крок лінійного пошуку поперек межі h ≈ 1/λmax(HΦ) ≈ 1/(2r‖a‖²) = 1/(10r): при r = 1e6 це 1e-7, тобто менше за точність")
    print("золотого перерізу 1e-6 - зі ls_eps = 1e-6 такі кроки «нульові», і ФР застряє (теплий старт на r = 1e6 не збігся).")

    # ---------- П.6 графіки ----------
    print("\n" + SEP)
    print("П.4 Графіки")
    def contour_base(ax, con, xlim, ylim, levels):
        X, Y = np.meshgrid(np.linspace(*xlim, 600), np.linspace(*ylim, 600))
        Z = X**2 + 2*Y**2 + 2*X*Y - 9*X - 13*Y
        G = con.a[0]*X + con.a[1]*Y - con.beta
        ax.contourf(X, Y, G, levels=[-1e9, 0], colors=["#cfeccf"], alpha=0.6)
        cs = ax.contour(X, Y, Z, levels=levels, cmap="viridis", linewidths=0.9); ax.clabel(cs, fontsize=7)
        ax.contour(X, Y, G, levels=[0], colors="k", linewidths=1.8)
        ax.set_xlabel("x₁"); ax.set_ylabel("x₂"); ax.grid(alpha=.3)

    # B: контури + траєкторія
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    contour_base(ax, CON_B, (-8, 24), (-16, 10), [-24, 0, 25, 50, 100, 109, 200, 400, 700, 1000, 1500])
    ax.plot(*pathsB[0].T, "-", color="tab:orange", lw=1.5, label="Флетчер–Рівс усередині r = 1")
    xs_traj = [X0] + [w['x'] for w in rowsB]
    ax.plot(*np.array(xs_traj).T, "o--", color="tab:red", ms=5, lw=1.3, label="x(r): r = 1 … 10⁴")
    for w, off in zip(rowsB[:2], ((-52, 10), (14, 12))):
        ax.annotate(f"r={rfmt(w['r'])}", w['x'], textcoords="offset points", xytext=off, fontsize=9, color="tab:red")
    ax.annotate("r = 100 … 10⁴ зливаються з x*", xs_b, textcoords="offset points", xytext=(-20, -60), fontsize=9, color="tab:red",
                arrowprops=dict(arrowstyle="-", color="tab:red", lw=0.8))
    ax.plot(*X0, "k*", ms=14, label="Старт (-3; -2)")
    ax.plot(*xu, "bs", ms=9, label="Безумовний мінімум (2.5; 2)")
    ax.plot(*xs_b, "rX", ms=12, label=f"Умовний x* = ({xs_b[0]:.2f}; {xs_b[1]:.2f})")
    ax.set_title("Варіант B: g = 35 − x₁ + 2x₂ ≤ 0 (зелена область Ω), траєкторія x(r)")
    ax.legend(fontsize=8, loc="upper left"); ax.set_xlim(-8, 24); ax.set_ylim(-16, 10)
    fig.savefig("lr11_contours_B.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # A: обмеження неактивне
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    contour_base(ax, CON_A, (-8, 24), (-16, 10), [-24, 0, 25, 50, 100, 109, 200, 400, 700, 1000, 1500])
    ax.plot(*np.array([X0] + [w['x'] for w in rowsA]).T, "o-", color="tab:red", ms=6, label="x(r) (збігається з безумовним мінімумом)")
    ax.plot(*X0, "k*", ms=14, label="Старт (-3; -2)")
    ax.plot(*xu, "bs", ms=9, label="Безумовний мінімум = x*")
    ax.plot(*xb, "mX", ms=11, label=f"Мінімум на межі ({xb[0]:.2f}; {xb[1]:.2f}), F = {f_main(xb):.1f}")
    ax.set_title("Варіант A: x₁ − 2x₂ ≤ 35 (зелена область Ω): обмеження неактивне")
    ax.legend(fontsize=8, loc="upper left"); ax.set_xlim(-8, 24); ax.set_ylim(-16, 10)
    fig.savefig("lr11_contours_A.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # g(x(r)) від r, лог-шкала + 2r·g
    rr = np.array([w['r'] for w in rowsB]); gg = np.array([w['g'] for w in rowsB]); lm = np.array([w['lam'][0] for w in rowsB])
    rgrid = np.logspace(0, 6, 200)
    ga = np.array([analytic_xr(CON_B, q)[1] for q in rgrid])
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2))
    axs[0].loglog(rgrid, ga, "-", color="tab:blue", label="g(x(r)) аналітично")
    axs[0].loglog(rr, gg, "o", color="tab:red", ms=7, label="g(x(r)) чисельно (ФР)")
    axs[0].loglog(rgrid, lam_b/(2*rgrid), "--", color="gray", label="λ/(2r)")
    axs[0].set_xlabel("r"); axs[0].set_ylabel("порушення g(x(r))"); axs[0].grid(alpha=.3, which="both"); axs[0].legend()
    axs[0].set_title("Порушення обмеження спадає як 1/r")
    axs[1].semilogx(rgrid, 2*rgrid*ga, "-", color="tab:blue", label="2r·g аналітично")
    axs[1].semilogx(rr, lm, "o", color="tab:red", ms=7, label="2r·g чисельно")
    axs[1].axhline(lam_b, color="k", ls="--", label=f"λ = {lam_b:g}")
    axs[1].set_xlabel("r"); axs[1].set_ylabel("2r·g(x(r))"); axs[1].grid(alpha=.3, which="both"); axs[1].legend()
    axs[1].set_title("Оцінка множника Лагранжа → λ")
    fig.tight_layout(); fig.savefig("lr11_violation.png", dpi=140); plt.close(fig)

    # графік для A: g(x(r)) та 2r·max(0,g)
    rA = np.array([1, 10, 100, 1e3, 1e4])
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    axs[0].semilogx(rA, [w['g'] for w in rowsA2], "o-", color="tab:red", label="g(x(r)) чисельно")
    axs[0].axhline(0, color="k", lw=1.5, label="межа g = 0 (Ω: g ≤ 0)")
    axs[0].set_xlabel("r"); axs[0].set_ylabel("g(x(r))"); axs[0].grid(alpha=.3, which="both"); axs[0].legend()
    axs[0].set_title("Обмеження неактивне: g(x(r)) ≈ −36.5 < 0 при всіх r")
    axs[1].semilogx(rA, [w['lam'][0] for w in rowsA2], "o-", color="tab:blue", label="2r·max(0, g)")
    axs[1].axhline(0, color="k", ls="--", label="λ = 0")
    axs[1].set_ylim(-1, 1); axs[1].set_xlabel("r"); axs[1].set_ylabel("оцінка множника"); axs[1].grid(alpha=.3, which="both"); axs[1].legend()
    axs[1].set_title("Оцінка множника Лагранжа = 0")
    fig.tight_layout(); fig.savefig("lr11_violation_A.png", dpi=140); plt.close(fig)
    print("Збережено: lr11_contours_A.png, lr11_violation_A.png (варіант A); lr11_contours_B.png, lr11_violation.png (додаток, варіант B)")


if __name__ == "__main__":
    main()