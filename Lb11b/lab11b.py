"""ЛР 11, частина 2. Штрафні й бар'єрні функції: c, внутрішній ε, рівність, лог-бар'єр, два обмеження, коло.
F = x1^2 + 2x2^2 + 2x1x2 - 9x1 - 13x2,  x0 = (-3; -2)
Обмеження: (1) x1 - 2x2 <= 35;  (2) x2 <= -11;  коло x1^2 + x2^2 >= 441.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

X0 = np.array([-3.0, -2.0])
HESS = np.array([[2.0, 2.0], [2.0, 4.0]]); BVEC = np.array([9.0, 13.0])
EPS = 0.01
INF = float("inf")

f_main = lambda x: x[0]**2 + 2*x[1]**2 + 2*x[0]*x[1] - 9*x[0] - 13*x[1]
g_main = lambda x: np.array([2*x[0] + 2*x[1] - 9, 2*x[0] + 4*x[1] - 13])


# ======================= обмеження =======================
class Con:
    """g(x) <= 0 (kind='ineq') або h(x) = 0 (kind='eq')."""
    def __init__(self, g, grad, kind, name):
        self.g, self.grad, self.kind, self.name = g, grad, kind, name
    def s(self, x):                      
        v = self.g(x)
        return max(0.0, v) if self.kind == "ineq" else v

def lin(a, beta, kind, name):
    a = np.array(a, float)
    return Con(lambda x: float(a @ x - beta), lambda x: a, kind, name)

C1 = lin([1, -2], 35, "ineq", "x1 - 2x2 <= 35")
C1EQ = lin([1, -2], 35, "eq", "x1 - 2x2 = 35")
C2 = lin([0, 1], -11, "ineq", "x2 <= -11")
CIRC = Con(lambda x: 441.0 - float(x @ x), lambda x: -2*np.asarray(x, float), "ineq", "x1^2 + x2^2 >= 441")


# ======================= лічильники та Φ =======================
class Obj:
    def __init__(self, f, g):
        self._f, self._g, self.nf, self.ng = f, g, 0, 0
    def F(self, x):
        self.nf += 1; return self._f(np.asarray(x, float))
    def grad(self, x):
        self.ng += 1; return self._g(np.asarray(x, float))

def penalty(F, gradF, cons, r):
    """Φr = F + r·Σ s_j^2;  ∇Φr = ∇F + 2r·Σ s_j·∇g_j (s = max(0,g) для нерівності, h для рівності)."""
    def Phi(x): return F(x) + r*sum(c.s(x)**2 for c in cons)
    def gradPhi(x):
        v = np.array(gradF(x), float)
        for c in cons:
            sj = c.s(x)
            if sj != 0.0: v = v + 2*r*sj*c.grad(x)
        return v
    return Phi, gradPhi

def barrier(F, gradF, cons, mu):
    """Φμ = F - μ·Σ ln(-g_j); поза Ω (будь-яке g >= 0) Φ = +∞."""
    def Phi(x):
        gs = [c.g(x) for c in cons]
        if max(gs) >= 0: return INF
        return F(x) - mu*sum(np.log(-v) for v in gs)
    def gradPhi(x):
        v = np.array(gradF(x), float)
        for c in cons: v = v + (-mu/c.g(x))*c.grad(x)
        return v
    return Phi, gradPhi


# ======================= Свенн + золотий + Флетчер-Рівс =======================
def svenn(phi, delta0=0.01):
    pts, vals = [0.0], [phi(0.0)]; h, step = 0.0, delta0
    for _ in range(200):
        h += step; v = phi(h); pts.append(h); vals.append(v)
        if not (v <= vals[-2]): break              
        step *= 2
    return (pts[-3] if len(pts) >= 3 else pts[0]), pts[-1]

def golden(phi, a, b, eps=1e-6):
    t = (np.sqrt(5) - 1) / 2
    x1, x2 = b - t*(b - a), a + t*(b - a); f1, f2 = phi(x1), phi(x2)
    while b - a > eps:
        if f1 < f2 or (f1 == INF and f2 == INF):      
            b, x2, f2 = x2, x1, f1; x1 = b - t*(b - a); f1 = phi(x1)
        else:
            a, x1, f1 = x1, x2, f2; x2 = a + t*(b - a); f2 = phi(x2)
    m = (a + b) / 2
    if (f1 == INF or f2 == INF) and phi(m) == INF:   
        m = a
    return m

def fletcher_reeves(obj, x0, eps=EPS, restart=2, max_iter=5000, ls_eps=1e-6):
    x = np.array(x0, float); g = obj.grad(x); d = -g; k = 0
    gn = float(np.linalg.norm(g))
    while gn > eps and k < max_iter:
        if not np.isfinite(gn) or gn > 1e100:        
            break
        phi = lambda h: obj.F(x + h*d)
        a, b = svenn(phi, delta0=0.01)
        h = golden(phi, a, b, ls_eps)
        x = x + h*d; g_new = obj.grad(x)
        gn_new = float(np.linalg.norm(g_new))
        if not np.isfinite(gn_new) or gn_new > 1e100:
            g, gn = g_new, gn_new
            break
        beta = (g_new @ g_new) / (g @ g)
        if (k + 1) % restart == 0: beta = 0.0
        d = -g_new + beta*d
        if g_new @ d >= 0:                            
            d = -g_new
        g, gn = g_new, gn_new; k += 1
    return x, k, gn

def spherical(obj, x0, h0=1.0, eps=EPS, max_iter=5000):
    """ЛР 8: сферична метрика, ділення кроку навпіл (крок не повертається)."""
    x = np.array(x0, float); h = h0; Fx = obj.F(x); g = obj.grad(x); k = 0
    while np.linalg.norm(g) > eps and k < max_iter:
        d = -g / np.linalg.norm(g)
        while True:
            xn = x + h*d; Fn = obj.F(xn)
            if Fn >= Fx:
                h /= 2
                if h < 1e-14: break
            else: break
        if h < 1e-14 and Fn >= Fx: break
        x, Fx = xn, Fn; g = obj.grad(x); k += 1
    return x, k, float(np.linalg.norm(g))

def ls_rule(r, adapt=True):
    """Точність золотого перерізу: 1e-6; при r > 1e4 звужується до 1e-2/r (див. частину 1: крок ~ 1/(10r))."""
    return min(1e-6, 1e-2/r) if adapt else 1e-6


# ======================= SUMT: зовнішній штраф =======================
def sumt_ext(F, gradF, cons, x0, r0=1.0, c=10.0, eps=EPS, inner_eps=EPS, r_list=None, max_outer=40,
             adapt=True, inner="fr"):
    x = np.array(x0, float); r = r0 if r_list is None else r_list[0]
    rows, tot = [], 0
    for k in range(1, max_outer + 1):
        Phi, gradPhi = penalty(F, gradF, cons, r)
        o = Obj(Phi, gradPhi); xs = x.copy()
        if inner == "fr": x, it, gn = fletcher_reeves(o, x, eps=inner_eps, ls_eps=ls_rule(r, adapt))
        else:             x, it, gn = spherical(o, x, eps=inner_eps)
        tot += o.nf
        ss = [cn.s(x) for cn in cons]
        P = sum(v*v for v in ss)
        viol = max(abs(v) for v in ss)
        rows.append(dict(k=k, r=r, start=xs, it=it, nphi=o.nf, ngphi=o.ng, x=x.copy(), F=F(x), viol=viol, s=ss,
                         P=P, rP=r*P, lam=[2*r*v for v in ss], tot=tot, gn=gn, conv=gn <= inner_eps))
        if r_list is None:
            if r*P <= eps: break
            r *= c
        else:
            if k >= len(r_list): break
            r = r_list[k]
    return x, rows


# ======================= внутрішній штраф (бар'єр) =======================
def sumt_int(F, gradF, cons, x0, mu0=10.0, c=10.0, eps=EPS, inner_eps=EPS, max_outer=12, extra=0):
    x = np.array(x0, float)
    assert max(cn.g(x) for cn in cons) < 0, "старт має бути строго всередині Ω"
    mu, rows, tot, m = mu0, [], 0, len(cons)
    stop_at = None
    for k in range(1, max_outer + 1):
        Phi, gradPhi = barrier(F, gradF, cons, mu)
        o = Obj(Phi, gradPhi); xs = x.copy()
        x, it, gn = fletcher_reeves(o, x, eps=inner_eps, ls_eps=min(1e-6, 1e-3*mu))
        tot += o.nf
        gs = [cn.g(x) for cn in cons]
        rows.append(dict(k=k, mu=mu, start=xs, it=it, nphi=o.nf, ngphi=o.ng, x=x.copy(), F=F(x), g=gs,
                         lam=[-mu/v for v in gs], tot=tot, gn=gn, conv=gn <= inner_eps, inside=max(gs) < 0))
        if stop_at is None and mu*m <= eps: stop_at = k
        if stop_at is not None and k >= stop_at + extra: break
        mu /= c
    return x, rows, stop_at


# ======================= аналітика =======================
def kkt_active(a_list, beta_list):
    """min F при a_j·x = beta_j (активні): [H A'; A 0][x; λ] = [b; β] ->  ∇F + Σ λ_j a_j = 0."""
    A = np.array(a_list, float); m = len(A)
    M = np.block([[HESS, A.T], [A, np.zeros((m, m))]])
    sol = np.linalg.solve(M, np.concatenate([BVEC, np.array(beta_list, float)]))
    return sol[:2], sol[2:]

def analytic_penalty(a_list, beta_list, r):
    """Мінімум Φr = F + r·Σ (a_j·x - β_j)^2 (усі обмеження «діють»): (H + 2r·Σ a a')x = b + 2r·Σ β a."""
    A = np.array(a_list, float); Bt = np.array(beta_list, float)
    Hm = HESS + 2*r*sum(np.outer(a, a) for a in A)
    return np.linalg.solve(Hm, BVEC + 2*r*sum(b*a for a, b in zip(A, Bt)))

def circle_minima(R=21.0):
    th = np.linspace(0, 2*np.pi, 400001)
    f = lambda t: f_main(np.array([R*np.cos(t), R*np.sin(t)]))
    vals = np.array([f(t) for t in th[:-1]])
    n = len(vals); mins = []
    for i in range(n):
        if vals[i] < vals[i-1] and vals[i] <= vals[(i+1) % n]:
            t = th[i]; lo, hi = t - 1e-3, t + 1e-3
            t = golden(f, lo, hi, 1e-12); x = np.array([R*np.cos(t), R*np.sin(t)])
            lam = float(g_main(x) @ x / (2*(x @ x)))
            mins.append((x, f_main(x), lam))
    return sorted(mins, key=lambda m: m[1])


# ======================= форматування =======================
def num(v, nd=6):
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s
def pt(v, nd=5): return "(" + "; ".join(num(t, nd) for t in v) + ")"
def rf(r): return f"{r:g}"
def flag(rows): return "так" if all(w['conv'] for w in rows) else "НІ (ліміт ітерацій)"

def print_ext(rows, nsj=1, names=None):
    sh = "".join(f"{('s'+str(j+1)):>10}" for j in range(nsj))
    lh = "".join(f"{('2r·s'+str(j+1)):>10}" for j in range(nsj))
    hdr = f"{'k':>2}{'r':>9} {'старт':<20}{'іт.':>5}{'NΦ':>7}{'N∇Φ':>6}  {'x(r)':<26}{'F(x(r))':>13}{sh}{'r·P':>11}{lh}{'NΦ разом':>9}"
    print(hdr); print("-"*len(hdr))
    for w in rows:
        ss = "".join(f"{v:>10.5f}" for v in w['s']); ll = "".join(f"{v:>10.4f}" for v in w['lam'])
        print(f"{w['k']:>2}{rf(w['r']):>9} {pt(w['start'],3):<20}{w['it']:>5}{w['nphi']:>7}{w['ngphi']:>6}  {pt(w['x'],6):<26}{w['F']:>13.5f}{ss}{w['rP']:>11.3e}{ll}{w['tot']:>9}")

def print_int(rows, Fstar=None, nlam=1):
    gh = "".join(f"{('g'+str(j+1)):>11}" for j in range(nlam)); lh = "".join(f"{('μ/(-g'+str(j+1)+')'):>11}" for j in range(nlam))
    hdr = f"{'k':>2}{'μ':>9} {'x(μ)':<26}{gh}{'F(x(μ))':>13}{'F-F*':>11}{lh}{'іт.':>5}{'NΦ':>7}{'NΦ разом':>9}{'в Ω':>5}{'збіг.':>7}"
    print(hdr); print("-"*len(hdr))
    for w in rows:
        gs = "".join(f"{v:>11.5f}" for v in w['g']); ls = "".join(f"{v:>11.4f}" for v in w['lam'])
        d = f"{w['F'] - Fstar:>11.5f}" if Fstar is not None else f"{'':>11}"
        print(f"{w['k']:>2}{w['mu']:>9g} {pt(w['x'],5):<26}{gs}{w['F']:>13.5f}{d}{ls}{w['it']:>5}{w['nphi']:>7}{w['tot']:>9}{('так' if w['inside'] else 'НІ'):>5}{('так' if w['conv'] else 'НІ'):>7}")

SUMMARY = []
SUM_C, SUM_E = [], []
def add_summary(name, rows, F_val, lam, viol, x, n_outer=None, ng=None, nphi=None):
    SUMMARY.append(dict(name=name, n=len(rows) if n_outer is None else n_outer,
                        nphi=rows[-1]['tot'] if nphi is None else nphi,
                        ng=sum(w['ngphi'] for w in rows) if ng is None else ng, x=x, viol=viol, F=F_val, lam=lam))


def main():
    SEP = "=" * 128
    xu = np.array([2.5, 2.0])
    # ---------- П.0 ----------
    print(SEP); print("П.0 Постановка та аналітика (для контролю)")
    print("F = x1^2 + 2x2^2 + 2x1x2 - 9x1 - 13x2,  x0 = (-3; -2),  безумовний мінімум (2.5; 2), F = -24.25")
    print("Обмеження: (1) g1 = x1 - 2x2 - 35 <= 0;  (2) g2 = x2 + 11 <= 0;  коло g3 = 441 - x1^2 - x2^2 <= 0 (R = 21)")
    print(f"g1(мін) = {C1.g(xu):.2f} (виконано), g2(мін) = {C2.g(xu):.2f} (порушено), g3(мін) = {CIRC.g(xu):.2f} (порушено); g1(старт) = {C1.g(X0):.0f}, g2(старт) = {C2.g(X0):.0f}")
    xE, lamE = kkt_active([[1, -2]], [35]); FE = f_main(xE)
    print(f"\nРівність h = x1 - 2x2 - 35 = 0: x*_E = {pt(xE,4)}, F = {FE:.5f}, λ_h = {lamE[0]:.4f} (∇F + λ∇h = 0, знак довільний)")
    x2a, l2a = kkt_active([[0, 1]], [-11])
    print(f"Обмеження (2) окремо: x* = {pt(x2a,4)}, F = {f_main(x2a):.4f}, λ2 = {l2a[0]:.3f}; але g1 там = {C1.g(x2a):.2f} > 0 (порушується обмеження 1)")
    xB, lB = kkt_active([[1, -2], [0, 1]], [35, -11]); FB = f_main(xB)
    print(f"Обмеження (1)+(2) разом (Куна-Таккер, обидва активні): x* = {pt(xB,4)}, F* = {FB:.4f}, λ = ({lB[0]:.4f}; {lB[1]:.4f})  -> обидва ≥ 0")
    print(f"   перевірка: ∇F(x*) = {pt(g_main(xB),4)} = -λ1·(1; -2) - λ2·(0; 1) = {pt(-lB[0]*np.array([1,-2]) - lB[1]*np.array([0,1]),4)}")
    cm = circle_minima()
    print("Коло x1^2 + x2^2 = 441 (мінімуми F на колі, чисельно за кутом):")
    for x, fv, lam in cm:
        print(f"   x = {pt(x,4)}, ‖x‖ = {np.linalg.norm(x):.4f}, F = {fv:.4f}, λ = {lam:.5f} (∇F = -λ∇g = 2λx)")

    # ---------- П.1 самоперевірка ----------
    print("\n" + SEP); print("П.1 Самоперевірка реалізації на прикладах зі слайдів (F = 2x1^2+3x2^2+x1x2, g = 4 - x1 - x2 <= 0)")
    fs = lambda x: 2*x[0]**2 + 3*x[1]**2 + x[0]*x[1]
    gs_ = lambda x: np.array([4*x[0] + x[1], x[0] + 6*x[1]])
    gA = lin([-1, -1], -4, "ineq", "4-x1-x2<=0")
    _, rb, _ = sumt_int(fs, gs_, [gA], [5, 1], extra=0)
    print(f"Бар'єр зі старту (5; 1): μ = 10 → 0.01:  ітерацій {[w['it'] for w in rb]}, NΦ = {[w['nphi'] for w in rb]} (разом {rb[-1]['tot']}), "
          f"N∇Φ = {sum(w['ngphi'] for w in rb)}; x(10) = {pt(rb[0]['x'],5)}, F(0.01) = {rb[-1]['F']:.4f}")
    print("   за слайдами: 4,4,5,2; NΦ 132,125,138,48 (443); N∇Φ = 19; x(10) = (2.95913; 1.77547); F = 23.010")
    gB2 = lin([0, 1], 1, "ineq", "x2<=1")
    _, r2 = sumt_ext(fs, gs_, [gA, gB2], [5, -3])
    print(f"Два обмеження, старт (5; -3): ітерацій {[w['it'] for w in r2]}, NΦ = {r2[-1]['tot']}, N∇Φ = {sum(w['ngphi'] for w in r2)}, "
          f"x(1e4) = {pt(r2[-1]['x'],4)}, 2r·s = ({r2[-1]['lam'][0]:.1f}; {r2[-1]['lam'][1]:.1f})   (слайди: 453, 22, (2.9992; 1.0002), (13.0; 4.0))")
    circ3 = Con(lambda x: 9.0 - float(x @ x), lambda x: -2*np.asarray(x, float), "ineq", "R=3")
    _, rc = sumt_ext(fs, gs_, [circ3], [5, -3])
    print(f"Коло R = 3, старт (5; -3): NΦ = {rc[-1]['tot']}, N∇Φ = {sum(w['ngphi'] for w in rc)}, r = 100: ‖x‖ = {np.linalg.norm(rc[-1]['x']):.4f}, F = {rc[-1]['F']:.3f}   (слайд: 255, 12, 2.9985, 16.12)")
    hEq = lin([-1, -1], -4, "eq", "h=4-x1-x2")
    _, re_ = sumt_ext(fs, gs_, [hEq], [5, -3])
    print(f"Рівність x1 + x2 = 4: NΦ = {re_[-1]['tot']}, N∇Φ = {sum(w['ngphi'] for w in re_)}   (слайд: 368, 18)")
    ok2 = r2[-1]['tot'] == 453 and sum(w['ngphi'] for w in r2) == 22
    okc = rc[-1]['tot'] == 255 and sum(w['ngphi'] for w in rc) == 12
    oke = re_[-1]['tot'] == 368 and sum(w['ngphi'] for w in re_) == 18
    okb = abs(rb[0]['x'][0] - 2.95913) < 1e-5 and abs(rb[-1]['F'] - 23.010) < 5e-4 and [w['nphi'] for w in rb[:2]] == [132, 125]
    print(f"Збіг: два обмеження {'ТАК' if ok2 else 'НІ'}, коло {'ТАК' if okc else 'НІ'}, рівність {'ТАК' if oke else 'НІ'}; "
          f"бар'єр: x(μ) і F збігаються, NΦ перших двох стадій збігаються (132, 125) - {'ТАК' if okb else 'НІ'}, "
          f"але 3-тя стадія має {rb[2]['it']} ітерації замість 5 (NΦ {rb[-1]['tot']} замість 443) - різниця в одному лінійному пошуку біля стіни")

    # ---------- П.2 Завдання 1: c ----------
    print("\n" + SEP); print("П.2 Завдання 1. Множник c (r0 = 1, ε = 0.01, внутрішній ε = 0.01, ФР, теплий старт, зупинка r·P ≤ ε)")
    cres = {}
    for pname, cons, xs_, Fs_ in (("Е: рівність x1 - 2x2 = 35", [C1EQ], xE, FE), ("Д: обмеження (1)+(2) разом", [C1, C2], xB, FB)):
        print(f"\nЗадача {pname}; точна відповідь {pt(xs_,4)}, F* = {Fs_:.4f}")
        print(f"{'c':>4}{'зовн.ітер.':>11}{'кінц. r':>11}{'NΦ':>8}{'N∇Φ':>6}{'порушення':>12}  {'x':<26}{'‖x-x*‖':>10}{'збіг.':>7}  оцінка log_c(r_кін/r0)")
        for cc in (2, 4, 10, 100):
            x, rows = sumt_ext(f_main, g_main, cons, X0, c=cc)
            w = rows[-1]; cres[(pname, cc)] = rows
            est = np.log(w['r']/1.0)/np.log(cc)
            SUM_C.append((pname[:1], cc, rows, xs_))
            print(f"{cc:>4}{len(rows):>11}{rf(w['r']):>11}{w['tot']:>8}{sum(q['ngphi'] for q in rows):>6}{w['viol']:>12.5f}  {pt(w['x'],4):<26}{np.linalg.norm(w['x']-xs_):>10.2e}{'так' if all(q['conv'] for q in rows) else 'НІ':>7}  {est:.1f}")
    print("Для обмеження (1) окремо (неактивного) c не має значення: зупинка при r = 1, 1 зовнішня ітерація, NΦ = 76 (частина 1).")

    # ---------- П.3 Завдання 2: внутрішній ε ----------
    print("\n" + SEP); print("П.3 Завдання 2. Внутрішній ε (‖∇Φ‖ ≤ ε_вн): 0.1 / 0.01 / 0.001 (c = 10, зовнішній ε = 0.01, прогін r = 1 … r_кін)")
    eres = {}
    for pname, cons, xs_ in (("Е: рівність x1 - 2x2 = 35", [C1EQ], xE), ("Д: обмеження (1)+(2) разом", [C1, C2], xB)):
        print(f"\nЗадача {pname}")
        print(f"{'ε_вн':>7}{'зовн.ітер.':>11}{'кінц. r':>10}{'NΦ':>8}{'N∇Φ':>6}  {'x при r_кін':<26}{'‖x-x*‖':>10}{'порушення':>12}{'збіг.':>7}")
        for ei in (0.1, 0.01, 0.001):
            x, rows = sumt_ext(f_main, g_main, cons, X0, inner_eps=ei)
            w = rows[-1]; eres[(pname, ei)] = rows
            SUM_E.append((pname[:1], ei, rows, xs_))
            print(f"{ei:>7g}{len(rows):>11}{rf(w['r']):>10}{w['tot']:>8}{sum(q['ngphi'] for q in rows):>6}  {pt(w['x'],5):<26}{np.linalg.norm(w['x']-xs_):>10.2e}{w['viol']:>12.6f}{'так' if all(q['conv'] for q in rows) else 'НІ':>7}")
    print("Порушення задає r (воно ≈ λ/(2r)), а положення вздовж межі залежить від внутрішнього ε.")

    # ---------- П.4 Завдання 3: рівність ----------
    print("\n" + SEP); print("П.4 Завдання 3. Рівність h = x1 - 2x2 - 35 = 0: штраф r·h² (без зрізки), r = 1 … r_кін")
    xe, rowsE = sumt_ext(f_main, g_main, [C1EQ], X0)
    print(f"Аналітично: x*_E = {pt(xE,4)}, F = {FE:.5f}, λ_h = {lamE[0]:.4f};  оцінка λ = 2r·h")
    print_ext(rowsE)
    print(f"Результат: x = {pt(xe,6)}, F = {rowsE[-1]['F']:.5f}, |h| = {rowsE[-1]['viol']:.3e}, 2r·h = {rowsE[-1]['lam'][0]:.4f}; NΦ = {rowsE[-1]['tot']}, N∇Φ = {sum(w['ngphi'] for w in rowsE)}")
    print("\nПеревірка аналітикою: (H + 2r·aa')x = b + 2rβ·a (для рівності справедлива при будь-якому знаку h)")
    print(f"{'r':>9} {'x чисельне':<26}{'x аналітичне':<26}{'‖Δx‖':>10}{'h чисельне':>13}{'h аналіт.':>12}{'2r·h':>10}")
    a1 = np.array([1.0, -2.0])
    for w in rowsE:
        xa = analytic_penalty([a1], [35], w['r'])
        print(f"{rf(w['r']):>9} {pt(w['x'],5):<26}{pt(xa,5):<26}{np.linalg.norm(w['x']-xa):>10.2e}{w['s'][0]:>13.6f}{C1EQ.g(xa):>12.6f}{w['lam'][0]:>10.4f}")
    print(f"Теорія: 2r·h → λ_h = {lamE[0]:.4f}; h ≈ λ_h/(2r) (h < 0: наближення з боку безумовного мінімуму)")
    xi, rowsI = sumt_ext(f_main, g_main, [C1], X0)
    print(f"\nПорівняння з НЕРІВНІСТЮ x1 - 2x2 <= 35 (та сама межа): x = {pt(xi,5)}, F = {rowsI[-1]['F']:.4f}, штраф 0, 1 зовнішня ітерація, NΦ = {rowsI[-1]['tot']}")
    print(f"   рівність: x = {pt(xe,5)}, F = {rowsE[-1]['F']:.3f}; нерівність: x = (2.5; 2), F = -24.25. Різниця: безумовний мінімум нерівність виконує, а рівність вимагає лежати НА лінії")
    add_summary("Рівність x1-2x2=35, r=1…1e4 (ФР)", rowsE, rowsE[-1]['F'], f"{rowsE[-1]['lam'][0]:.3f}", rowsE[-1]['viol'], xe)
    add_summary("Нерівність x1-2x2<=35 (неактивна)", rowsI, rowsI[-1]['F'], "0", 0.0, xi)

    # ---------- П.5 Завдання 4: бар'єр ----------
    print("\n" + SEP); print("П.5 Завдання 4. Лог-бар'єр Φμ = F - μ·Σ ln(-g_j), μ0 = 10, c = 10, стоп μ·m ≤ ε = 0.01; Φ = +∞ поза Ω")
    # (а) обмеження 1 окремо, старт (-3;-2) - строго всередині
    print("\n(а) Обмеження (1) окремо, старт (-3; -2): g1 = -34 < 0 (строго всередині), x* = (2.5; 2), F* = -24.25, λ = 0")
    xa_, rowsA, st = sumt_int(f_main, g_main, [C1], X0, extra=1)
    print_int(rowsA, Fstar=-24.25)
    print(f"   F(x(μ)) ≥ F* для всіх μ (спадає до -24.25 зверху); μ/(-g) = {rowsA[st-1]['lam'][0]:.5f} → λ = 0; усі точки в Ω")
    # (б) обидва обмеження, внутрішній старт
    xin = np.array([0.0, -12.0])
    print(f"\n(б) Обмеження (1)+(2): старт (-3; -2) НЕ внутрішній (g2 = 9 > 0), тому внутрішній старт {pt(xin,1)}: g = ({C1.g(xin):.0f}; {C2.g(xin):.0f}) < 0, F = {f_main(xin):.0f}")
    xb_, rowsBi, stb = sumt_int(f_main, g_main, [C1, C2], xin, extra=1)
    print_int(rowsBi, Fstar=FB, nlam=2)
    print(f"   F* = {FB:.4f}, λ = ({lB[0]:.3f}; {lB[1]:.3f}); μ/(-g_j) → λ_j")
    wb = rowsBi[stb-1]
    print(f"Стоп при μ = {wb['mu']:g}: x = {pt(wb['x'],5)}, F = {wb['F']:.4f}, NΦ = {wb['tot']}, N∇Φ = {sum(q['ngphi'] for q in rowsBi[:stb])}")
    # двостороння оцінка
    xp, rowsP = sumt_ext(f_main, g_main, [C1, C2], X0)
    Fr, Fm = rowsP[-1]['F'], wb['F']
    print(f"\nДвостороння оцінка: F(x(r)) ≤ F* ≤ F(x(μ)):  {Fr:.4f} ≤ {FB:.4f} ≤ {Fm:.4f}   (ширина інтервалу {Fm - Fr:.4f}; точне F* = {FB:.4f})")
    print(f"   зовнішній штраф: r = {rf(rowsP[-1]['r'])}, x = {pt(xp,5)}, порушення {rowsP[-1]['viol']:.2e}, NΦ = {rowsP[-1]['tot']};  бар'єр: x = {pt(wb['x'],5)}, g = ({wb['g'][0]:.5f}; {wb['g'][1]:.5f}) < 0")
    add_summary("Бар'єр (1)+(2), μ=10…0.001", rowsBi[:stb], wb['F'], f"({wb['lam'][0]:.2f}; {wb['lam'][1]:.2f})", max(0.0, max(wb['g'])), wb['x'],
                ng=sum(q['ngphi'] for q in rowsBi[:stb]), nphi=wb['tot'])

    # ---------- П.6 Завдання 5: два обмеження ----------
    print("\n" + SEP); print("П.6 Завдання 5. Два обмеження (1)+(2): P = max(0,g1)² + max(0,g2)², ∇Φ = ∇F + 2r·s1·∇g1 + 2r·s2·∇g2")
    print(f"Куна-Таккер: x* = {pt(xB,4)}, F* = {FB:.4f}, λ = ({lB[0]:.4f}; {lB[1]:.4f}) обидва ≥ 0 -> обидва обмеження активні")
    xp2, rowsT = sumt_ext(f_main, g_main, [C1, C2], X0)
    print_ext(rowsT, nsj=2)
    w = rowsT[-1]
    print(f"Результат: x ≈ {pt(w['x'],5)}, F = {w['F']:.4f}, порушення ({w['s'][0]:.5f}; {w['s'][1]:.5f}), 2r·s = ({w['lam'][0]:.3f}; {w['lam'][1]:.3f}) → λ = ({lB[0]:.2f}; {lB[1]:.2f}); NΦ = {w['tot']}, N∇Φ = {sum(q['ngphi'] for q in rowsT)}")
    print("Аналітична перевірка штрафної системи: (H + 2r·(a1a1' + a2a2'))x = b + 2r·(β1a1 + β2a2) (правильна, коли обидва s > 0):")
    a_l = [[1, -2], [0, 1]]; b_l = [35, -11]
    print(f"{'r':>9} {'x чисельне':<26}{'x аналітичне':<26}{'‖Δx‖':>10}{'g1':>12}{'g2':>12}  обидва порушені?")
    for q in rowsT:
        xa = analytic_penalty(a_l, b_l, q['r']); g1a, g2a = C1.g(xa), C2.g(xa)
        print(f"{rf(q['r']):>9} {pt(q['x'],5):<26}{pt(xa,5):<26}{np.linalg.norm(q['x']-xa):>10.2e}{g1a:>12.5f}{g2a:>12.5f}  {'так' if g1a > 0 and g2a > 0 else 'ні -> система не застосовна'}")
    print("Обмеження (1) стає порушеним лише коли обмеження (2) притискає розв'язок до x2 = -11: до цього воно неактивне (s1 = 0 при малих r).")
    add_summary("Два обмеження (1)+(2), r=1…r_кін", rowsT, w['F'], f"({w['lam'][0]:.2f}; {w['lam'][1]:.2f})", w['viol'], w['x'])

    # ---------- П.7 Завдання 6: коло ----------
    print("\n" + SEP); print("П.7 Завдання 6. Коло x1² + x2² ≥ 441 (g = 441 - x1² - x2², ∇g = (-2x1; -2x2) залежить від точки), два старти")
    print("Мінімуми F на колі (точно):")
    for x, fv, lam in cm: print(f"   x = {pt(x,4)}, F = {fv:.4f}, λ = {lam:.5f}")
    circ_runs = {}
    for lab, st_ in (("старт 1: (-3; -2)", X0), ("старт 2: (5; -3)", np.array([5.0, -3.0]))):
        print(f"\n{lab}, F(старт) = {f_main(st_):.2f}, g = {CIRC.g(st_):.1f} > 0 (порушено)")
        xc, rowsC = sumt_ext(f_main, g_main, [CIRC], st_); circ_runs[lab] = (xc, rowsC)
        print_ext(rowsC)
        w = rowsC[-1]
        best = min(cm, key=lambda m: np.linalg.norm(m[0] - w['x']))
        print(f"   збіглось до x ≈ {pt(w['x'],4)}, ‖x‖ = {np.linalg.norm(w['x']):.4f}, F = {w['F']:.4f}, порушення s = {w['s'][0]:.4f}, 2r·s = {w['lam'][0]:.4f}"
              f"  → мінімум {pt(best[0],3)} (F = {best[1]:.3f}, λ = {best[2]:.4f}); NΦ = {w['tot']}, N∇Φ = {sum(q['ngphi'] for q in rowsC)}")
        add_summary(f"Коло R=21, {lab}", rowsC, w['F'], f"{w['lam'][0]:.3f}", w['viol'], w['x'])
    print("Ω = {x1² + x2² ≥ 441} невипукла: два локальні мінімуми з різною F; порівнювати треба F з обох стартів.")

    # ---------- П.8 Завдання 7: внутрішні методи ----------
    print("\n" + SEP); print("П.8 Завдання 7. Внутрішній метод: сферична метрика ЛР 8 проти Флетчера-Рівса (рівність x1 - 2x2 = 35, ‖∇Φ‖ ≤ 0.01, ліміт 5000)")
    print(f"{'внутрішній метод':<32}{'задача':<28}{'ітер.':>7}{'NΦ':>9}{'N∇Φ':>7}{'|h|':>12}{'збіг.':>7}")
    imp = []
    for rr in (1, 10, 100):
        Phi, gP = penalty(f_main, g_main, [C1EQ], rr); o = Obj(Phi, gP)
        x, it, gn = spherical(o, X0)
        imp.append(("сферична метрика (ЛР 8)", f"r = {rr} зі старту", it, o.nf, o.ng, abs(C1EQ.g(x)), gn <= EPS))
    _, rw = sumt_ext(f_main, g_main, [C1EQ], X0, r_list=[1, 10, 100, 1e3, 1e4])
    imp.append(("Флетчер–Рівс (ЛР 9)", "r = 1 → … → 10⁴, теплий старт", sum(q['it'] for q in rw), rw[-1]['tot'], sum(q['ngphi'] for q in rw), rw[-1]['viol'], all(q['conv'] for q in rw)))
    for ad in (False, True):
        _, rs = sumt_ext(f_main, g_main, [C1EQ], X0, r_list=[1e6], adapt=ad)
        imp.append((f"Флетчер–Рівс, ls_eps {'адапт.' if ad else '1e-6'}", "r = 10⁶ зі старту", rs[-1]['it'], rs[-1]['nphi'], rs[-1]['ngphi'], rs[-1]['viol'], rs[-1]['conv']))
    # Ньютон: Φr квадратична (рівність) -> один крок
    for rr in (100, 1e6):
        a = np.array([1.0, -2.0]); Hphi = HESS + 2*rr*np.outer(a, a)
        Phi, gP = penalty(f_main, g_main, [C1EQ], rr)
        xn = X0 - np.linalg.solve(Hphi, gP(X0))
        imp.append(("Ньютон (H + 2r·aa' відомий)", f"r = {rf(rr)}, 1 крок", 1, 1, 1, abs(C1EQ.g(xn)), True))
    for (mth, prob, it, nphi, ng, hh, cv) in imp:
        hs = f"{hh:>12.3e}" if np.isfinite(hh) else f"{'розбіг':>12}"
        print(f"{mth:<32}{prob:<28}{it:>7}{nphi:>9}{ng:>7}{hs}{'так' if cv else 'НІ':>7}")
    print("Ньютон: Φr (рівність) квадратична всюди, тому один крок дає точне x(r); його вартість - 1 градієнт + 1 гесіан.")

    # ---------- П.9 зведена таблиця ----------
    print("\n" + SEP); print("П.9 Завдання 8. Зведена таблиця прогонів (c = 10, r0 = 1 / μ0 = 10, ε = 0.01, внутрішній ε = 0.01)")
    def lam_str(rows_):
        l_ = rows_[-1]['lam']; return f"{l_[0]:.2f}" if len(l_) == 1 else f"({l_[0]:.2f}; {l_[1]:.2f})"
    for (pn, cc, rows_, xs_) in SUM_C:
        if cc != 10: add_summary(f"{pn}: c = {cc}", rows_, rows_[-1]['F'], lam_str(rows_), rows_[-1]['viol'], rows_[-1]['x'])
    for (pn, ei, rows_, xs_) in SUM_E:
        if ei != 0.01: add_summary(f"{pn}: ε_вн = {ei:g}", rows_, rows_[-1]['F'], lam_str(rows_), rows_[-1]['viol'], rows_[-1]['x'])
    print(f"{'прогін':<44}{'зовн.':>6}{'NΦ':>8}{'N∇Φ':>6}  {'відповідь':<26}{'порушення':>12}{'F':>13}  λ (оцінка)")
    for s_ in SUMMARY:
        print(f"{s_['name']:<44}{s_['n']:>6}{s_['nphi']:>8}{s_['ng']:>6}  {pt(s_['x'],5):<26}{s_['viol']:>12.2e}{s_['F']:>13.4f}  {s_['lam']}")
    print("Е = рівність x1-2x2=35, Д = обмеження (1)+(2) разом. Рядки з c = 10 і ε_вн = 0.01 див. вище (рівність, два обмеження).")
    print("Точно: рівність (17.1; -8.95), F = 108.975, λ_h = -7.3;  (1)+(2): (13; -11), F = 151, λ = (5; 41);  коло: див. П.7;  (1) окремо: (2.5; 2), F = -24.25, λ = 0")

    # ---------- П.10 графіки ----------
    print("\n" + SEP); print("П.10 Графіки")
    levels = [-24, 0, 50, 100, 151, 250, 400, 700, 1000, 1500, 2500]
    def base(ax, xlim, ylim, lv):
        X, Y = np.meshgrid(np.linspace(*xlim, 600), np.linspace(*ylim, 600))
        Z = X**2 + 2*Y**2 + 2*X*Y - 9*X - 13*Y
        cs = ax.contour(X, Y, Z, levels=lv, cmap="viridis", linewidths=0.9); ax.clabel(cs, fontsize=7)
        ax.set_xlabel("x₁"); ax.set_ylabel("x₂"); ax.grid(alpha=.3)
        return X, Y
    # 1) два обмеження + штраф + бар'єр
    fig, ax = plt.subplots(figsize=(10, 7.5))
    X, Y = base(ax, (-12, 32), (-22, 8), levels)
    G1 = X - 2*Y - 35; G2 = Y + 11
    ax.contourf(X, Y, np.maximum(G1, G2), levels=[-1e9, 0], colors=["#cfeccf"], alpha=.6)
    ax.contour(X, Y, G1, levels=[0], colors="k", linewidths=1.6); ax.contour(X, Y, G2, levels=[0], colors="k", linewidths=1.6, linestyles="--")
    ax.plot(*np.array([X0] + [q['x'] for q in rowsT]).T, "o-", color="tab:red", ms=5, label="зовнішній штраф x(r), r = 1…10⁵, старт (-3; -2)")
    ax.plot(*np.array([xin] + [q['x'] for q in rowsBi]).T, "s-", color="tab:blue", ms=5, label="бар'єр x(μ), μ = 10…0.001, старт (0; -12)")
    ax.plot(*X0, "k*", ms=13, label="старт (-3; -2)"); ax.plot(*xin, "b^", ms=10, label="внутрішній старт (0; -12)")
    ax.plot(*xu, "ms", ms=8, label="безумовний мінімум (2.5; 2)"); ax.plot(*xB, "rX", ms=13, label="x* = (13; -11), λ = (5; 41)")
    ax.set_xlim(-12, 32); ax.set_ylim(-22, 8); ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Обмеження (1)+(2): Ω (зелена) — кут; штраф зовні, бар'єр зсередини")
    fig.savefig("lr11b_combined.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    # 2) рівність
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    ax = axs[0]; X, Y = base(ax, (-8, 24), (-16, 10), [-24, 0, 50, 109, 200, 400, 700, 1000, 1500])
    ax.contour(X, Y, X - 2*Y - 35, levels=[0], colors="k", linewidths=1.8)
    ax.plot(*np.array([X0] + [q['x'] for q in rowsE]).T, "o-", color="tab:red", ms=5, label="x(r), рівність")
    ax.plot(*X0, "k*", ms=13, label="старт"); ax.plot(*xu, "bs", ms=9, label="(2.5; 2): розв'язок нерівності")
    ax.plot(*xE, "rX", ms=12, label="x*_E = (17.1; -8.95)"); ax.legend(fontsize=8, loc="upper left"); ax.set_xlim(-8, 24); ax.set_ylim(-16, 10)
    ax.set_title("Рівність h = x₁ − 2x₂ − 35 = 0 (чорна пряма)")
    rg = np.logspace(0, 5, 100); ha = [abs(C1EQ.g(analytic_penalty([a1], [35], q))) for q in rg]
    axs[1].loglog(rg, ha, "-", label="|h(x(r))| аналітично"); axs[1].loglog([w['r'] for w in rowsE], [w['viol'] for w in rowsE], "ro", label="чисельно")
    axs[1].loglog(rg, abs(lamE[0])/(2*rg), "--", color="gray", label="|λ|/(2r)"); axs[1].grid(alpha=.3, which="both"); axs[1].legend()
    axs[1].set_xlabel("r"); axs[1].set_ylabel("|h(x(r))|"); axs[1].set_title("Порушення рівності ≈ |λ|/(2r)")
    fig.tight_layout(); fig.savefig("lr11b_equality.png", dpi=140); plt.close(fig)
    # 3) коло
    fig, ax = plt.subplots(figsize=(8.5, 8))
    X, Y = base(ax, (-30, 30), (-30, 30), [-24, 0, 100, 250, 500, 800, 1200, 2000, 3500])
    ax.contourf(X, Y, 441 - X**2 - Y**2, levels=[-1e9, 0], colors=["#cfeccf"], alpha=.6); ax.contour(X, Y, 441 - X**2 - Y**2, levels=[0], colors="k", linewidths=1.8)
    cols = {"старт 1: (-3; -2)": "tab:red", "старт 2: (5; -3)": "tab:blue"}
    for lab, (xc, rowsC) in circ_runs.items():
        st_ = X0 if lab.startswith("старт 1") else np.array([5., -3.])
        ax.plot(*np.array([st_] + [q['x'] for q in rowsC]).T, "o-", color=cols[lab], ms=5, label=f"x(r), {lab}")
        ax.plot(*st_, "*", color=cols[lab], ms=13)
    for i, (x, fv, lam) in enumerate(cm):
        ax.plot(*x, "X", color="k", ms=11); ax.annotate(f"F = {fv:.1f}", x, textcoords="offset points", xytext=(8, 8), fontsize=9)
    ax.plot(*xu, "ms", ms=8, label="безумовний мінімум"); ax.set_aspect("equal"); ax.set_xlim(-30, 30); ax.set_ylim(-30, 30)
    ax.set_title("Коло ‖x‖ ≥ 21 (Ω зелена): два мінімуми на колі"); ax.legend(fontsize=8, loc="upper right")
    fig.savefig("lr11b_circle.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    # 4) експерименти c та ε
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))
    cs_ = [2, 4, 10, 100]
    for pname, col in (("Е: рівність x1 - 2x2 = 35", "tab:green"), ("Д: обмеження (1)+(2) разом", "tab:purple")):
        axs[0].plot(cs_, [cres[(pname, c)][-1]['tot'] for c in cs_], "o-", color=col, label=pname[:1] + ": NΦ")
        axs[1].plot(cs_, [len(cres[(pname, c)]) for c in cs_], "o-", color=col, label=pname[:1])
    axs[0].set_xscale("log"); axs[0].set_xlabel("c"); axs[0].set_ylabel("NΦ"); axs[0].set_title("Вартість залежно від c"); axs[0].grid(alpha=.3, which="both"); axs[0].legend()
    axs[1].set_xscale("log"); axs[1].set_xlabel("c"); axs[1].set_ylabel("зовнішніх ітерацій"); axs[1].set_title("Зовнішні ітерації залежно від c"); axs[1].grid(alpha=.3, which="both"); axs[1].legend()
    es = [0.1, 0.01, 0.001]
    for pname, col, xs_ in (("Е: рівність x1 - 2x2 = 35", "tab:green", xE), ("Д: обмеження (1)+(2) разом", "tab:purple", xB)):
        axs[2].loglog(es, [np.linalg.norm(eres[(pname, e)][-1]['x'] - xs_) for e in es], "o-", color=col, label=pname[:1] + ": ‖x − x*‖")
        axs[2].loglog(es, [eres[(pname, e)][-1]['viol'] for e in es], "s--", color=col, label=pname[:1] + ": порушення")
    axs[2].set_xlabel("внутрішній ε"); axs[2].set_title("Похибка положення та порушення від ε_вн"); axs[2].grid(alpha=.3, which="both"); axs[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("lr11b_experiments.png", dpi=140); plt.close(fig)
    print("Збережено: lr11b_combined.png, lr11b_equality.png, lr11b_circle.png, lr11b_experiments.png")


if __name__ == "__main__":
    main()