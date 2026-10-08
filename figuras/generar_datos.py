"""Genera los datos numéricos de las gráficas (SIR y Lotka-Volterra).
Uso:  python3 figuras/generar_datos.py   (desde la raíz del proyecto)
Requiere numpy y scipy. Los .dat resultantes ya están incluidos en el repositorio,
de modo que Overleaf no necesita ejecutar este script."""
import numpy as np
from scipy.special import lambertw
from scipy.integrate import solve_ivp
import os

D = os.path.join(os.path.dirname(__file__), "datos")
os.makedirs(D, exist_ok=True)

def guarda(nombre, cabecera, filas):
    np.savetxt(os.path.join(D, nombre), filas, header=cabecera, comments="", fmt="%.6g")

# ---------------------------------------------------------------- SIR
N, gam = 10000.0, 0.1

def sir(t, u, beta):
    S, I, R = u
    return [-beta*S*I, beta*S*I - gam*I, gam*I]

def resuelve_sir(R0, I0=1.0, T=200.0, n=401):
    beta = R0*gam/N
    t = np.linspace(0, T, n)
    s = solve_ivp(sir, (0, T), [N-I0, I0, 0.0], t_eval=t, args=(beta,),
                  rtol=1e-9, atol=1e-9)
    return s.t, s.y

t, y = resuelve_sir(4.0)
guarda("sir_tiempo.dat", "t S I R", np.column_stack([t, y[0], y[1], y[2]]))
for R0 in (0.8, 1.5, 2.5, 4.0):
    t, y = resuelve_sir(R0)
    guarda("sir_R0_%s.dat" % str(R0).replace(".", "_"), "t I", np.column_stack([t, y[1]]))

# plano de fases (S, I): trayectoria exacta  I = N - S + (gamma/beta) ln(S/N)  (I0 ~ 0)
R0 = 4.0
S = np.linspace(N*np.exp(-R0)*0 + 190, N, 400)
S = np.linspace(150, N, 500)
Itr = N - S + (N/R0)*np.log(S/N)
guarda("sir_fase.dat", "S I", np.column_stack([S, Itr]))
# campo de direcciones
rows = []
beta = R0*gam/N
for Sg in np.arange(0, N+1, 500):
    for Ig in np.arange(0, N+1, 500):
        if Sg + Ig > N: continue
        dS, dI = -beta*Sg*Ig, beta*Sg*Ig - gam*Ig
        m = np.hypot(dS, dI)
        if m < 1e-12: continue
        rows.append([Sg, Ig, dS/m, dI/m])
guarda("sir_campo.dat", "S I u v", np.array(rows))

# ---------------------------------------------------------------- Lotka-Volterra
al = be = ga = de = 1.0     # equilibrio en (ga/de, al/be) = (1, 1)

def H(x, y):
    return de*x - ga*np.log(x) + be*y - al*np.log(y)

def orbita(c, n=160):
    # x_min, x_max: donde el discriminante de Lambert se anula
    f = lambda x: (c - de*x + ga*np.log(x)) - al*(1 + np.log(be/al))
    xs = np.linspace(1e-4, 8, 200000)
    ok = f(xs) >= 0
    x1, x2 = xs[ok][0], xs[ok][-1]
    th = np.linspace(0, np.pi, n)
    x = 0.5*(x1+x2) - 0.5*(x2-x1)*np.cos(th)
    K = c - de*x + ga*np.log(x)
    arg = -(be/al)*np.exp(-K/al)
    arg = np.maximum(arg, -1/np.e)
    ylo = -(al/be)*lambertw(arg, 0).real
    yhi = -(al/be)*lambertw(arg, -1).real
    X = np.concatenate([x, x[::-1], x[:1]])
    Y = np.concatenate([ylo, yhi[::-1], ylo[:1]])
    return X, Y

for etiqueta, c in (("a", 2.3), ("b", 2.8), ("c", 3.4)):
    X, Y = orbita(c)
    guarda("lv_orbita_%s.dat" % etiqueta, "x y", np.column_stack([X, Y]))

# series temporales para c = 2.8 (condición inicial en x = 0.5 sobre la órbita)
c = 2.8
X, Y = orbita(c)
x0 = 0.55
y0 = float(Y[np.argmin(np.abs(X[:160] - x0))])
def lv(t, u):
    x, y = u
    return [al*x - be*x*y, -ga*y + de*x*y]
t = np.linspace(0, 14, 600)
s = solve_ivp(lv, (0, 14), [x0, y0], t_eval=t, rtol=1e-10, atol=1e-12)
guarda("lv_tiempo.dat", "t x y", np.column_stack([s.t, s.y[0], s.y[1]]))
# campo de direcciones
rows = []
for xg in np.arange(0.2, 3.01, 0.4):
    for yg in np.arange(0.2, 3.01, 0.4):
        dx, dy = lv(0, [xg, yg]); m = np.hypot(dx, dy)
        rows.append([xg, yg, dx/m, dy/m])
guarda("lv_campo.dat", "x y u v", np.array(rows))
print("datos generados en", D)
