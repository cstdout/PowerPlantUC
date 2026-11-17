import math
import time
from dataclasses import dataclass
from typing import Optional
import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn

import numpy as np

try:
    from pyomo.core.base.var import VarData as cls  # Pyomo >= 6.8
except ImportError:
    from pyomo.core.base.var import _GeneralVarData as cls
cls.x = property(lambda self: self.value)


INF = float("inf")
# simplex pivot / reduced-cost tolerance
TOL = 1e-9
# "is this value integer?" tolerance
INT_TOL = 1e-6


def _pivot(tab, basis, i, j):
    tab[i] /= tab[i, j]
    f = tab[:, j].copy()
    f[i] = 0.0
    tab -= np.outer(f, tab[i])
    basis[i] = j


def _simplex(tab, basis, ncols, max_iter):
    m = tab.shape[0] - 1
    bland_after = 20 * (m + ncols)
    for it in range(max_iter):
        z = tab[-1, :ncols]
        if it < bland_after:
            j = int(np.argmin(z))
            if z[j] >= -TOL:
                return "optimal"
        else:
            neg = np.flatnonzero(z < -TOL)
            if neg.size == 0:
                return "optimal"
            j = int(neg[0])
        col = tab[:m, j]
        pos = col > TOL
        if not pos.any():
            return "unbounded"
        ratios = np.full(m, INF)
        ratios[pos] = np.maximum(tab[:m, -1][pos], 0.0) / col[pos]
        cand = np.flatnonzero(ratios <= ratios.min() + 1e-12)
        i = int(cand[np.argmin(np.asarray(basis)[cand])])
        _pivot(tab, basis, i, j)
    raise RuntimeError("simplex iteration limit reached")


def solve_lp(c, A_ub, b_ub, A_eq, b_eq, lb, ub):
    n = len(c)
    if np.any(lb > ub + 1e-9):
        return "infeasible", None, None

    # x = offset + T y with y >= 0
    cols, extra = [], []
    offset = np.zeros(n)
    for j in range(n):
        if np.isfinite(lb[j]):
            offset[j] = lb[j]
            cols.append((j, 1.0))
            if np.isfinite(ub[j]):
                extra.append((len(cols) - 1, ub[j] - lb[j]))
        elif np.isfinite(ub[j]):
            offset[j] = ub[j]
            cols.append((j, -1.0))
        else:
            # free variable
            cols.append((j, 1.0))
            cols.append((j, -1.0))
    N = len(cols)
    T = np.zeros((n, N))
    for k, (j, s) in enumerate(cols):
        T[j, k] = s

    G = np.vstack([A_ub @ T, np.zeros((len(extra), N))])
    h = np.concatenate([b_ub - A_ub @ offset, np.zeros(len(extra))])
    for r, (k, rhs) in enumerate(extra):
        G[len(b_ub) + r, k] = 1.0
        h[len(b_ub) + r] = rhs
    E = A_eq @ T
    f = b_eq - A_eq @ offset
    mi, me = len(h), len(f)
    m = mi + me

    # standard form with slacks / artificials, rhs >= 0
    flip = np.concatenate([h < 0, f < 0])
    art_rows = [i for i in range(m) if flip[i] or i >= mi]   # eq rows and flipped <= rows
    na = len(art_rows)
    ncol = N + mi + na
    tab = np.zeros((m + 1, ncol + 1))
    tab[:mi, :N] = G
    tab[mi:m, :N] = E
    tab[:mi, N:N + mi] = np.eye(mi)
    tab[:m, -1] = np.concatenate([h, f])
    for i in range(m):
        if flip[i]:
            tab[i] *= -1.0
    basis = [N + i for i in range(mi)]
    basis += [-1] * me
    for a, i in enumerate(art_rows):
        tab[i, N + mi + a] = 1.0
        basis[i] = N + mi + a

    max_iter = 100 * (m + ncol) + 1000

    # minimise sum of artificials
    if na:
        tab[-1, N + mi:ncol] = 1.0
        for i in art_rows:
            tab[-1] -= tab[i]
        _simplex(tab, basis, ncol, max_iter)
        if -tab[-1, -1] > 1e-7:
            return "infeasible", None, None
        # drive artificials out of basis
        for i, b in enumerate(basis):
            if b >= N + mi:
                nz = np.flatnonzero(np.abs(tab[i, :N + mi]) > 1e-9)
                if nz.size:
                    _pivot(tab, basis, i, int(nz[0]))

    # real objective
    cost = np.zeros(ncol + 1)
    cost[:N] = c @ T
    tab[-1] = cost - cost[basis] @ tab[:m]
    if _simplex(tab, basis, N + mi, max_iter) == "unbounded":
        return "unbounded", None, None

    y = np.zeros(ncol)
    for i, b in enumerate(basis):
        y[b] = tab[i, -1]
    x = offset + T @ y[:N]
    return "optimal", x, float(c @ x)


# Branch Bound
@dataclass
class Result:
    # optimal | feasible | infeasible | unbounded | no_solution
    status: str

    # best solution found
    x: Optional[np.ndarray]

    # objective of the best solution (nan if none)
    incumbent: float

    # best proven bound on the objective
    best_bd: float

    # |incumbent - best_bd| / |incumbent|
    gap: float
    nodes: int
    time: float

    def __str__(self):
        return (f"status={self.status}  Incumbent={self.incumbent:.6g}  "
                f"BestBd={self.best_bd:.6g}  Gap={self.gap:.4%}  "
                f"nodes={self.nodes}  time={self.time:.2f}s")


def _gap(inc, bd):
    if not math.isfinite(inc):
        return INF
    return abs(inc - bd) / max(abs(inc), 1e-10)


def solve_milp(c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, lb=None, ub=None,
               integer=None, maximize=False, const=0.0,
               time_limit=None, mip_gap=1e-6, node_limit=None, verbose=False):
    t0 = time.time()
    c = np.asarray(c, float)
    n = len(c)

    # set defaults
    A_ub = np.zeros((0, n)) if A_ub is None else np.asarray(A_ub, float).reshape(-1, n)
    b_ub = np.zeros(0) if b_ub is None else np.asarray(b_ub, float)
    A_eq = np.zeros((0, n)) if A_eq is None else np.asarray(A_eq, float).reshape(-1, n)
    b_eq = np.zeros(0) if b_eq is None else np.asarray(b_eq, float)
    lb = np.zeros(n) if lb is None else np.asarray(lb, float).copy()
    ub = np.full(n, INF) if ub is None else np.asarray(ub, float).copy()
    integer = np.zeros(n, bool) if integer is None else np.asarray(integer, bool)

    # integer variables: tighten bounds to integers
    lb[integer & np.isfinite(lb)] = np.ceil(lb[integer & np.isfinite(lb)] - INT_TOL)
    ub[integer & np.isfinite(ub)] = np.floor(ub[integer & np.isfinite(ub)] + INT_TOL)

    sign = -1.0 if maximize else 1.0
    # internally always minimise
    cm = sign * c

    # internal -> user objective
    out = lambda v: sign * v + const

    # if the objective is integer-valued, bounds can be rounded up
    int_obj = bool(np.all(np.abs(cm - np.round(cm))[cm != 0] < 1e-12)
                   and np.all(integer[cm != 0]))

    def lp(lo, hi):
        st, x, obj = solve_lp(cm, A_ub, b_ub, A_eq, b_eq, lo, hi)
        if st == "optimal" and int_obj:
            obj = math.ceil(obj - 1e-6)
        return st, x, obj

    st, x, obj = lp(lb, ub)
    if st != "optimal":
        return Result(st, None, math.nan, math.nan, math.nan, 1, time.time() - t0)

    open_nodes = [(obj, x, lb, ub)]
    inc, best_x, nodes = INF, None, 1
    status_bd = obj
    while open_nodes:
        bd = min(nd[0] for nd in open_nodes)
        status_bd = bd
        if best_x is not None and _gap(inc, bd) <= mip_gap:
            break
        if (time_limit and time.time() - t0 > time_limit) or \
           (node_limit and nodes >= node_limit):
            break

        # depth-first until we have an incumbent, then best-bound-first
        k = len(open_nodes) - 1 if best_x is None else \
            next(i for i, nd in enumerate(open_nodes) if nd[0] == bd)
        obj, x, lo, hi = open_nodes.pop(k)
        if obj >= inc - 1e-9:
            continue
        frac = np.abs(x - np.round(x))
        frac[~integer] = 0.0
        j = int(np.argmax(frac))
        # integer feasible -> new incumbent
        if frac[j] <= INT_TOL:
            inc, best_x = obj, x.copy()
            if verbose:
                print(f"  node {nodes}: incumbent {out(inc):.6g}")
            open_nodes = [nd for nd in open_nodes if nd[0] < inc - 1e-9]
            continue
        down_first = x[j] - math.floor(x[j]) < 0.5
        kids = []
        # 0: x<=floor, 1: x>=ceil
        for side in (0, 1):
            lo2, hi2 = lo.copy(), hi.copy()
            if side == 0:
                hi2[j] = math.floor(x[j])
            else:
                lo2[j] = math.ceil(x[j])
            nodes += 1
            s2, x2, o2 = lp(lo2, hi2)
            if s2 == "optimal" and o2 < inc - 1e-9:
                kids.append((side, (o2, x2, lo2, hi2)))

        # push the preferred child last so DFS pops it first
        kids.sort(key=lambda t: (t[0] == 0) == down_first)
        open_nodes.extend(nd for _, nd in kids)

    if open_nodes:
        bd = min(nd[0] for nd in open_nodes)
    else:
        bd = inc if best_x is not None else INF
    if best_x is None:
        status = "infeasible" if not open_nodes else "no_solution"
        return Result(status, None, math.nan, out(status_bd) if open_nodes else math.nan,
                      INF, nodes, time.time() - t0)
    bd = min(bd, inc)
    best_x = np.where(integer, np.round(best_x), best_x)
    g = _gap(inc, bd)
    return Result("optimal" if g <= mip_gap else "feasible", best_x,
                  out(inc), out(bd), g, nodes, time.time() - t0)


# Pyomo adapter
def solve_pyomo(model, load_solution=True, **options):
    vars_ = [v for v in model.component_data_objects(pyo.Var, descend_into=True)
             if not v.fixed]
    pos = {id(v): i for i, v in enumerate(vars_)}
    n = len(vars_)
    lb = np.array([-INF if v.lb is None else float(v.lb) for v in vars_])
    ub = np.array([INF if v.ub is None else float(v.ub) for v in vars_])
    integer = np.array([v.is_integer() or v.is_binary() for v in vars_], bool)

    def row(expr):
        r = generate_standard_repn(expr, compute_values=True)
        if r.nonlinear_expr is not None or len(r.quadratic_vars or ()) > 0:
            raise ValueError("solve_pyomo supports only linear expressions: "
                             f"{expr}")
        a = np.zeros(n)
        for coef, v in zip(r.linear_coefs, r.linear_vars):
            a[pos[id(v)]] += float(coef)
        return a, float(r.constant)

    objs = list(model.component_data_objects(pyo.Objective, active=True))
    if len(objs) != 1:
        raise ValueError(f"expected exactly one active Objective, found {len(objs)}")
    c, const = row(objs[0].expr)
    maximize = objs[0].sense == pyo.maximize

    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for con in model.component_data_objects(pyo.Constraint, active=True, descend_into=True):
        a, k = row(con.body)
        lo = None if con.lower is None else float(pyo.value(con.lower)) - k
        up = None if con.upper is None else float(pyo.value(con.upper)) - k
        if lo is not None and up is not None and abs(lo - up) < 1e-12:
            A_eq.append(a); b_eq.append(up)
            continue
        if up is not None:
            A_ub.append(a); b_ub.append(up)
        if lo is not None:
            A_ub.append(-a); b_ub.append(-lo)

    res = solve_milp(c, A_ub or None, b_ub or None, A_eq or None, b_eq or None,
                     lb, ub, integer, maximize, const, **options)
    if load_solution and res.x is not None:
        for v, val in zip(vars_, res.x):
            v.set_value(float(val))
    return res
