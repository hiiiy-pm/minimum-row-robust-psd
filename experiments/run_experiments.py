from __future__ import annotations

import csv
import importlib.metadata
import json
import platform
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent

import numpy as np
from scipy import optimize
import sympy as sp

DATA = HERE.parent / "data"
DATA.mkdir(parents=True, exist_ok=True)
SEED = 20261030
RNG = np.random.default_rng(SEED)
FEAS_TOL = 2e-7


def emit(name, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows for {name}")
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (DATA / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def jstd(m):
    assert m % 2 == 0
    return np.kron(np.eye(m // 2), np.array([[0., -1.], [1., 0.]]))


def sym(M):
    return (M + M.T) / 2


def adj2(K):
    return np.array([[K[1, 1], -K[0, 1]], [-K[1, 0], K[0, 0]]])


def eps_family(e):
    B = [np.diag([1., -1.]), np.array([[e, 1.], [1., 1.]]),
         np.array([[e, -1.], [-1., 1.]])]
    K = [adj2(b) / (-np.linalg.det(b)) for b in B]
    r, t = 2 / (1 + e), (1 - e) / (1 + e)
    return B, K, r, t, 1 / e, r + t


def optimal_factor(e, m, J=None):
    _, _, r, t, alpha, _ = eps_family(e)
    ne = m - m % 2
    W1 = np.sqrt(r) * np.eye(m)
    W2 = np.zeros((m, m))
    if ne:
        W2[:ne, :ne] = -np.sqrt(t) * (jstd(ne) if J is None else J)
    if m % 2:
        W1[-1, -1] = np.sqrt(alpha)
    return np.vstack([W1, W2])


def qmi_eigs(W, B):
    m = W.shape[1]
    return np.concatenate([np.linalg.eigvalsh(sym(W.T @ np.kron(b, np.eye(m)) @ W)) - 1
                           for b in B])


def original_min_eig(H, K, m):
    return min(float(np.linalg.eigvalsh(sym(np.kron(k, np.eye(m)) + H))[0]) for k in K)


def finite_scalar(B, d=None):


    d = np.ones(len(B)) if d is None else np.asarray(d)
    candidates = [0.]
    p = [np.array([b[0, 0], 2 * b[0, 1], b[1, 1]]) for b in B]

    def real_roots(coeff):
        coeff = np.trim_zeros(np.asarray(coeff, dtype=float), "f")
        if len(coeff) <= 1:
            return []
        return [float(z.real) for z in np.roots(coeff) if abs(z.imag) < 1e-9]

    for b in B:
        a, h, c = b[0, 0], b[0, 1], b[1, 1]
        candidates += real_roots([h, a - c, -h])
    for i in range(len(B)):
        for j in range(i):
            candidates += real_roots((d[i] * p[j] - d[j] * p[i])[::-1])
    evaluated = []
    for z in candidates:
        denom = np.array([np.polynomial.polynomial.polyval(z, pp) for pp in p])
        if np.all(denom > 1e-13):
            evaluated.append((float(np.max(d * (1 + z * z) / denom)), float(z)))
    if all(b[1, 1] > 0 for b in B):
        evaluated.append((float(max(di / b[1, 1] for b, di in zip(B, d))), float("inf")))
    if not evaluated:
        raise ValueError("No feasible finite rank-one candidate")
    value, z = min(evaluated)
    v = np.array([0., np.sqrt(value)]) if np.isinf(z) else np.sqrt(value / (1 + z * z)) * np.array([1., z])
    return value, z, len(candidates), min(float(v @ b @ v - di) for b, di in zip(B, d))


def closed_form_experiment():
    rows, spectra = [], []
    for e in [.001, .005, .01, .03, .1, .2, .5, .8, .95, .99]:
        B, K, _, _, alpha, kappa = eps_family(e)
        for m in range(1, 17):
            W = optimal_factor(e, m)
            H, G = W @ W.T, W.T @ W
            eig = np.linalg.eigvalsh(G)
            delta = m % 2
            target = np.array([kappa] * (m - delta) + [alpha] * delta)
            rows.append(dict(epsilon=e, m=m, delta=delta, alpha=alpha, kappa=kappa,
                             target_trace=float(target.sum()), actual_trace=float(np.trace(H)),
                             target_op=float(target[-1]), actual_op=float(eig[-1]),
                             target_square_trace=float(target @ target), actual_square_trace=float(np.sum(H * H)),
                             min_original_eig=original_min_eig(H, K, m),
                             min_qmi_eig=float(qmi_eigs(W, B).min()),
                             rank=int(np.linalg.matrix_rank(H, tol=1e-8)),
                             relative_saving_vs_scalar=1 - float(target.sum()) / (m * alpha)))
            spectra.extend(dict(epsilon=e, m=m, index=i + 1, eigenvalue=float(x), target=float(y))
                           for i, (x, y) in enumerate(zip(eig, target)))
    emit("epsilon_closed_form.csv", rows)
    emit("epsilon_spectra.csv", spectra)
    return dict(rows=len(rows), max_trace_error=max(abs(x["actual_trace"] - x["target_trace"]) for x in rows),
                max_op_error=max(abs(x["actual_op"] - x["target_op"]) for x in rows),
                max_square_trace_relative_error=max(abs(x["actual_square_trace"] - x["target_square_trace"]) / x["target_square_trace"] for x in rows),
                min_original_eig=min(x["min_original_eig"] for x in rows),
                min_qmi_eig=min(x["min_qmi_eig"] for x in rows))


def sdp_scalar_experiment():
    import cvxpy as cp
    rows = []
    for e in [.001, .005, .01, .03, .1, .2, .5, .8, .95, .99]:
        B, _, r, t, alpha, kappa = eps_family(e)
        X = cp.Variable((2, 2), symmetric=True)
        problem = cp.Problem(cp.Minimize(cp.trace(X)), [X >> 0] + [cp.sum(cp.multiply(b, X)) >= 1 for b in B])
        begin = time.perf_counter()
        with warnings.catch_warnings(record=True) as ws:
            problem.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_feas=1e-10, tol_gap_rel=1e-10, max_iter=300)
        elapsed = time.perf_counter() - begin
        av, z, nc, scalar_slack = finite_scalar(B)
        value = X.value
        rows.append(dict(epsilon=e, solver="CVXPY/CLARABEL", status=problem.status,
                         kappa_formula=kappa, kappa_numeric=float(problem.value),
                         kappa_error=float(problem.value - kappa),
                         matrix_frobenius_error=float(np.linalg.norm(value - np.diag([r, t]))),
                         sdp_min_slack=min(float(np.sum(b * value) - 1) for b in B),
                         sdp_min_psd_eig=float(np.linalg.eigvalsh(value)[0]),
                         alpha_formula=alpha, alpha_numeric=av, alpha_error=av - alpha,
                         scalar_direction_z=z, finite_candidate_count=nc, scalar_min_slack=scalar_slack,
                         seconds=elapsed, solver_iterations=problem.solver_stats.num_iters,
                         warnings=" | ".join(str(w.message) for w in ws)))
    emit("epsilon_sdp_scalar.csv", rows)
    return dict(rows=len(rows), statuses=sorted(set(x["status"] for x in rows)),
                max_abs_kappa_error=max(abs(x["kappa_error"]) for x in rows),
                max_abs_alpha_error=max(abs(x["alpha_error"]) for x in rows),
                max_matrix_error=max(x["matrix_frobenius_error"] for x in rows))


def scale_feasible(W, B, buffer=2e-12):

    minimum = float(qmi_eigs(W, B).min()) + 1
    if minimum <= 0:
        return None
    return W * max(1., np.sqrt((1 + buffer) / minimum))


def nonconvex_experiment():
    rows, factors = [], {}
    for e in [.03, .2, .7]:
        B, K, _, _, alpha, kappa = eps_family(e)
        for m in [2, 3, 4]:
            n = 2 * m * m
            matrices = [np.kron(b, np.eye(m)) for b in B]
            target = (m - m % 2) * kappa + (m % 2) * alpha

            def constraint(x):
                return qmi_eigs(x.reshape(2 * m, m), B)

            def cjac(x):
                W = x.reshape(2 * m, m)
                out = []
                for A in matrices:
                    _, V = np.linalg.eigh(sym(W.T @ A @ W))
                    AW = A @ W
                    out.extend((2 * np.outer(AW @ V[:, i], V[:, i])).ravel() for i in range(m))
                return np.array(out)

            starts = []
            paired = optimal_factor(e, m)
            starts.append(("paired_informed_perturbation", scale_feasible(paired + .03 * RNG.normal(size=paired.shape), B)))
            scalar = np.vstack([np.sqrt(alpha) * np.eye(m), np.zeros((m, m))])
            starts.append(("scalar_axis_perturbation", scale_feasible(1.1 * scalar + .01 * RNG.normal(size=scalar.shape), B)))
            for _ in range(4):
                for attempt in range(100):
                    trial = np.vstack([np.sqrt(alpha) * (np.eye(m) + .25 * RNG.normal(size=(m, m))),
                                       .1 * np.sqrt(e) * RNG.normal(size=(m, m))])
                    feasible = scale_feasible(trial, B)
                    if feasible is not None:
                        break
                starts.append(("random_full_feasible", feasible))
            for _ in range(2):
                starts.append(("random_infeasible", RNG.normal(size=(2 * m, m))))
            for sid, (kind, W0) in enumerate(starts):
                if W0 is None:
                    rows.append(dict(epsilon=e, m=m, start_id=sid, kind=kind, success=False,
                                     feasible=False, status=-999, message="Unable to construct start"))
                    continue
                begin = time.perf_counter()
                result = optimize.minimize(lambda x: float(x @ x), W0.ravel(), jac=lambda x: 2 * x,
                                           method="SLSQP", constraints=[dict(type="ineq", fun=constraint, jac=cjac)],
                                           options=dict(maxiter=450, ftol=2e-10, disp=False))
                elapsed = time.perf_counter() - begin
                W = result.x.reshape(2 * m, m)
                mineig = float(constraint(result.x).min())
                repaired = scale_feasible(W, B)
                repaired_objective = None if repaired is None else float(np.sum(repaired * repaired))
                factor_key = f"e{e:.3f}_m{m}_start{sid}".replace(".", "p")
                factors[factor_key + "_initial"] = W0
                factors[factor_key + "_final"] = W
                rows.append(dict(epsilon=e, m=m, start_id=sid, kind=kind,
                                 initial_min_qmi_eig=float(constraint(W0.ravel()).min()),
                                 success=bool(result.success), feasible=mineig >= -FEAS_TOL,
                                 status=int(result.status), message=str(result.message),
                                 objective=float(result.fun), target=target, gap=float(result.fun - target),
                                 relative_gap=float((result.fun - target) / target),
                                 min_qmi_eig=mineig, min_original_eig=original_min_eig(W @ W.T, K, m),
                                 scaled_repair_objective=repaired_objective,
                                 scaled_repair_relative_gap=None if repaired is None else (repaired_objective - target) / target,
                                 scaled_repair_min_qmi_eig=None if repaired is None else float(qmi_eigs(repaired, B).min()),
                                 factor_archive_key=factor_key,
                                 seconds=elapsed, iterations=int(result.nit), function_evaluations=int(result.nfev),
                                 n_variables=n))
                print(f"nonconvex epsilon={e} m={m} start={sid} success={result.success} gap={result.fun-target:.3g}", flush=True)
    emit("nonconvex_runs.csv", rows)
    np.savez_compressed(DATA / "nonconvex_factors.npz", **factors)
    grouped = []
    for e in [.03, .2, .7]:
        for m in [2, 3, 4]:
            group = [x for x in rows if x["epsilon"] == e and x["m"] == m]
            feasible = [x for x in group if x.get("feasible") and "objective" in x]
            repaired = [x for x in group if x.get("scaled_repair_objective") is not None]
            grouped.append(dict(epsilon=e, m=m, runs=len(group), success_count=sum(x["success"] for x in group),
                                feasible_count=len(feasible),
                                best_feasible_objective=min((x["objective"] for x in feasible), default=None),
                                target=group[0].get("target"),
                                best_feasible_relative_gap=min((x["relative_gap"] for x in feasible), default=None),
                                best_scaled_repair_objective=min((x["scaled_repair_objective"] for x in repaired), default=None),
                                best_scaled_repair_relative_gap=min((x["scaled_repair_relative_gap"] for x in repaired), default=None),
                                scaled_repair_count=len(repaired),
                                total_seconds=sum(x.get("seconds", 0) for x in group)))
    emit("nonconvex_summary.csv", grouped)
    return dict(runs=len(rows), groups=grouped,
                solver_success_count=sum(x["success"] for x in rows),
                feasible_at_tolerance_count=sum(x["feasible"] for x in rows),
                caution="SLSQP is local, eigenvalue constraints are nonsmooth at multiplicities; all failures retained. Feasibility tolerance 2e-7. Slightly negative uncorrected gaps can arise from tolerance. Scaled-repair columns restore QMI feasibility with a 2e-12 positive buffer when possible. No numerical run proves global optimality.")


def nuclear(M):
    return float(np.linalg.svd(M, compute_uv=False).sum())


def recover_H(W, r, t):
    m = W.shape[1]
    W1, W2 = W[:m], W[m:]
    A = sym(W1.T @ W1)
    vals, V = np.linalg.eigh(A)
    U = W1 @ ((V * vals ** -.5) @ V.T)
    Z = W1.T @ W2
    S = (Z - Z.T) / 2
    ss, Q = np.linalg.eigh(sym(-S @ S))
    if ss.min() < 1e-16:
        raise ValueError("Recovery path approached a singular skew factor")
    J = S @ ((Q * ss ** -.5) @ Q.T)
    Wstar = np.vstack([np.sqrt(r) * U, U @ (np.sqrt(r * t) * J) / np.sqrt(r)])
    return Wstar @ Wstar.T, max(float(np.linalg.norm(J + J.T)), float(np.linalg.norm(J @ J + np.eye(m))))


def stability_experiment():
    rows = []
    for e in [.2, .5, .8]:
        B, _, r, t, _, kappa = eps_family(e)
        Wstar = optimal_factor(e, 2)
        candidates = [optimal_factor(e, 2, J=sign * jstd(2)) for sign in [-1, 1]]
        Hc = [W @ W.T for W in candidates]
        D = RNG.normal(size=Wstar.shape)
        D /= np.linalg.norm(D)
        for path in ["radial_reference", "full_random_perturbation", "skew_amplitude"]:
            for tau in np.logspace(-6, -1, 16):
                if path == "radial_reference":
                    W = (1 + tau) * Wstar
                elif path == "full_random_perturbation":
                    W = scale_feasible(Wstar + tau * D, B)
                else:
                    b = np.sqrt(t) * (1 - tau)
                    a = np.sqrt((1 - b * b) / e)
                    W = np.vstack([a * np.eye(2), -b * jstd(2)])
                H = W @ W.T
                gap = float(np.trace(H) - 2 * kappa)
                distance = min(nuclear(H - hc) for hc in Hc)
                rows.append(dict(family="epsilon", epsilon=e, m=2, path=path, tau=tau,
                                 gap=gap, distance=distance, distance_kind="exact_minimum_over_J_plus_minus",
                                 min_qmi_eig=float(qmi_eigs(W, B).min()), ratio=distance / gap))
    for m in [4, 6, 8]:
        e = .2
        B, _, r, t, _, kappa = eps_family(e)
        O, _ = np.linalg.qr(RNG.normal(size=(m, m)))
        J = O @ jstd(m) @ O.T
        Wstar = optimal_factor(e, m, J=J)
        D = RNG.normal(size=Wstar.shape)
        D /= np.linalg.norm(D)
        for tau in np.logspace(-6, -1, 16):
            W = scale_feasible(Wstar + tau * D, B)
            H = W @ W.T
            recovered, residual = recover_H(W, r, t)
            gap = float(np.trace(H) - m * kappa)
            distance = nuclear(H - recovered)
            rows.append(dict(family="epsilon", epsilon=e, m=m, path="random_structure_full_perturbation", tau=tau,
                             gap=gap, distance=distance, distance_kind="explicit_recovery_upper_bound",
                             recovery_structure_residual=residual,
                             min_qmi_eig=float(qmi_eigs(W, B).min()), ratio=distance / gap))
    for e in [.9, .99, .999, .9999]:
        B, _, r, t, _, kappa = eps_family(e)
        Wstar = optimal_factor(e, 2)
        Wopposite = optimal_factor(e, 2, J=-jstd(2))
        Hstar = Wstar @ Wstar.T
        Hopposite = Wopposite @ Wopposite.T
        for tau in np.logspace(-3, -1, 9):
            b = np.sqrt(t) * (1 - tau)
            a = np.sqrt((1 - b * b) / e)
            W = np.vstack([a * np.eye(2), -b * jstd(2)])
            H = W @ W.T

            gap = 2 * (1 / e - 1) * (t - b * b)
            distance = min(nuclear(H - Hstar), nuclear(H - Hopposite))
            rows.append(dict(family="epsilon_near_certificate_degeneracy", epsilon=e, m=2,
                             path="skew_amplitude", tau=tau, gap=gap, numeric_gap=float(np.trace(H) - 2 * kappa),
                             distance=distance, distance_kind="exact_minimum_over_J_plus_minus",
                             min_qmi_eig=float(qmi_eigs(W, B).min()), ratio=distance / gap))
    Hstar = np.diag([1., 1., 0., 0.])
    for tau in np.logspace(-6, -1, 21):
        W = np.vstack([np.diag([np.sqrt(1 + tau * tau), 1.]), np.diag([tau, 0.])])
        distance = nuclear(W @ W.T - Hstar)
        gap = 2 * tau * tau
        rows.append(dict(family="rank_one_degenerate", epsilon="", m=2, path="tilted_scalar_axis", tau=tau,
                         gap=gap, numeric_gap=float(np.sum(W * W) - 2), distance=distance,
                         distance_formula=2 * tau * np.sqrt(1 + tau * tau),
                         distance_kind="exact_distance_to_unique_optimum",
                         min_qmi_eig=float(qmi_eigs(W, [np.diag([1., -1.])]).min()), ratio=distance / gap))
    emit("stability_paths.csv", rows)
    fits = []
    groups = sorted(set((x["family"], str(x["epsilon"]), x["m"], x["path"]) for x in rows))
    for family, e, m, path in groups:
        group = [x for x in rows if x["family"] == family and str(x["epsilon"]) == e and x["m"] == m and x["path"] == path]
        fitrows = [x for x in group if x["tau"] <= .01 and x["gap"] > 1e-13]
        lx, ly = np.log([x["gap"] for x in fitrows]), np.log([x["distance"] for x in fitrows])
        slope, intercept = np.polyfit(lx, ly, 1)
        predicted = slope * lx + intercept
        fits.append(dict(family=family, epsilon=e, m=m, path=path, n_fit=len(fitrows), max_fit_tau=.01,
                         distance_kind=group[0]["distance_kind"], log_distance_vs_log_gap_slope=float(slope),
                         log_fit_r2=float(1 - np.sum((ly - predicted) ** 2) / np.sum((ly - ly.mean()) ** 2)),
                         min_ratio=min(x["ratio"] for x in group), max_ratio=max(x["ratio"] for x in group),
                         min_qmi_eig=min(x["min_qmi_eig"] for x in group)))
    emit("stability_fits.csv", fits)
    return dict(rows=len(rows), fits=fits, caution="For m>=4 distance is an explicit recovered-candidate upper bound, not exact distance to the optimum set. Slopes describe selected deterministic feasible paths, not a proof or uniform lower bound.")


def k4_experiment():
    root3 = np.sqrt(3.)
    P = np.array([[3., root3], [root3, 1.]]) / 8
    Q = np.array([[3., -root3], [-root3, 1.]]) / 8
    C0 = np.diag([7 / 16, 13 / 16])
    L, G, S = sp.Rational(49, 256), sp.Rational(327, 512), sp.Rational(169, 256)
    g, u = sp.symbols("g u", real=True)
    Ps = sp.Matrix([[3, sp.sqrt(3)], [sp.sqrt(3), 1]]) / 8
    Qs = sp.Matrix([[3, -sp.sqrt(3)], [-sp.sqrt(3), 1]]) / 8
    C0s = sp.diag(sp.Rational(7, 16), sp.Rational(13, 16))
    alpha_high = (1024 * g**2 - 4304 * g + 1849) / (4 * (256 * g - 433))
    r_high = -(256*g-49)*(2048*g-293)/(2560*(256*g-265))
    t_high = 9*(256*g-169)*(512*g-327)/(2560*(256*g-265))
    polynomials = {}
    for branch in ["low", "high"]:
        al = g - L if branch == "low" else alpha_high
        X = sp.diag(g - L, 0) if branch == "low" else sp.diag(r_high, t_high)
        for label, Xc in [("scalar", sp.diag(al, 0)), ("moment", X)]:
            for edge, U in [("P", Ps), ("Q", Qs)]:
                K = (C0s + u * U)**2 - g * sp.eye(2)
                polynomials[(branch, label, edge)] = sp.Poly(sp.factor(K.det() + sp.trace(K.adjugate() * Xc)), u)
    direction_rows = []
    for i in [0, 1]:
        for j in [0, 1]:
            C = C0s + sp.Rational(11, 4) * (i * Ps + j * Qs)
            D = (Ps + Qs) * C + C * (Ps + Qs)
            direction_rows.append(dict(corner=f"{i}{j}", first_principal_minor=str(sp.factor(D[0, 0])),
                                       determinant=str(sp.factor(D.det())),
                                       positive=bool(D[0, 0] > 0 and D.det() > 0)))
    emit("k4_common_direction_certificate.csv", direction_rows)
    gamma_values = sorted(set([L + (S - L) * sp.Rational(i, 80) for i in range(1, 80)] + [G, sp.Rational(13, 20)]))
    rows, certs = [], []
    grid = np.linspace(0., 11 / 4, 31)
    Cgrid = np.array([C0 + x * P + y * Q for x in grid for y in grid])
    for gs in gamma_values:
        gamma = float(gs)
        branch = "low" if gs <= G else "high"
        al = gs - L if branch == "low" else sp.factor(alpha_high.subs(g, gs))
        rs = gs - L if branch == "low" else sp.factor(r_high.subs(g, gs))
        ts = sp.Integer(0) if branch == "low" else sp.factor(t_high.subs(g, gs))
        alpha, r, t = float(al), float(rs), float(ts)
        kappa = r + t
        X = np.diag([r, t])
        scalar = np.diag([alpha, 0.])
        Hpair = np.block([[r * np.eye(2), np.sqrt(max(0., r*t)) * jstd(2)],
                          [-np.sqrt(max(0., r*t)) * jstd(2), t * np.eye(2)]])
        moment_min, scalar_min, paired_min = float("inf"), float("inf"), float("inf")
        for C in Cgrid:
            K = C @ C - gamma * np.eye(2)
            ell = np.linalg.det(K) + np.sum(adj2(K) * X)
            moment_min = min(moment_min, float(ell))
            scalar_min = min(scalar_min, float(np.linalg.eigvalsh(K + scalar)[0]))
            paired_min = min(paired_min, float(np.linalg.eigvalsh(np.kron(K, np.eye(2)) + Hpair)[0]))
        all_exact_pass = True
        poly_minimum = float("inf")
        for label in ["scalar", "moment"]:
            for edge in ["P", "Q"]:
                poly = polynomials[(branch, label, edge)]
                cc = [sp.factor(poly.nth(k).subs(g, gs)) for k in range(3)]
                candidates = [(sp.Integer(0), cc[0]), (sp.Rational(11,4), sum(cc[k]*sp.Rational(11,4)**k for k in range(3)))]
                if cc[2] > 0:
                    vertex = -cc[1] / (2 * cc[2])
                    if 0 < vertex < sp.Rational(11,4):
                        candidates.append((vertex, sum(cc[k]*vertex**k for k in range(3))))
                minpoint, minval = min(candidates, key=lambda item: item[1])
                passed = bool(minval >= 0)
                all_exact_pass &= passed
                poly_minimum = min(poly_minimum, float(minval))
                certs.append(dict(gamma=gamma, gamma_rational=str(gs), branch=branch, candidate=label, edge=edge,
                                  c0=str(cc[0]), c1=str(cc[1]), c2=str(cc[2]),
                                  minimizer_rational=str(minpoint), min_value_rational=str(sp.factor(minval)),
                                  min_value=float(minval), exact_nonnegative=passed))
        corners = [C0, C0 + (11/4)*P, C0 + (11/4)*Q]
        Kcorners = [C@C - gamma*np.eye(2) for C in corners]
        Bs = [adj2(K) for K in Kcorners if np.linalg.det(K) < -1e-12]
        ds = [-np.linalg.det(K) for K in Kcorners if np.linalg.det(K) < -1e-12]
        scalar_numeric, direction, _, scalar_slack = finite_scalar(Bs, ds)
        rows.append(dict(gamma=gamma, gamma_rational=str(gs), branch=branch, alpha=alpha, kappa=kappa,
                         r=r, t=t, absolute_gap=alpha-kappa, relative_saving=1-kappa/alpha,
                         scalar_numeric_corners=scalar_numeric, scalar_corner_error=scalar_numeric-alpha,
                         scalar_direction_z=direction, scalar_corner_min_slack=scalar_slack,
                         sample_grid_n=31, sample_points=31**2,
                         sample_min_moment_slack=moment_min, sample_min_scalar_psd_eig=scalar_min,
                         sample_min_pair_psd_eig=paired_min, edge_polynomial_min=poly_minimum,
                         polynomial_exact_pass=all_exact_pass,
                         common_direction_exact_pass=all(x["positive"] for x in direction_rows)))
    emit("k4_scan.csv", rows)
    emit("k4_polynomial_certificate.csv", certs)
    witness = min(rows, key=lambda x: abs(x["gamma"] - .65))
    return dict(rows=len(rows), certificate_rows=len(certs),
                all_rational_polynomial_certificates_pass=all(x["exact_nonnegative"] for x in certs),
                common_direction_certificate_pass=all(x["positive"] for x in direction_rows),
                max_scalar_corner_abs_error=max(abs(x["scalar_corner_error"]) for x in rows),
                min_sample_moment_slack=min(x["sample_min_moment_slack"] for x in rows),
                min_sample_scalar_eig=min(x["sample_min_scalar_psd_eig"] for x in rows),
                min_sample_pair_eig=min(x["sample_min_pair_psd_eig"] for x in rows),
                max_relative_saving=max(x["relative_saving"] for x in rows), witness_gamma_13_20=witness,
                caution="31x31 sampling is only diagnostic. Continuous feasibility uses exact rational edge-polynomial minima and an exact positive common-direction certificate at all four corners.")


def environment():
    packages = {}
    for package in ["numpy", "scipy", "sympy", "cvxpy", "clarabel", "osqp", "scs", "highspy", "qdldl", "sparsediffpy"]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not installed"
    return dict(seed=SEED, python=sys.version, platform=platform.platform(),
                packages=packages,
                thread_environment={k: __import__("os").environ.get(k) for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]},
                deterministic_inputs=True, timing_is_machine_dependent=True)


def main():
    begin = time.perf_counter()
    summary = dict(seed=SEED, data_type="deterministic mathematical test families; no empirical sensor measurements",
                   distance_scope="Nuclear-norm distances between Gram matrices H. No gauge-free entrywise stability of measurement factors R is asserted.")
    stages = [("closed_form", closed_form_experiment), ("sdp_scalar", sdp_scalar_experiment),
              ("nonconvex", nonconvex_experiment), ("stability", stability_experiment), ("k4", k4_experiment)]
    for name, func in stages:
        start = time.perf_counter()
        print(f"Starting {name}", flush=True)
        summary[name] = func()
        summary[name]["seconds"] = time.perf_counter() - start
        print(f"Completed {name}: {summary[name]['seconds']:.2f}s", flush=True)
    validation = dict(
        construction_trace_matches=summary["closed_form"]["max_trace_error"] < 1e-9,
        construction_original_psd=summary["closed_form"]["min_original_eig"] > -1e-9,
        sdp_scalar_crosscheck=summary["sdp_scalar"]["max_abs_kappa_error"] < 1e-8 and summary["sdp_scalar"]["max_abs_alpha_error"] < 1e-8,
        k4_exact_continuous_certificates=summary["k4"]["all_rational_polynomial_certificates_pass"] and summary["k4"]["common_direction_certificate_pass"],
        stability_all_paths_feasible=min(x["min_qmi_eig"] for x in summary["stability"]["fits"]) > -1e-9,
        all_nonconvex_groups_find_a_close_repaired_candidate=all(x["best_scaled_repair_relative_gap"] is not None and -1e-9 < x["best_scaled_repair_relative_gap"] < 1e-6 for x in summary["nonconvex"]["groups"]))
    summary["validation"] = validation
    if not all(validation.values()):
        raise AssertionError(f"Validation checks failed: {validation}")
    summary["total_seconds"] = time.perf_counter() - begin
    (DATA / "environment.json").write_text(json.dumps(environment(), indent=2, ensure_ascii=False), encoding="utf-8")
    (DATA / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"total_seconds": summary["total_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
