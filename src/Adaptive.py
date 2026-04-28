# controllers/robust_adaptive_z.py
import numpy as np
from collections import deque
from dataclasses import dataclass

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

@dataclass
class ZCLFParams:
    alpha: float = 1.0
    P: np.ndarray = np.array([[2.0, 0.5],
                              [0.5, 1.0]], dtype=float)  # SPD

@dataclass
class ThetaProjBounds:
    theta_min: float
    theta_max: float

def V_and_Vx_quadratic(e: np.ndarray, clf: ZCLFParams):
    e = np.asarray(e).reshape(2,)
    P = clf.P
    V = 0.5 * float(e.T @ P @ e)
    Vx = (P @ e).reshape(2,)  # dV/de
    return V, Vx

class ThetaEstimator:
    """
    dot(theta_hat) = Gamma * (PY)^T * (Vx)^T
    + projection theta_hat in [theta_min, theta_max]
    """
    def __init__(self, Gamma: float, bounds: ThetaProjBounds):
        self.Gamma = float(Gamma)
        self.bounds = bounds
        self.theta_hat = 0.0

    def reset(self, theta_hat0: float):
        self.theta_hat = clamp(theta_hat0, self.bounds.theta_min, self.bounds.theta_max)

    def step(self, dt: float, PY: np.ndarray, Vx: np.ndarray):
        PY = np.asarray(PY).reshape(2,)
        Vx = np.asarray(Vx).reshape(2,)
        dtheta = self.Gamma * float(np.dot(PY, Vx))  # (PY)^T (Vx)^T
        self.theta_hat = clamp(self.theta_hat + dt * dtheta,
                               self.bounds.theta_min, self.bounds.theta_max)
        return self.theta_hat

class ACIQuantileBound:
    """
    Online conformal-ish bound for a nonnegative score s_k.
    Maintains rolling window, updates alpha_k to target miscoverage.
    """
    def __init__(self, alpha_target=0.1, window=400, eta=0.01,
                 alpha_min=1e-3, alpha_max=0.5):
        self.alpha_target = float(alpha_target)
        self.alpha_k = float(alpha_target)
        self.eta = float(eta)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.buf = deque(maxlen=int(window))
        self.Rbar = 0.0
        self.rho_bar = 0.0

    def update(self, score: float):
        s = float(abs(score))
        self.buf.append(s)
        """
        if len(self.buf) < 20:
            self.Rbar = max(self.Rbar, s)
            return self.Rbar
        """
        q = float(np.quantile(np.array(self.buf), 1.0 - self.alpha_k))
        self.Rbar = q

        violation = 1.0 if (s > q) else 0.0
        # if violation -> alpha decreases -> more conservative
        self.alpha_k = self.alpha_k + self.eta * (self.alpha_target - violation)
        self.alpha_k = clamp(self.alpha_k, self.alpha_min, self.alpha_max)
        return self.Rbar

def scalar_min_intervention_halfspace(a: float, rhs: float):
    """
    min |u| s.t. a*u <= rhs (scalar).
    """
    eps = 1e-9
    if abs(a) < eps:
        return 0.0
    if a > 0:
        umax = rhs / a
        return 0.0 if (0.0 <= umax) else float(umax)
    else:
        umin = rhs / a
        return 0.0 if (0.0 >= umin) else float(umin)

def robust_clf_filter(
    e: np.ndarray,
    f: np.ndarray,
    B: np.ndarray,
    uc: float,
    uad: float,
    rho_bar: float,
    clf: ZCLFParams,
    u_limits: tuple[float, float]
):
    """
    Enforce:
      Vx(f + B(uc+uad+usafe)) + ||Vx|| Rbar <= -2 alpha V
    Return: u_total, usafe, V, Vx
    """
    V, Vx = V_and_Vx_quadratic(e, clf)
    f = np.asarray(f).reshape(2,)
    B = np.asarray(B).reshape(2,)

    a = float(np.dot(Vx, B))
    base = float(np.dot(Vx, f + B * (uc + uad)))
    
    rhs = -2 * clf.alpha * V - rho_bar - base

    usafe = scalar_min_intervention_halfspace(a, rhs)
    u = uc + uad + usafe
    u = clamp(u, u_limits[0], u_limits[1])
    return u, usafe, V, Vx
