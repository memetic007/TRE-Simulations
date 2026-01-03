"""
TRE Simulation 2 (Corrected): Adaptive Kuramoto Model (Slowly Evolving Constraints)

This script implements a Kuramoto oscillator network with a slowly evolving
constraint parameter K(t), giving explicit temporal separation between:

  • Fast micro-dynamics: oscillator phases φ_i(t)
  • Slow constraint dynamics: coupling strength K(t)

Model:
    dφ_i/dt = ω_i + K(t) * R(t) * sin(Ψ(t) - φ_i) + σ ξ_i(t)
    dK/dt   = ε * (α * R(t) + K_basal - K(t))

where:
    Z(t) = (1/N) Σ exp(i φ_j(t)) = R(t) exp(i Ψ(t))

TRE signatures demonstrated:
  1) Lagged co-evolution: R(t) tends to lead K(t) when ε is small.
  2) Two-phase recovery: scrambling microstate recovers quickly if K is intact,
     but recovers slowly if K is destroyed and must rebuild (timescale ~1/ε).
  3) TE-based directionality (optional but included): compute windowed TE in
     micro-scale vs macro-scale windows.

Corrections vs the previous version:
  • Cross-correlation lag search is scaled to the expected lag ~O(1/ε) instead
    of being hard-coded to ±100 steps.
  • Recovery-time measurement uses a *sustained* threshold crossing.
  • The TE section is implemented explicitly (the old header claimed TE but
    the code did not compute it).

Author: (corrected by ChatGPT)
"""

from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Literal

from scipy.ndimage import uniform_filter1d


# =============================================================================
# Utilities
# =============================================================================

def _wrap_to_2pi(ph: np.ndarray) -> np.ndarray:
    return np.mod(ph, 2.0 * np.pi)


def _compute_order_parameter(phases: np.ndarray) -> Tuple[float, float, complex]:
    """Return (R, Psi, Z) for a phase vector."""
    Z = np.mean(np.exp(1j * phases))
    return float(np.abs(Z)), float(np.angle(Z)), Z


def scramble_phases_uniform(N: int, rng: np.random.Generator) -> np.ndarray:
    """Randomize phases uniformly to destroy synchronization."""
    return rng.uniform(0, 2*np.pi, size=N)


# =============================================================================
# 1) Simulation: Adaptive Kuramoto
# =============================================================================

@dataclass
class AdaptiveKuramotoConfig:
    N: int = 100
    T: float = 200.0
    dt: float = 0.05
    epsilon: float = 0.02  # slow constraint evolution
    alpha: float = 6.0     # gain driving K from R
    K_basal: float = 1.0
    noise_std: float = 0.5
    omega_mean: float = 1.0
    omega_std: float = 0.5
    seed: int = 42


def run_adaptive_kuramoto(cfg: AdaptiveKuramotoConfig,
                          init_phases: Optional[np.ndarray] = None,
                          init_K: Optional[float] = None,
                          fixed_K: Optional[float] = None,
                          omegas: Optional[np.ndarray] = None,
                          seed: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Simulate the adaptive Kuramoto model (Euler–Maruyama).

    If fixed_K is not None, the coupling is clamped to fixed_K (control).
    """
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    steps = int(cfg.T / cfg.dt)
    time = np.arange(steps) * cfg.dt

    # Frequencies
    if omegas is None:
        omegas = rng.normal(cfg.omega_mean, cfg.omega_std, size=cfg.N)
    else:
        omegas = np.array(omegas, dtype=float).copy()
        if omegas.shape != (cfg.N,):
            raise ValueError(f"omegas must be shape (N,), got {omegas.shape}")

    # Initial phases
    if init_phases is None:
        phases = rng.uniform(0, 2*np.pi, size=cfg.N)
    else:
        phases = np.array(init_phases, dtype=float).copy()
        if phases.shape != (cfg.N,):
            raise ValueError(f"init_phases must be shape (N,), got {phases.shape}")

    K = float(0.0 if init_K is None else init_K)

    R_hist = np.zeros(steps)
    K_hist = np.zeros(steps)
    Z_hist = np.zeros(steps, dtype=complex)

    for t in range(steps):
        # Macro state
        R, Psi, Z = _compute_order_parameter(phases)
        Z_hist[t] = Z
        R_hist[t] = R
        K_hist[t] = (fixed_K if fixed_K is not None else K)

        # Slow constraint dynamics
        if fixed_K is None:
            target_K = cfg.alpha * R + cfg.K_basal
            K = K + cfg.epsilon * (target_K - K) * cfg.dt
            K_current = K
        else:
            K_current = float(fixed_K)

        # Micro dynamics
        noise = rng.normal(0.0, cfg.noise_std, size=cfg.N) * math.sqrt(cfg.dt)
        coupling = K_current * R * np.sin(Psi - phases)
        phases = _wrap_to_2pi(phases + (omegas + coupling) * cfg.dt + noise)

    return {
        "time": time,
        "R": R_hist,
        "K": K_hist,
        "Z": Z_hist,
        "phases_final": phases,
        "omegas": omegas,
    }


def run_to_steady_state(cfg: AdaptiveKuramotoConfig,
                        T_max: float = 250.0,
                        R_threshold: float = 0.85) -> Dict[str, np.ndarray]:
    """
    Run until near steady-state (high synchronization) and return final state.
    """
    cfg2 = AdaptiveKuramotoConfig(**{**cfg.__dict__, "T": T_max})
    sim = run_adaptive_kuramoto(cfg2)
    R_tail = float(np.mean(sim["R"][-200:]))
    K_tail = float(np.mean(sim["K"][-200:]))
    print(f"  Steady-state estimate: R≈{R_tail:.3f}, K≈{K_tail:.3f}")
    if R_tail < R_threshold:
        print("  [warn] R_threshold not reached; consider increasing K_basal/alpha or lowering noise.")
    return sim


# =============================================================================
# 2) Analysis helpers
# =============================================================================

def cross_correlation(x: np.ndarray, y: np.ndarray, max_lag_steps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cross-correlation for lags in [-max_lag_steps, +max_lag_steps].

    Positive lag means x leads y (x(t) correlates with y(t+lag)).
    """
    x = np.asarray(x); y = np.asarray(y)
    xz = (x - x.mean()) / (x.std() + 1e-12)
    yz = (y - y.mean()) / (y.std() + 1e-12)

    lags = np.arange(-max_lag_steps, max_lag_steps + 1)
    corrs = np.zeros_like(lags, dtype=float)

    for i, lag in enumerate(lags):
        if lag == 0:
            corrs[i] = float(np.mean(xz * yz))
        elif lag > 0:
            corrs[i] = float(np.mean(xz[:-lag] * yz[lag:]))
        else:
            L = -lag
            corrs[i] = float(np.mean(xz[L:] * yz[:-L]))
    return lags, corrs


def measure_recovery_time(R: np.ndarray, dt: float,
                          threshold_fraction: float = 0.8,
                          sustain_time: float = 2.0,
                          smooth_window_sec: float = 1.0) -> float:
    """
    Recovery time = first time the smoothed R exceeds the threshold
    and stays above it for at least sustain_time.

    Returns np.inf if no recovery.
    """
    R = np.asarray(R)
    R_final = float(np.mean(R[-200:]))
    thr = threshold_fraction * R_final

    win = max(1, int(round(smooth_window_sec / dt)))
    R_s = uniform_filter1d(R, size=win) if win > 1 else R

    sustain_steps = max(1, int(round(sustain_time / dt)))
    above = (R_s >= thr).astype(int)

    # rolling sum to find sustained segments
    if len(above) < sustain_steps:
        return float("inf")
    rolling = np.convolve(above, np.ones(sustain_steps, dtype=int), mode="valid")
    idx = np.where(rolling == sustain_steps)[0]
    if idx.size == 0:
        return float("inf")
    first = int(idx[0])
    return first * dt


# =============================================================================
# 3) Optional: TE tools (fast discrete estimator)
# =============================================================================

def _quantile_edges(x: np.ndarray, bins: int) -> np.ndarray:
    qs = np.linspace(0, 100, bins + 1)
    e = np.percentile(x, qs)
    e[0] -= 1e-12
    e[-1] += 1e-12
    for i in range(1, len(e)):
        if e[i] <= e[i-1]:
            e[i] = np.nextafter(e[i-1], np.inf)
    return e


def _discretize(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = len(edges) - 1
    d = np.digitize(x, edges[1:-1], right=False)
    return np.clip(d, 0, bins - 1).astype(np.int64)


def _encode(digits: np.ndarray, bins: int) -> np.ndarray:
    digits = np.asarray(digits, dtype=np.int64)
    if digits.ndim == 1:
        digits = digits[:, None]
    powers = (bins ** np.arange(digits.shape[1], dtype=np.int64))
    return (digits * powers).sum(axis=1)


def _entropy(states: np.ndarray) -> float:
    if states.size == 0:
        return 0.0
    _, c = np.unique(states, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def _cond_entropy(x_states: np.ndarray, y_states: np.ndarray) -> float:
    nx = int(x_states.max()) + 1 if x_states.size else 1
    joint = x_states + nx * y_states
    return _entropy(joint) - _entropy(y_states)


def te_discrete(source: np.ndarray, target: np.ndarray, lag: int,
                k_target: int = 3, bins: int = 6,
                edges_source: Optional[List[np.ndarray]] = None,
                edges_target: Optional[List[np.ndarray]] = None) -> float:
    """
    TE(source -> target) with conditioning on target past.

    All variables can be multivariate but should remain low-dimensional.
    """
    source = np.asarray(source); target = np.asarray(target)
    if source.ndim == 1: source = source[:, None]
    if target.ndim == 1: target = target[:, None]
    T = target.shape[0]
    lag = int(lag)
    start = k_target - 1
    end = T - lag
    if end <= start + 1:
        return float("nan")

    tgt_future = target[start + lag : end + lag]
    src_present = source[start : end]
    past_segments = [target[start - j : end - j] for j in range(k_target)]

    dt_dim = target.shape[1]
    ds_dim = source.shape[1]

    if edges_target is None:
        edges_target = [_quantile_edges(target[:, d], bins) for d in range(dt_dim)]
    if edges_source is None:
        edges_source = [_quantile_edges(source[:, d], bins) for d in range(ds_dim)]

    future_digits = np.column_stack([_discretize(tgt_future[:, d], edges_target[d]) for d in range(dt_dim)])
    future_state = _encode(future_digits, bins=bins)

    source_digits = np.column_stack([_discretize(src_present[:, d], edges_source[d]) for d in range(ds_dim)])
    source_state = _encode(source_digits, bins=bins)
    n_source_states = bins ** ds_dim

    past_digits_list = []
    for seg in past_segments:
        seg_digits = np.column_stack([_discretize(seg[:, d], edges_target[d]) for d in range(dt_dim)])
        past_digits_list.append(seg_digits)
    past_state = _encode(np.concatenate(past_digits_list, axis=1), bins=bins)

    H_future_given_past = _cond_entropy(future_state, past_state)
    joint_sp = source_state + n_source_states * past_state
    H_future_given_sp = _cond_entropy(future_state, joint_sp)
    return max(0.0, float(H_future_given_past - H_future_given_sp))


# =============================================================================
# 4) Main experiments
# =============================================================================

if __name__ == "__main__":
    print("=" * 78)
    print("TRE Simulation 2 (Corrected): Adaptive Kuramoto (Slowly Evolving Constraints)")
    print("=" * 78)

    # Base parameters
    cfg = AdaptiveKuramotoConfig(
        N=100,
        T=200.0,
        dt=0.05,
        epsilon=0.02,
        alpha=6.0,
        K_basal=1.0,
        noise_std=0.5,
        seed=42,
    )

    # -------------------------------------------------------------------------
    # Experiment 1: Lagged co-evolution (R leads K)
    # -------------------------------------------------------------------------
    print("\n[Experiment 1] Lag analysis: does R lead K?")
    cfg_lag = AdaptiveKuramotoConfig(**{**cfg.__dict__, "T": 400.0, "epsilon": 0.02})
    sim_lag = run_adaptive_kuramoto(cfg_lag, init_K=0.0)

    # Expected lag ~ O(1/epsilon)
    expected_lag_sec = 1.0 / cfg_lag.epsilon
    max_lag_sec = min(3.0 * expected_lag_sec, 200.0)  # cap to keep plots reasonable
    max_lag_steps = int(round(max_lag_sec / cfg_lag.dt))

    lags, corrs = cross_correlation(sim_lag["R"], sim_lag["K"], max_lag_steps=max_lag_steps)
    lag_times = lags * cfg_lag.dt

    peak_idx = int(np.argmax(corrs))
    peak_lag_sec = float(lag_times[peak_idx])
    print(f"  Expected lag scale ~ 1/ε = {expected_lag_sec:.1f}s")
    print(f"  Cross-corr peak at lag = {peak_lag_sec:.2f}s (positive = R leads K)")

    # -------------------------------------------------------------------------
    # Experiment 2: Two-phase recovery (perturbation test)
    # -------------------------------------------------------------------------
    print("\n[Experiment 2] Two-phase recovery: constraint sensitivity")
    print("Phase 1: run to steady state...")
    steady = run_to_steady_state(cfg, T_max=200.0)

    # Prepare perturbation initial conditions
    rng = np.random.default_rng(123)
    scrambled = scramble_phases_uniform(cfg.N, rng)

    T_recovery = 80.0
    cfg_rec = AdaptiveKuramotoConfig(**{**cfg.__dict__, "T": T_recovery})

    print("Phase 2: perturbation scenarios...")

    # Scenario A: micro perturbation only (scramble phases, keep K)
    sim_A = run_adaptive_kuramoto(
        cfg_rec,
        init_phases=scrambled.copy(),
        init_K=float(steady["K"][-1]),
        omegas=steady["omegas"],
        seed=200
    )

    # Scenario B: micro + constraint perturbation (scramble phases, reset K)
    sim_B = run_adaptive_kuramoto(
        cfg_rec,
        init_phases=scrambled.copy(),
        init_K=0.0,
        omegas=steady["omegas"],
        seed=200
    )

    # Scenario C: control (fixed K at steady value, no adaptation)
    sim_C = run_adaptive_kuramoto(
        cfg_rec,
        init_phases=scrambled.copy(),
        fixed_K=float(steady["K"][-1]),
        omegas=steady["omegas"],
        seed=200
    )

    # Recovery times
    tau_A = measure_recovery_time(sim_A["R"], dt=cfg.dt, threshold_fraction=0.8, sustain_time=2.0)
    tau_B = measure_recovery_time(sim_B["R"], dt=cfg.dt, threshold_fraction=0.8, sustain_time=2.0)
    tau_C = measure_recovery_time(sim_C["R"], dt=cfg.dt, threshold_fraction=0.8, sustain_time=2.0)

    print(f"  Recovery to 80% (sustained):")
    print(f"    A (K intact): {tau_A:.2f}s")
    print(f"    B (K reset):  {tau_B:.2f}s")
    print(f"    C (K fixed):  {tau_C:.2f}s")
    if math.isfinite(tau_A) and math.isfinite(tau_B):
        print(f"    Ratio B/A: {tau_B/(tau_A+1e-9):.2f}x slower when constraint destroyed")

    # -------------------------------------------------------------------------
    # Experiment 3: TE-based directional signature (optional)
    # -------------------------------------------------------------------------
    print("\n[Experiment 3] TE directionality (coarse, discrete estimator)")
    # Build macro time series for TE: we include both R and K as the "macro+constraint" source
    # because K is the slowly evolving constraint that actually modulates micro-dynamics.
    # Micro observable: (cos φ, sin φ) for a sample of oscillators.
    #
    # NOTE: This is an empirical diagnostic; TE magnitudes depend on bins, k_target, and noise.
    # The key idea is scale-separation: short-lag TE↑ tends to capture bottom-up emergence,
    # while long-lag TE↓ can reflect slow constraint-mediated top-down influence.
    bins = 6
    k_micro = 5
    k_macro = 3
    max_lag_sec = min(60.0, 3.0 / cfg.epsilon)  # cover O(1/ε)
    max_lag_steps = int(max_lag_sec / cfg.dt)
    # Use a coarse lag grid for speed
    lag_grid = np.unique(np.linspace(1, max_lag_steps, num=min(60, max_lag_steps), dtype=int))

    # Use post-transient segment from the lag-run sim
    start = len(sim_lag["R"]) // 4
    R_seg = sim_lag["R"][start:]
    K_seg = sim_lag["K"][start:]
    Z_seg = sim_lag["Z"][start:]
    phases_for_te = None  # we didn't store full trajectory; for a TE demo, re-run with stored phases if desired

    # For a simple, self-contained TE demo, re-run a shorter simulation storing phases:
    cfg_te = AdaptiveKuramotoConfig(**{**cfg.__dict__, "T": 220.0})
    rng = np.random.default_rng(999)
    # Re-run with explicit storing by slightly modifying the sim:
    # (We just re-run and reconstruct phases from order parameter isn't possible.)
    # We'll keep the TE diagnostic minimal: TE from (R,K) → R (macro autocausality) isn't meaningful,
    # so we do a small run saving phases for TE.
    #
    # To keep code simple, we implement a small local run here.
    steps = int(cfg_te.T / cfg_te.dt)
    time = np.arange(steps) * cfg_te.dt
    omegas = rng.normal(cfg_te.omega_mean, cfg_te.omega_std, size=cfg_te.N)
    phases = rng.uniform(0, 2*np.pi, size=cfg_te.N)
    K = 0.0
    R_hist = np.zeros(steps)
    K_hist = np.zeros(steps)
    ph_traj = np.zeros((steps, cfg_te.N))
    for t in range(steps):
        R, Psi, _ = _compute_order_parameter(phases)
        R_hist[t]=R; K_hist[t]=K; ph_traj[t]=phases
        target_K = cfg_te.alpha * R + cfg_te.K_basal
        K = K + cfg_te.epsilon * (target_K - K) * cfg_te.dt
        noise = rng.normal(0.0, cfg_te.noise_std, size=cfg_te.N) * math.sqrt(cfg_te.dt)
        phases = _wrap_to_2pi(phases + (omegas + K * R * np.sin(Psi - phases)) * cfg_te.dt + noise)

    # Define TE series
    start = steps // 4
    macro_source = np.column_stack([R_hist[start:], K_hist[start:]])  # (T,2)
    macro_target = R_hist[start:].reshape(-1,1)  # (T,1)

    # Choose oscillators and define micro variable (cos,sin)
    idx = rng.choice(cfg_te.N, size=12, replace=False)
    micro = np.stack([np.cos(ph_traj[start:, idx]), np.sin(ph_traj[start:, idx])], axis=2)  # (T,12,2)

    # Compute TE↑: micro -> macro_target (R)
    # Compute TE↓: macro_source -> micro
    te_up = []
    te_down = []
    edges_R = [_quantile_edges(macro_target[:,0], bins)]
    edges_macro_src = [_quantile_edges(macro_source[:,0], bins), _quantile_edges(macro_source[:,1], bins)]
    edges_micro = [_quantile_edges(micro[...,0].ravel(), bins), _quantile_edges(micro[...,1].ravel(), bins)]

    for lag in lag_grid:
        # TE↑: pool across oscillators
        up_vals=[]
        for j in range(micro.shape[1]):
            up_vals.append(te_discrete(
                source=micro[:,j,:],
                target=macro_target,
                lag=lag,
                k_target=k_macro,
                bins=bins,
                edges_source=edges_micro,
                edges_target=edges_R
            ))
        te_up.append(float(np.nanmean(up_vals)))

        # TE↓: pool across oscillators
        down_vals=[]
        for j in range(micro.shape[1]):
            down_vals.append(te_discrete(
                source=macro_source,
                target=micro[:,j,:],
                lag=lag,
                k_target=k_micro,
                bins=bins,
                edges_source=edges_macro_src,
                edges_target=edges_micro
            ))
        te_down.append(float(np.nanmean(down_vals)))

    lag_sec = lag_grid * cfg_te.dt

    # Windows
    W_micro = (0.05, 0.5)
    W_macro = (0.5/ cfg_te.epsilon, 1.5/ cfg_te.epsilon)  # around O(1/ε)

    up_micro = np.nanmean([v for t,v in zip(lag_sec, te_up) if W_micro[0] <= t <= W_micro[1]])
    down_micro = np.nanmean([v for t,v in zip(lag_sec, te_down) if W_micro[0] <= t <= W_micro[1]])
    up_macro = np.nanmean([v for t,v in zip(lag_sec, te_up) if W_macro[0] <= t <= W_macro[1]])
    down_macro = np.nanmean([v for t,v in zip(lag_sec, te_down) if W_macro[0] <= t <= W_macro[1]])

    tre_strength = math.log((down_macro + 1e-6)/(up_micro + 1e-6))

    print(f"  Micro window [{W_micro[0]:.2f},{W_micro[1]:.2f}]s: TE↑={up_micro:.4g}, TE↓={down_micro:.4g}")
    print(f"  Macro window [{W_macro[0]:.1f},{W_macro[1]:.1f}]s: TE↑={up_macro:.4g}, TE↓={down_macro:.4g}")
    print(f"  TRE strength log(TE↓_macro/TE↑_micro): {tre_strength:.3f}")

    # -------------------------------------------------------------------------
    # Plotting summary
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("TRE Empirical Signatures: Slowly Evolving Constraint (Corrected)", fontsize=14)

    # Panel A: R and K co-evolution
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(sim_lag["time"], sim_lag["R"], label="R(t) (macro)", alpha=0.8)
    ax1.plot(sim_lag["time"], sim_lag["K"], label="K(t) (constraint)", alpha=0.8)
    ax1.set_title("A. Co-evolution: R and slowly adapting K")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Value")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Panel B: Cross-correlation (lag)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(lag_times, corrs, lw=2)
    ax2.axvline(0, linestyle=":", alpha=0.6)
    ax2.axvline(peak_lag_sec, linestyle="--", label=f"peak lag={peak_lag_sec:.2f}s")
    ax2.set_title("B. Cross-correlation: positive lag means R leads K")
    ax2.set_xlabel("Lag (s)")
    ax2.set_ylabel("Corr")
    ax2.grid(alpha=0.3)
    ax2.legend()

    # Panel C: Recovery in R
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(sim_A["time"], sim_A["R"], label=f"A: K intact (τ={tau_A:.1f}s)", lw=2)
    ax3.plot(sim_B["time"], sim_B["R"], label=f"B: K reset (τ={tau_B:.1f}s)", lw=2)
    ax3.plot(sim_C["time"], sim_C["R"], label=f"C: K fixed (τ={tau_C:.1f}s)", lw=2, linestyle="--")
    ax3.set_title("C. Two-phase recovery in R after perturbation")
    ax3.set_xlabel("Time after perturbation (s)")
    ax3.set_ylabel("R")
    ax3.set_ylim(0, 1.05)
    ax3.grid(alpha=0.3)
    ax3.legend(loc="lower right", fontsize=9)

    # Panel D: Constraint dynamics during recovery
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(sim_A["time"], sim_A["K"], label="A: K intact", lw=2)
    ax4.plot(sim_B["time"], sim_B["K"], label="B: K rebuilding", lw=2)
    ax4.plot(sim_C["time"], sim_C["K"], label="C: K fixed", lw=2, linestyle="--")
    ax4.set_title("D. K dynamics during recovery")
    ax4.set_xlabel("Time after perturbation (s)")
    ax4.set_ylabel("K")
    ax4.grid(alpha=0.3)
    ax4.legend(loc="lower right", fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_png = "output_adaptive_kuramoto_tre_corrected.png"
    plt.savefig(out_png, dpi=160)
    print(f"\nSaved: {out_png}")
    plt.close()
