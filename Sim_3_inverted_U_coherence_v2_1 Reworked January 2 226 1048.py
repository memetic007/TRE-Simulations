"""
TRE Simulation 3 (Revised): Inverted-U Cross-Scale Coherence with Controls

This script is a revised, appendix-grade simulation intended to support the TRE paper's
Signature #4 ("Non-monotonic cross-scale coherence"). Compared to the earlier version, it adds:

  1) Efficient correlation-time estimation using FFT-based autocorrelation (O(n log n)).
  2) Leave-one-out macro predictor for micro↔macro metrics (avoids self-inclusion artifacts).
  3) Multiple trials per coupling K with mean ± 95% CI.
  4) Surrogate controls (circular time shifts) for each cross-scale metric.
  5) Proper PAC estimator (Tort modulation index) using zero-phase filtering (filtfilt).
  6) Optional diachronic top-down influence via a macro delay in the coupling term:
       φ̇_i(t) = ω_i + K r(t-τ_d) sin(ψ(t-τ_d) - φ_i(t)) + noise

DEFAULT / RECOMMENDED RUN (no command-line options):
  python Sim_3_inverted_U_coherence_v2_1.py

Recommended defaults used when no CLI options are provided:
  N=100, T=200.0 s, dt=0.05 s, noise=0.4, trials=10, delay=2.0 s, seed=42
  K sweep (built-in): [0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.8, 3.5, 4.5, 6.0, 8.0, 10.0, 15.0]

Outputs (by default):
  - output_inverted_u_coherence_v2.png     (single 2×2 figure for appendix)
  - output_inverted_u_coherence_v2.csv     (per-K summary results incl. mean ± 95% CI)

Dependencies: numpy, matplotlib, scipy

Optional arguments (examples):
  --trials 20 --N 200 --T 300 --dt 0.02 --delay 0 --noise 0.4
  --out myfigure.png --csv myresults.csv
"""


from __future__ import annotations

import argparse
import csv
import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, coherence, hilbert
from scipy.stats import t as student_t


# -----------------------------
# Recommended defaults (used when no CLI options are provided)
# -----------------------------

DEFAULT_N = 100
DEFAULT_T = 200.0
DEFAULT_DT = 0.05
DEFAULT_NOISE = 0.4
DEFAULT_TRIALS = 10
DEFAULT_DELAY = 2.0
DEFAULT_SEED = 42
DEFAULT_OUT = "output_inverted_u_coherence_v2.png"
DEFAULT_CSV = None   # if None, derived from --out (replace extension with .csv)
DEFAULT_BURN_IN_FRAC = 0.2

# -----------------------------
# Utilities
# -----------------------------

def _ensure_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return x.reshape(-1)

def mean_ci(x: np.ndarray, ci: float = 0.95) -> Tuple[float, float, float]:
    """Mean and symmetric CI using Student-t (works well for small n)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    m = float(np.mean(x))
    if n == 1:
        return (m, m, m)
    s = float(np.std(x, ddof=1))
    se = s / math.sqrt(n)
    alpha = 1.0 - ci
    h = float(student_t.ppf(1 - alpha / 2.0, df=n - 1) * se)
    return (m, m - h, m + h)

def normalize01(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    return (x - lo) / (hi - lo + eps)

def random_circular_shift(rng: np.random.Generator, n: int, min_frac: float = 0.1) -> int:
    """Pick a shift away from 0 to avoid trivial surrogates."""
    if n < 10:
        return 1
    min_shift = int(max(1, min_frac * n))
    # Valid shifts: [min_shift, n-min_shift]
    max_shift = n - min_shift
    if max_shift <= min_shift:
        return int(n // 2)
    return int(rng.integers(min_shift, max_shift))

def bandpass_filtfilt(x: np.ndarray, fs: float, f_lo: float, f_hi: float, order: int = 4) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass filter.
    Returns filtered signal; if the band is invalid, returns demeaned x.
    """
    x = _ensure_1d(x).astype(float)
    x = x - np.mean(x)
    nyq = 0.5 * fs
    f_lo2 = max(1e-6, float(f_lo))
    f_hi2 = float(f_hi)
    if f_hi2 <= f_lo2:
        return x
    if f_hi2 >= nyq * 0.999:
        f_hi2 = nyq * 0.999
    if f_lo2 <= 0:
        f_lo2 = 1e-6
    if f_lo2 >= nyq * 0.999:
        return x
    wn = [f_lo2 / nyq, f_hi2 / nyq]
    try:
        b, a = butter(order, wn, btype="bandpass")
        return filtfilt(b, a, x, method="pad")
    except Exception:
        # Fallback: return demeaned signal if filtering fails
        return x

def autocorr_fft(x: np.ndarray) -> np.ndarray:
    """
    FFT-based autocorrelation (normalized so acf[0] = 1).
    Uses Wiener–Khinchin theorem: acf = ifft(|fft(x)|^2).
    """
    x = _ensure_1d(x).astype(float)
    x = x - np.mean(x)
    n = len(x)
    var = np.var(x)
    if n < 4 or var < 1e-15:
        return np.ones(max(n, 1), dtype=float)
    nfft = 1 << int((2 * n - 1).bit_length())  # next pow2
    f = np.fft.rfft(x, n=nfft)
    p = f * np.conjugate(f)
    acf = np.fft.irfft(p, n=nfft)[:n].real
    acf /= acf[0] + 1e-15
    return acf

def integral_timescale(x: np.ndarray, dt: float, max_lag_s: float = 30.0) -> float:
    """
    Integral correlation time τ = dt * sum_{lag>=0} acf(lag),
    truncated at first negative acf or max_lag_s (whichever comes first).
    """
    acf = autocorr_fft(x)
    max_lag = min(len(acf) - 1, int(max_lag_s / dt))
    if max_lag < 1:
        return dt
    # integrate positive portion
    k_end = 1
    for k in range(1, max_lag + 1):
        if acf[k] < 0:
            break
        k_end = k
    tau = float(dt * np.sum(acf[: k_end + 1]))
    return max(tau, dt)


# -----------------------------
# Proper PAC (Tort MI)
# -----------------------------

def tort_modulation_index(phase: np.ndarray, amp: np.ndarray, n_bins: int = 18) -> float:
    """
    Tort et al. modulation index: KL divergence of phase-binned amplitude distribution
    from uniform distribution.

    phase: instantaneous phase (radians, -pi..pi)
    amp: amplitude envelope (>=0)
    """
    phase = _ensure_1d(phase)
    amp = _ensure_1d(amp)
    if len(phase) != len(amp) or len(phase) < 10:
        return 0.0
    amp = np.maximum(amp, 0.0)

    # Bin phase
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_idx = np.digitize(phase, edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    mean_amp = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=float)

    for b in range(n_bins):
        mask = (bin_idx == b)
        counts[b] = float(np.sum(mask))
        if counts[b] > 0:
            mean_amp[b] = float(np.mean(amp[mask]))
        else:
            mean_amp[b] = 0.0

    if np.all(mean_amp <= 1e-12):
        return 0.0

    p = mean_amp / (np.sum(mean_amp) + 1e-15)
    # KL(p || uniform)
    u = 1.0 / n_bins
    kl = float(np.sum(p * np.log((p + 1e-15) / u)))
    mi = kl / math.log(n_bins)
    return float(max(mi, 0.0))


# -----------------------------
# Kuramoto Simulation (with macro delay)
# -----------------------------

@dataclass
class SimConfig:
    N: int = DEFAULT_N
    T: float = DEFAULT_T
    dt: float = DEFAULT_DT
    K: float = 2.0
    noise_std: float = DEFAULT_NOISE
    omega_mean: float = 1.0
    omega_std: float = 0.4
    macro_delay: float = DEFAULT_DELAY   # seconds; 0 disables
    burn_in: float = DEFAULT_BURN_IN_FRAC  # fraction of time removed as transient

def run_kuramoto(cfg: SimConfig, seed: int) -> Dict[str, np.ndarray]:
    """
    Simulate Kuramoto oscillators with optional delayed macro coupling.
    Returns time, phases (steps x N), and macro trajectory Z(t).
    """
    rng = np.random.default_rng(seed)

    steps = int(round(cfg.T / cfg.dt))
    t = np.arange(steps, dtype=float) * cfg.dt

    phases = rng.uniform(0.0, 2.0 * np.pi, size=cfg.N)
    omegas = rng.normal(cfg.omega_mean, cfg.omega_std, size=cfg.N)

    phase_traj = np.empty((steps, cfg.N), dtype=float)
    Z_hist = np.empty(steps, dtype=np.complex128)

    delay_steps = int(round(max(0.0, cfg.macro_delay) / cfg.dt))

    for s in range(steps):
        # Save state at time t_s
        phase_traj[s, :] = phases
        Z_now = np.mean(np.exp(1j * phases))
        Z_hist[s] = Z_now

        # Macro state for coupling (possibly delayed)
        if delay_steps > 0 and s - delay_steps >= 0:
            Z_use = Z_hist[s - delay_steps]
        else:
            Z_use = Z_now

        R_use = np.abs(Z_use)
        Psi_use = np.angle(Z_use)

        coupling = cfg.K * R_use * np.sin(Psi_use - phases)

        # Euler-Maruyama update
        noise = rng.normal(0.0, cfg.noise_std, size=cfg.N) * math.sqrt(cfg.dt)
        phases = phases + (omegas + coupling) * cfg.dt + noise
        phases = np.mod(phases, 2.0 * np.pi)

    return {
        "t": t,
        "phases": phase_traj,
        "Z": Z_hist,
        "omegas": omegas,
    }


# -----------------------------
# Metrics (leave-one-out + surrogates)
# -----------------------------

@dataclass
class MetricConfig:
    # Sampling
    n_micro_samples: int = 10   # oscillators sampled for micro↔macro metrics
    n_pac_samples: int = 6
    # Surrogates
    surrogate_min_shift_frac: float = 0.15

    # Coherence band (for spectral coherence)
    coh_f_lo: float = 0.02
    coh_f_hi: float = 1.5

    # PAC bands (low phase from macro R, high amp from micro cos(phi))
    pac_low_lo: float = 0.02
    pac_low_hi: float = 0.12
    pac_high_lo: float = 0.12
    pac_high_hi: float = 1.2
    pac_bins: int = 18

    # MI estimation
    mi_bins: int = 12

    # Timescale estimation
    max_lag_s: float = 30.0


def mutual_information_discrete(x_bin: np.ndarray, y_bin: np.ndarray, n_x: int, n_y: int) -> float:
    """Mutual information (nats) between two discrete variables."""
    x_bin = np.asarray(x_bin, dtype=int)
    y_bin = np.asarray(y_bin, dtype=int)
    if len(x_bin) != len(y_bin) or len(x_bin) == 0:
        return 0.0
    joint = np.zeros((n_x, n_y), dtype=float)
    for xb, yb in zip(x_bin, y_bin):
        if 0 <= xb < n_x and 0 <= yb < n_y:
            joint[xb, yb] += 1.0
    pxy = joint / (np.sum(joint) + 1e-15)
    px = np.sum(pxy, axis=1, keepdims=True)
    py = np.sum(pxy, axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = pxy / (px @ py + 1e-15)
        mi = np.nansum(pxy * np.log(ratio + 1e-15))
    return float(max(mi, 0.0))


def quantile_bin(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-frequency binning to avoid empty bins."""
    x = np.asarray(x, dtype=float)
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(x, qs)
    # ensure strictly increasing
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-12
    # digitize (n_bins categories)
    b = np.digitize(x, edges[1:-1], right=False)
    b = np.clip(b, 0, n_bins - 1)
    return b.astype(int)


def compute_metrics_one_trial(
    phases: np.ndarray,
    dt: float,
    rng: np.random.Generator,
    mcfg: MetricConfig,
) -> Dict[str, float]:
    """
    Compute metrics on a stable segment given phase trajectories.
    Implements leave-one-out macro predictors and time-shift surrogate baselines.
    """
    phases = np.asarray(phases, dtype=float)
    T, N = phases.shape
    fs = 1.0 / dt

    expiph = np.exp(1j * phases)                # (T,N)
    sumZ = np.sum(expiph, axis=1)               # (T,)
    Z_all = sumZ / N
    R_all = np.abs(Z_all)

    # Mechanism signals
    # Micro: phase velocity estimated from unwrapped phases
    ph_unwrap = np.unwrap(phases, axis=0)
    dphi = np.diff(ph_unwrap, axis=0) / dt      # (T-1,N)
    vel_spread = np.std(dphi, axis=1)           # (T-1,)
    R_aligned = R_all[:-1]

    # Timescales (FFT-based autocorr integral)
    tau_macro = integral_timescale(R_all - np.mean(R_all), dt, max_lag_s=mcfg.max_lag_s)

    # Micro timescale: mean over sampled oscillators using velocity fluctuations
    sample_idx = rng.choice(N, size=min(mcfg.n_micro_samples, N), replace=False)
    tau_micro_list = []
    for i in sample_idx:
        v = dphi[:, i]
        v = v - np.mean(v)
        if np.std(v) < 1e-9:
            continue
        tau_micro_list.append(integral_timescale(v, dt, max_lag_s=mcfg.max_lag_s))
    tau_micro = float(np.mean(tau_micro_list)) if len(tau_micro_list) > 0 else dt
    scale_sep = float(tau_macro / (tau_micro + 1e-12))

    # R variability
    R_cv = float(np.std(R_all) / (np.mean(R_all) + 1e-12))

    # Effective coupling proxy (R vs velocity spread)
    eff_corr = 0.0
    if np.std(R_aligned) > 1e-9 and np.std(vel_spread) > 1e-9:
        eff_corr = float(np.corrcoef(R_aligned, vel_spread)[0, 1])
    eff_coupling = abs(eff_corr)
    # Surrogate: shift vel_spread relative to R
    shift_vs = random_circular_shift(rng, len(vel_spread), min_frac=mcfg.surrogate_min_shift_frac)
    eff_coupling_surr = abs(float(np.corrcoef(R_aligned, np.roll(vel_spread, shift_vs))[0, 1]))

    # Cross-scale micro↔macro metrics with leave-one-out macro predictor
    corr_list, corr_surr_list = [], []
    coh_list, coh_surr_list = [], []
    mi_list, mi_surr_list = [], []
    pac_list, pac_surr_list = [], []

    # Use separate sampling for PAC if desired
    pac_idx = rng.choice(N, size=min(mcfg.n_pac_samples, N), replace=False)

    # Precompute for MI binning (macro changes per i, but bins can be per i)
    for i in sample_idx:
        # Leave-one-out macro
        Z_minus = (sumZ - expiph[:, i]) / (N - 1)
        R_minus = np.abs(Z_minus)
        micro = np.cos(phases[:, i])

        # Surrogate: circular shift micro relative to macro
        shift = random_circular_shift(rng, len(micro), min_frac=mcfg.surrogate_min_shift_frac)
        micro_s = np.roll(micro, shift)

        # 1) Micro–macro correlation (NOT called PAC)
        if np.std(R_minus) > 1e-9 and np.std(micro) > 1e-9:
            corr = abs(float(np.corrcoef(R_minus, micro)[0, 1]))
            corr_s = abs(float(np.corrcoef(R_minus, micro_s)[0, 1]))
        else:
            corr, corr_s = 0.0, 0.0
        corr_list.append(corr)
        corr_surr_list.append(corr_s)

        # 2) Spectral coherence (average in band)
        try:
            f, cxy = coherence(R_minus, micro, fs=fs, nperseg=min(512, len(micro)))
            mask = (f >= mcfg.coh_f_lo) & (f <= mcfg.coh_f_hi)
            coh = float(np.mean(cxy[mask])) if np.any(mask) else 0.0

            f2, cxy2 = coherence(R_minus, micro_s, fs=fs, nperseg=min(512, len(micro)))
            mask2 = (f2 >= mcfg.coh_f_lo) & (f2 <= mcfg.coh_f_hi)
            coh_s = float(np.mean(cxy2[mask2])) if np.any(mask2) else 0.0
        except Exception:
            coh, coh_s = 0.0, 0.0
        coh_list.append(coh)
        coh_surr_list.append(coh_s)

        # 3) Mutual information (quantile bins)
        try:
            xb = quantile_bin(R_minus, mcfg.mi_bins)
            yb = quantile_bin(micro, mcfg.mi_bins)
            yb_s = quantile_bin(micro_s, mcfg.mi_bins)
            mi = mutual_information_discrete(xb, yb, mcfg.mi_bins, mcfg.mi_bins)
            mi_s = mutual_information_discrete(xb, yb_s, mcfg.mi_bins, mcfg.mi_bins)
        except Exception:
            mi, mi_s = 0.0, 0.0
        mi_list.append(mi)
        mi_surr_list.append(mi_s)

    # Proper PAC (Tort MI) computed on separate subset for speed
    for i in pac_idx:
        Z_minus = (sumZ - expiph[:, i]) / (N - 1)
        R_minus = np.abs(Z_minus)
        micro = np.cos(phases[:, i])

        # Low-frequency phase from macro R
        R_low = bandpass_filtfilt(R_minus, fs, mcfg.pac_low_lo, mcfg.pac_low_hi, order=4)
        phase_low = np.angle(hilbert(R_low))

        # High-frequency amplitude envelope from micro
        micro_high = bandpass_filtfilt(micro, fs, mcfg.pac_high_lo, mcfg.pac_high_hi, order=4)
        amp_high = np.abs(hilbert(micro_high))

        pac = tort_modulation_index(phase_low, amp_high, n_bins=mcfg.pac_bins)

        # Surrogate: shift amplitude envelope relative to phase
        shift = random_circular_shift(rng, len(amp_high), min_frac=mcfg.surrogate_min_shift_frac)
        pac_s = tort_modulation_index(phase_low, np.roll(amp_high, shift), n_bins=mcfg.pac_bins)

        pac_list.append(pac)
        pac_surr_list.append(pac_s)

    # Aggregate
    corr_mean = float(np.mean(corr_list)) if len(corr_list) else 0.0
    coh_mean = float(np.mean(coh_list)) if len(coh_list) else 0.0
    mi_mean = float(np.mean(mi_list)) if len(mi_list) else 0.0
    pac_mean = float(np.mean(pac_list)) if len(pac_list) else 0.0

    corr_surr = float(np.mean(corr_surr_list)) if len(corr_surr_list) else 0.0
    coh_surr = float(np.mean(coh_surr_list)) if len(coh_surr_list) else 0.0
    mi_surr = float(np.mean(mi_surr_list)) if len(mi_surr_list) else 0.0
    pac_surr = float(np.mean(pac_surr_list)) if len(pac_surr_list) else 0.0

    return {
        "mean_R": float(np.mean(R_all)),
        "R_cv": R_cv,
        "tau_macro": float(tau_macro),
        "tau_micro": float(tau_micro),
        "scale_sep": scale_sep,

        "mmcorr": corr_mean,
        "mmcorr_surr": corr_surr,

        "spec_coh": coh_mean,
        "spec_coh_surr": coh_surr,

        "mi": mi_mean,
        "mi_surr": mi_surr,

        "pac": pac_mean,
        "pac_surr": pac_surr,

        "eff_coupling": float(eff_coupling),
        "eff_coupling_surr": float(eff_coupling_surr),
    }


# -----------------------------
# Main sweep + plotting
# -----------------------------

def default_K_values() -> np.ndarray:
    # Denser around expected transition/peak
    return np.array([0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2, 2.8, 3.5, 4.5, 6.0, 8.0, 10.0, 15.0], dtype=float)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TRE Simulation 3 (Revised): Inverted-U Cross-Scale Coherence with Controls"
    )
    p.add_argument("--N", type=int, default=DEFAULT_N, help="number of oscillators")
    p.add_argument("--T", type=float, default=DEFAULT_T, help="total simulated time (seconds)")
    p.add_argument("--dt", type=float, default=DEFAULT_DT, help="integration time step (seconds)")
    p.add_argument("--noise", type=float, default=DEFAULT_NOISE, help="noise standard deviation")
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS, help="independent trials per K (for CI bands)")
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY, help="macro delay τ_d in seconds (0 disables)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="base RNG seed")
    p.add_argument("--out", type=str, default=DEFAULT_OUT, help="output PNG filename")
    p.add_argument(
        "--csv",
        type=str,
        default=DEFAULT_CSV,
        help="output CSV filename (default: derived from --out by replacing extension with .csv)",
    )
    return p.parse_args()



def write_results_csv(
    csv_path: str,
    Ks: np.ndarray,
    peak_idx: int,
    K_peak: float,
    sim_cfg: SimConfig,
    mcfg: MetricConfig,
    trials: int,
    seed: int,
    ci_level: float,
    stats: Dict[str, Dict[str, np.ndarray]],
    debiased: Dict[str, Dict[str, np.ndarray]],
    composite_s: Dict[str, np.ndarray],
) -> None:
    """
    Write a per-K summary CSV.

    The CSV includes:
      - Run configuration (repeated on each row for convenience)
      - Mean ± CI for core signals (R, timescales, separation)
      - Mean ± CI for raw cross-scale metrics and their surrogate baselines
      - Mean ± CI for debiased effects (real − surrogate, clipped ≥ 0)
      - Mean ± CI for the composite coherence index
    """
    def _put_stat(row: Dict[str, float], prefix: str, st: Dict[str, np.ndarray], i: int) -> None:
        row[f"{prefix}_mean"] = float(st["mean"][i])
        row[f"{prefix}_lo"] = float(st["lo"][i])
        row[f"{prefix}_hi"] = float(st["hi"][i])

    # Column order
    fieldnames: List[str] = [
        # Sweep
        "K", "is_peak", "K_peak",

        # Run configuration (repeated)
        "N", "T", "dt", "noise_std", "macro_delay_s", "burn_in_frac",
        "trials", "seed", "ci_level",

        # Metric configuration (repeated)
        "n_micro_samples", "n_pac_samples", "surrogate_min_shift_frac",
        "coh_f_lo", "coh_f_hi",
        "pac_low_lo", "pac_low_hi", "pac_high_lo", "pac_high_hi", "pac_bins",
        "mi_bins", "max_lag_s",
    ]

    # Stats to emit (triples mean/lo/hi)
    stat_prefixes = [
        # Core signals
        "mean_R", "R_cv", "tau_macro", "tau_micro", "scale_sep",

        # Raw metrics + surrogates
        "mmcorr", "mmcorr_surr",
        "spec_coh", "spec_coh_surr",
        "mi", "mi_surr",
        "pac", "pac_surr",
        "eff_coupling", "eff_coupling_surr",

        # Debiased effects (real − surrogate, clipped ≥ 0)
        "mmcorr_debiased",
        "spec_coh_debiased",
        "mi_debiased",
        "pac_debiased",
        "eff_coupling_debiased",

        # Composite
        "composite",
    ]

    for pref in stat_prefixes:
        fieldnames.extend([f"{pref}_mean", f"{pref}_lo", f"{pref}_hi"])

    # Map prefixes to stats dicts
    pref_to_stat: Dict[str, Dict[str, np.ndarray]] = {
        "mean_R": stats["mean_R"],
        "R_cv": stats["R_cv"],
        "tau_macro": stats["tau_macro"],
        "tau_micro": stats["tau_micro"],
        "scale_sep": stats["scale_sep"],

        "mmcorr": stats["mmcorr"],
        "mmcorr_surr": stats["mmcorr_surr"],
        "spec_coh": stats["spec_coh"],
        "spec_coh_surr": stats["spec_coh_surr"],
        "mi": stats["mi"],
        "mi_surr": stats["mi_surr"],
        "pac": stats["pac"],
        "pac_surr": stats["pac_surr"],
        "eff_coupling": stats["eff_coupling"],
        "eff_coupling_surr": stats["eff_coupling_surr"],

        "mmcorr_debiased": debiased["mmcorr_debiased"],
        "spec_coh_debiased": debiased["spec_coh_debiased"],
        "mi_debiased": debiased["mi_debiased"],
        "pac_debiased": debiased["pac_debiased"],
        "eff_coupling_debiased": debiased["eff_coupling_debiased"],

        "composite": composite_s,
    }

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, K in enumerate(Ks):
            row: Dict[str, float] = {}

            # Sweep + peak
            row["K"] = float(K)
            row["is_peak"] = 1.0 if i == peak_idx else 0.0
            row["K_peak"] = float(K_peak)

            # Run config
            row["N"] = float(sim_cfg.N)
            row["T"] = float(sim_cfg.T)
            row["dt"] = float(sim_cfg.dt)
            row["noise_std"] = float(sim_cfg.noise_std)
            row["macro_delay_s"] = float(sim_cfg.macro_delay)
            row["burn_in_frac"] = float(sim_cfg.burn_in)
            row["trials"] = float(trials)
            row["seed"] = float(seed)
            row["ci_level"] = float(ci_level)

            # Metric config
            row["n_micro_samples"] = float(mcfg.n_micro_samples)
            row["n_pac_samples"] = float(mcfg.n_pac_samples)
            row["surrogate_min_shift_frac"] = float(mcfg.surrogate_min_shift_frac)
            row["coh_f_lo"] = float(mcfg.coh_f_lo)
            row["coh_f_hi"] = float(mcfg.coh_f_hi)
            row["pac_low_lo"] = float(mcfg.pac_low_lo)
            row["pac_low_hi"] = float(mcfg.pac_low_hi)
            row["pac_high_lo"] = float(mcfg.pac_high_lo)
            row["pac_high_hi"] = float(mcfg.pac_high_hi)
            row["pac_bins"] = float(mcfg.pac_bins)
            row["mi_bins"] = float(mcfg.mi_bins)
            row["max_lag_s"] = float(mcfg.max_lag_s)

            # Stats
            for pref in stat_prefixes:
                _put_stat(row, pref, pref_to_stat[pref], i)

            writer.writerow(row)


def main() -> None:
    args = parse_args()

    Ks = default_K_values()

    sim_base = SimConfig(
        N=args.N,
        T=args.T,
        dt=args.dt,
        noise_std=args.noise,
        macro_delay=args.delay,
        burn_in=0.2,
    )
    mcfg = MetricConfig()

    # Storage: metric -> [len(K), trials]
    metric_names = [
        "mean_R", "R_cv", "tau_macro", "tau_micro", "scale_sep",
        "mmcorr", "mmcorr_surr",
        "spec_coh", "spec_coh_surr",
        "mi", "mi_surr",
        "pac", "pac_surr",
        "eff_coupling", "eff_coupling_surr",
    ]
    data: Dict[str, List[List[float]]] = {k: [] for k in metric_names}

    print("=" * 78)
    print("TRE Simulation 3 (Revised): Inverted-U Cross-Scale Coherence with Controls")
    print("=" * 78)
    print(f"N={sim_base.N}, T={sim_base.T}s, dt={sim_base.dt}s, noise={sim_base.noise_std}, "
          f"macro_delay={sim_base.macro_delay}s, trials={args.trials}")
    print(f"K sweep: {Ks.tolist()}")
    print("")

    for ki, K in enumerate(Ks):
        cfg = SimConfig(**{**sim_base.__dict__, "K": float(K)})
        trial_metrics = {name: [] for name in metric_names}

        for tr in range(args.trials):
            seed = args.seed + 1000 * ki + tr
            rng = np.random.default_rng(seed + 12345)

            sim = run_kuramoto(cfg, seed=seed)
            phases = sim["phases"]

            # Stable segment
            start = int(round(cfg.burn_in * phases.shape[0]))
            phases_stable = phases[start:, :]

            metrics = compute_metrics_one_trial(phases_stable, cfg.dt, rng=rng, mcfg=mcfg)

            for name in metric_names:
                trial_metrics[name].append(metrics[name])

        # Commit per-K arrays
        for name in metric_names:
            data[name].append(trial_metrics[name])

        # Progress print
        mR = np.mean(trial_metrics["mean_R"])
        sep = np.mean(trial_metrics["scale_sep"])
        mmc = np.mean(trial_metrics["mmcorr"])
        mmc_s = np.mean(trial_metrics["mmcorr_surr"])
        print(f"[{ki+1:02d}/{len(Ks)}] K={K:>4.1f}  "
              f"R={mR:0.3f}  Sep={sep:0.1f}  "
              f"MMC={mmc:0.3f} (surr {mmc_s:0.3f})")

    # Convert data to arrays: (K, trials)
    arr: Dict[str, np.ndarray] = {k: np.asarray(v, dtype=float) for k, v in data.items()}

    # Compute mean + CI for plotting
    stats = {}
    for name, mat in arr.items():
        means, los, his = [], [], []
        for k_i in range(mat.shape[0]):
            m, lo, hi = mean_ci(mat[k_i, :], ci=0.95)
            means.append(m); los.append(lo); his.append(hi)
        stats[name] = {"mean": np.array(means), "lo": np.array(los), "hi": np.array(his)}

    
    # ------------------------------------------------------------------
    # Debiased (real - surrogate) versions for cross-scale metrics
    #   We compute these PER TRIAL (then aggregate), rather than subtracting
    #   the mean surrogate from the mean real, to better reflect uncertainty.
    # ------------------------------------------------------------------
    mmc_diff = np.maximum(arr["mmcorr"] - arr["mmcorr_surr"], 0.0)               # (K, trials)
    coh_diff = np.maximum(arr["spec_coh"] - arr["spec_coh_surr"], 0.0)
    mi_diff  = np.maximum(arr["mi"] - arr["mi_surr"], 0.0)
    pac_diff = np.maximum(arr["pac"] - arr["pac_surr"], 0.0)
    effc_diff = np.maximum(arr["eff_coupling"] - arr["eff_coupling_surr"], 0.0)

    def _row_stats(mat: np.ndarray) -> Dict[str, np.ndarray]:
        means, los, his = [], [], []
        for k_i in range(mat.shape[0]):
            m, lo, hi = mean_ci(mat[k_i, :], ci=0.95)
            means.append(m); los.append(lo); his.append(hi)
        return {"mean": np.array(means), "lo": np.array(los), "hi": np.array(his)}

    mmc_eff_s = _row_stats(mmc_diff)
    coh_eff_s = _row_stats(coh_diff)
    mi_eff_s  = _row_stats(mi_diff)
    pac_eff_s = _row_stats(pac_diff)
    effc_eff_s = _row_stats(effc_diff)

    # Composite index WITHOUT "intermediate-is-best" weighting:
    # Normalize each debiased metric globally (across all K and trials), then average.
    mmc_norm  = normalize01(mmc_diff)
    coh_norm  = normalize01(coh_diff)
    mi_norm   = normalize01(mi_diff)
    pac_norm  = normalize01(pac_diff)
    effc_norm = normalize01(effc_diff)

    composite_mat = (mmc_norm + coh_norm + mi_norm + pac_norm + effc_norm) / 5.0  # (K, trials)
    composite_s = _row_stats(composite_mat)

    # Peak K for composite (based on mean)
    peak_idx = int(np.nanargmax(composite_s["mean"]))
    K_peak = float(Ks[peak_idx])


    # -----------------------------
    # CSV output (per-K summary)
    # -----------------------------
    csv_path = args.csv
    if csv_path is None or str(csv_path).strip() == "":
        base, _ = os.path.splitext(args.out)
        csv_path = base + ".csv"

    debiased = {
        "mmcorr_debiased": mmc_eff_s,
        "spec_coh_debiased": coh_eff_s,
        "mi_debiased": mi_eff_s,
        "pac_debiased": pac_eff_s,
        "eff_coupling_debiased": effc_eff_s,
    }

    # Write CSV results before plotting (so results persist even if plotting fails).
    write_results_csv(
        csv_path=csv_path,
        Ks=Ks,
        peak_idx=peak_idx,
        K_peak=K_peak,
        sim_cfg=sim_base,
        mcfg=mcfg,
        trials=args.trials,
        seed=args.seed,
        ci_level=0.95,
        stats=stats,
        debiased=debiased,
        composite_s=composite_s,
    )



# -------------- Plot --------------
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.0))
    fig.suptitle(
        "TRE Signature #4: Non-monotonic Cross-Scale Coherence (with Controls)\n"
        "Leave-one-out macro predictors, multi-trial mean±95% CI, and time-shift surrogates",
        fontsize=13,
    )

    # Panel A: mean R and scale separation
    ax = axes[0, 0]
    ax2 = ax.twinx()
    ax.plot(Ks, stats["mean_R"]["mean"], marker="o", linewidth=2, label="Mean R")
    ax.fill_between(Ks,
                    np.clip(stats["mean_R"]["lo"], 0.0, 1.05),
                    np.clip(stats["mean_R"]["hi"], 0.0, 1.05),
                    alpha=0.2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Coupling K")
    ax.set_ylabel("Order parameter R")

    ax2.plot(Ks, stats["scale_sep"]["mean"], marker="s", linestyle="--", linewidth=2, label="τ_macro/τ_micro")
    ax2.fill_between(Ks,
                     np.maximum(stats["scale_sep"]["lo"], 0.0),
                     np.maximum(stats["scale_sep"]["hi"], 0.0),
                     alpha=0.15)
    ax2.set_ylabel("Scale separation (τ_macro / τ_micro)")

    ax.axvline(K_peak, linestyle=":", linewidth=2)
    ax.set_title("A. Synchronization and Scale Separation vs K")
    ax.grid(alpha=0.3)

    # combined legend (skip artists with default "_" labels)
    lines = [l for l in (ax.get_lines() + ax2.get_lines()) if not l.get_label().startswith("_")]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="best", fontsize=9)

    # Panel B: Cross-scale metrics (real vs surrogate)
    ax = axes[0, 1]
    # For readability, plot debiased effect and (optional) surrogate baseline
    ax.plot(Ks, mmc_eff_s["mean"], marker="o", linewidth=2, label="Micro–macro corr (debiased)")
    ax.fill_between(Ks,
                    np.clip(mmc_eff_s["lo"], 0.0, 1.0),
                    np.clip(mmc_eff_s["hi"], 0.0, 1.0),
                    alpha=0.15)
    ax.plot(Ks, coh_eff_s["mean"], marker="s", linewidth=2, label="Spectral coherence (debiased)")
    ax.fill_between(Ks,
                    np.clip(coh_eff_s["lo"], 0.0, 1.0),
                    np.clip(coh_eff_s["hi"], 0.0, 1.0),
                    alpha=0.15)
    ax.plot(Ks, mi_eff_s["mean"], marker="^", linewidth=2, label="Mutual information (debiased)")
    ax.fill_between(Ks,
                    np.maximum(mi_eff_s["lo"], 0.0),
                    np.maximum(mi_eff_s["hi"], 0.0),
                    alpha=0.15)
    ax.plot(Ks, pac_eff_s["mean"], marker="d", linewidth=2, label="PAC (Tort MI, debiased)")
    ax.fill_between(Ks,
                    np.clip(pac_eff_s["lo"], 0.0, 1.0),
                    np.clip(pac_eff_s["hi"], 0.0, 1.0),
                    alpha=0.15)
    ax.plot(Ks, effc_eff_s["mean"], marker="x", linewidth=2, label="Effective coupling proxy (debiased)")
    ax.fill_between(Ks,
                    np.clip(effc_eff_s["lo"], 0.0, 1.0),
                    np.clip(effc_eff_s["hi"], 0.0, 1.0),
                    alpha=0.15)
    ax.axvline(K_peak, linestyle=":", linewidth=2, label=f"Composite peak K={K_peak:.1f}")
    ax.set_xlabel("Coupling K")
    ax.set_ylabel("Metric value (real − surrogate, clipped ≥ 0)")
    ax.set_ylim(bottom=0.0)
    ax.set_title("B. Debiased Cross-Scale Coherence Metrics")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # Panel C: Composite coherence (no TRE-weighting)
    ax = axes[1, 0]
    ax.plot(Ks, composite_s["mean"], marker="o", linewidth=3, label="Composite coherence (unweighted)")
    ax.fill_between(Ks,
                    np.clip(composite_s["lo"], 0.0, 1.0),
                    np.clip(composite_s["hi"], 0.0, 1.0),
                    alpha=0.25)
    ax.axvline(K_peak, linestyle="--", linewidth=2, label=f"Peak K={K_peak:.1f}")
    ax.set_xlabel("Coupling K")
    ax.set_ylabel("Composite index (0–1)")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("C. Inverted-U Composite (No 'Intermediate' Weighting)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    # Panel D: Mechanism summary signals
    ax = axes[1, 1]
    ax.plot(Ks, stats["R_cv"]["mean"], marker="o", linewidth=2, label="R variability (CV)")
    ax.fill_between(Ks,
                    np.maximum(stats["R_cv"]["lo"], 0.0),
                    np.maximum(stats["R_cv"]["hi"], 0.0),
                    alpha=0.2)

    # normalized timescales
    tauM = stats["tau_macro"]["mean"]
    taum = stats["tau_micro"]["mean"]
    ax.plot(Ks, tauM / (np.nanmax(tauM) + 1e-12), marker="s", linestyle="--", linewidth=2, label="τ_macro (norm)")
    ax.plot(Ks, taum / (np.nanmax(taum) + 1e-12), marker="^", linestyle="--", linewidth=2, label="τ_micro (norm)")

    ax.axvline(K_peak, linestyle=":", linewidth=2)
    ax.set_xlabel("Coupling K")
    ax.set_ylabel("Value")
    ax.set_title("D. Mechanism: Variability and Timescales")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    # Annotate low/peak/high regimes
    ax.text(
        0.02, 0.98,
        "Interpretation:\n"
        "Low K: weak coupling → low cross-scale coherence\n"
        "Mid K: partial sync + preserved separation → peak coherence\n"
        "High K: tight locking → R fluctuations collapse → coherence drops",
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 78)
    print("Summary")
    print("=" * 78)
    print(f"Composite coherence peak at K={K_peak:.1f}")
    print(f"Saved: {args.out}")
    print(f"Saved: {csv_path}")
    print("Note: Cross-scale metrics are shown as (real − time-shift surrogate), clipped at 0.\n")

if __name__ == "__main__":
    main()
