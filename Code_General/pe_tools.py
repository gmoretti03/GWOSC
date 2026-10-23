"""MCMC parameter estimation with the exact IMRPhenomD (aligned-spin) model for notebook 32 (extension 35).

For the dominant (2,2) mode a single detector sees Re[C h(f)] with a complex constant C (distance, inclination, polarisation, antenna pattern and
phase together), so the likelihood marginalised over C with a Gaussian prior of width sigma_rho is analytic and the free parameters are
(M_tot, q, chi1, chi2, tau): lnL = 0.5 rho^2 s^2/(1+s^2) - ln(1+s^2), rho^2 = |sum_k X_k conj(T_k) exp(2 pi i k tau)|^2 with T the unit-norm
phase-free analytic template of the model. ``run_pe`` is a top-level function so that it can be used with a process pool."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gwosc_tools as g
import precession_tools as pt

SIGMA = 10.0
PRIOR = dict(M=(30.0, 100.0), q=(0.25, 1.0), chi=(-0.5, 0.5), tau=(0.2, 0.8))


def template(M, q, chi1, chi2, S1):
    m1 = M / (1 + q); m2 = M - m1
    H = pt.waveform("D", m1, m2, [0, 0, chi1], [0, 0, chi2], 0.0, 1.0, 0.0, 0.75)
    T = 2.0 * H * np.exp(2j * np.pi * g.FREQ * 0.75) / np.sqrt(S1) * g.BAND
    return T / np.sqrt(np.sum(np.abs(T) ** 2))


def lnprob(theta, X, S1):
    M, q, c1, c2, tau = theta
    if not (PRIOR["M"][0] <= M <= PRIOR["M"][1] and PRIOR["q"][0] <= q <= PRIOR["q"][1] and PRIOR["chi"][0] <= c1 <= PRIOR["chi"][1] and PRIOR["chi"][0] <= c2 <= PRIOR["chi"][1] and PRIOR["tau"][0] <= tau <= PRIOR["tau"][1]):
        return -np.inf
    T = template(M, q, c1, c2, S1)
    z = np.sum(X * np.conj(T) * np.exp(2j * np.pi * g.FREQ * tau)); r2 = np.abs(z) ** 2; s2 = SIGMA ** 2
    return 0.5 * r2 * s2 / (1 + s2) - np.log1p(s2)


def run_pe(args):
    """args = (X, S1, truth, seed, nwalk, nstep, start): returns thinned chain (n, 5) and autocorrelation time."""
    import emcee
    X, S1, truth, seed, nwalk, nstep, start = args
    r = np.random.default_rng(seed); p0 = np.array(start) + r.normal(size=(nwalk, 5)) * np.array([3.0, 0.05, 0.08, 0.08, 0.01])
    p0[:, 0] = np.clip(p0[:, 0], 31, 99); p0[:, 1] = np.clip(p0[:, 1], 0.27, 0.99); p0[:, 2:4] = np.clip(p0[:, 2:4], -0.45, 0.45); p0[:, 4] = np.clip(p0[:, 4], 0.22, 0.78)
    sam = emcee.EnsembleSampler(nwalk, 5, lnprob, args=(X, S1)); sam.run_mcmc(p0, nstep, progress=False)
    chain = sam.get_chain(discard=nstep // 3, flat=True, thin=5)
    try: tau = float(np.mean(sam.get_autocorr_time(tol=0)))
    except Exception: tau = float("nan")
    return chain, tau
