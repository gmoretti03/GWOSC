"""Goodness-of-fit and open-set scores for signals that belong to none of the hypotheses (notebook 10).

``FamilyBank`` is the bank of whitened templates of one morphology family (sine-Gaussian, Gaussian pulse, linear chirp) already
used in notebook 06, here with a ``fit`` method that returns the best template, the lag and the complex amplitude, so that the
best-fit signal can be subtracted and the *residual* tested. ``fit_best_family`` does it for the three families.
"""
import os
import sys
import numpy as np
from scipy import fft as sfft

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gwosc_tools as g


class FamilyBank:
    def __init__(self, kind, S1_, band=g.BAND):
        f = g.FREQ[None, :]; self.dof = 2; self.kind = kind
        if kind == "sg":
            f0, Q = np.meshgrid(np.geomspace(40, 450, 40), [3., 6., 10., 16.])
            T = g.sine_gaussian_ft(f, f0.ravel()[:, None], Q.ravel()[:, None], 0.0, positive_only=True)
        elif kind == "pulse":
            T = g.gaussian_pulse_ft(f, np.geomspace(0.8e-3, 5e-3, 14)[:, None]); self.dof = 1
        else:
            f0, fd, sg = np.meshgrid(np.linspace(50, 150, 6), np.linspace(150, 600, 5), np.linspace(.03, .08, 4))
            T = g._cgauss(f, f0.ravel()[:, None], fd.ravel()[:, None], sg.ravel()[:, None])
        T = T / np.sqrt(S1_)[None, :] * band
        self.T = (T / np.sqrt((np.abs(T) ** 2).sum(1, keepdims=True))).astype(np.complex64)

    def fit(self, X, tau=(0.15, 0.85), batch=48):
        """Best template, lag and complex amplitude of every window (X: complex bins (n, 513))."""
        i0, i1 = int(tau[0] * g.NWIN), int(tau[1] * g.NWIN); n = len(X)
        rho2 = np.zeros(n, np.float32); best = np.zeros(n, int); lag = np.zeros(n, int); z_best = np.zeros(n, np.complex64)
        for a in range(0, n, batch):
            xb = X[a:a + batch].astype(np.complex64)
            Z = np.zeros((len(xb), len(self.T), g.NWIN), np.complex64); Z[..., :g.NB] = xb[:, None, :] * np.conj(self.T)[None]
            z = sfft.ifft(Z, axis=-1, workers=-1)[..., i0:i1] * g.NWIN
            stat = (np.abs(z) ** 2 if self.dof == 2 else z.real ** 2).reshape(len(xb), -1)
            j = stat.argmax(1); it, il = np.unravel_index(j, (len(self.T), i1 - i0))
            rho2[a:a + batch] = stat[np.arange(len(xb)), j]; best[a:a + batch] = it; lag[a:a + batch] = il + i0
            zz = z[np.arange(len(xb)), it, il]; z_best[a:a + batch] = zz if self.dof == 2 else zz.real
        return dict(rho2=rho2, idx=best, tau=lag / g.NWIN, z=z_best)

    def model_bins(self, fit):
        """Best-fit signal in the whitened bins, (n, 513): a T e^{-2 pi i f tau}."""
        return fit["z"][:, None] * self.T[fit["idx"]] * np.exp(-2j * np.pi * g.FREQ[None, :] * fit["tau"][:, None])


def fit_best_family(X, banks, half_width=0.06):
    """Fit the three families, keep the one with the largest rho^2 and test its residual.
    Returns dict(family (0,1,2), rho2, tau, E_loc, E_full): the residual energy of the whitened series in +-half_width around the
    fitted lag (E_loc) and in the whole window (E_full). For noise E_loc ~ (number of samples) * 0.92."""
    fits = [b.fit(X) for b in banks]; r2 = np.stack([f["rho2"] for f in fits], 1); fam = r2.argmax(1); n = len(X)
    R = np.empty_like(X); tau = np.zeros(n); rho2 = np.zeros(n)
    for j, (b, f) in enumerate(zip(banks, fits)):
        m = fam == j
        if m.any():
            sub = {k: v[m] for k, v in f.items()}; R[m] = X[m] - b.model_bins(sub); tau[m] = f["tau"][m]; rho2[m] = f["rho2"][m]
    r = g.to_series(R).astype(np.float64); w = int(half_width * g.FS); c = (tau * g.FS).astype(int)
    idx = np.clip(c[:, None] + np.arange(-w, w + 1)[None, :], 0, g.NWIN - 1)
    return dict(family=fam, rho2=rho2, tau=tau, E_loc=(np.take_along_axis(r, idx, 1) ** 2).sum(1), E_full=(r ** 2).sum(1), n_loc=2 * w + 1)
