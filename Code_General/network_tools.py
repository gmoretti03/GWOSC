"""Two-detector helpers used by notebooks 31-32: complex matched-filter peaks of the IMR bank and physical network injections."""
import os, sys
import numpy as np
import lal
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gwosc_tools as g

I0, I1 = int(0.15 * g.NWIN), int(0.85 * g.NWIN)
DETS = {"H1": lal.CachedDetectors[lal.LHO_4K_DETECTOR], "L1": lal.CachedDetectors[lal.LLO_4K_DETECTOR]}


def peaks(bank, X, batch=64):
    """Per template: lag (sample index) and complex output of the maximum of |z| over the lag range."""
    nt = len(bank.T); tau = np.zeros((len(X), nt), np.int16); z = np.zeros((len(X), nt), np.complex64)
    for a in range(0, len(X), batch):
        xb = np.asarray(X[a:a + batch], np.complex64); Z = xb[:, None, :] * np.conj(bank.T)[None]; full = np.zeros(Z.shape[:2] + (g.NWIN,), np.complex64); full[..., :g.NB] = Z
        zz = np.fft.ifft(full, axis=-1)[..., I0:I1] * g.NWIN; j = np.abs(zz).argmax(-1); tau[a:a + batch] = j + I0; z[a:a + batch] = np.take_along_axis(zz, j[..., None], -1)[..., 0]
    return tau, z


def network_signal(Wv, S1, lib, j, norm, rng, glitch=None):
    """Physical injection of source ``j`` of ``lib`` (Hp, Hc) in both detectors. Wv: {det: noise windows (n, 1024)}, S1: {det: (n, 513)}. Returns {det: series}.
    The network optimal SNR is ``norm``; if ``glitch`` is a detector name, nothing physical is injected and a blip of norm ``norm`` is added to that detector only."""
    import glitch_tools as gt
    n = len(j); out = {}
    if glitch is not None:
        for d in Wv: out[d] = gt.inject_glitch(Wv[d], "blip", norm, rng) if d == glitch else Wv[d]
        return out
    ra = rng.uniform(0, 2 * np.pi, n); dec = np.arcsin(rng.uniform(-1, 1, n)); psi = rng.uniform(0, np.pi, n); gps = rng.uniform(1126051217, 1137254417, n); t0 = rng.uniform(-0.03, 0.03, n); Vd = {}
    for d in Wv:
        Vd[d] = np.zeros((n, g.NB), complex)
        for a in range(n):
            fp, fx = lal.ComputeDetAMResponse(DETS[d].response, ra[a], dec[a], psi[a], lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(gps[a]))); dt = lal.TimeDelayFromEarthCenter(DETS[d].location, ra[a], dec[a], lal.LIGOTimeGPS(gps[a]))
            Vd[d][a] = 2 * (fp * lib["Hp"][j[a]] + fx * lib["Hc"][j[a]]) * np.exp(-2j * np.pi * g.FREQ * (t0[a] + dt)) / np.sqrt(S1[d][a]) * g.BAND
    tot = np.sqrt(sum(np.sum(np.abs(Vd[d]) ** 2, axis=1) for d in Vd))
    for d in Wv: out[d] = g.to_series(g.to_bins(Wv[d]) + Vd[d] * (norm / tot)[:, None])
    return out
