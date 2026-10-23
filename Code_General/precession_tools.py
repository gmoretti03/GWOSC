"""Precessing and higher-mode binary-black-hole signals for notebook 29 (extension 10).

Same conventions as ``imr_tools`` (1 s window of 1024 Hz, merger placed at ``tc``, Hann ramps, single detector response with
antenna factors fp, fc) but with the approximants IMRPhenomPv2 (precessing spins) and IMRPhenomXPHM (precession + higher
modes) and spin vectors of magnitude up to 0.8 in random directions. Needs lalsuite (only to build): python precession_tools.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gwosc_tools as g
import imr_tools as it


def waveform(approx, m1, m2, s1, s2, iota, fp, fc, tc, phi_ref=0.0):
    import lal, lalsimulation as ls
    ap = {"Pv2": ls.IMRPhenomPv2, "XPHM": ls.IMRPhenomXPHM, "D": ls.IMRPhenomD}[approx]
    hp, hc = ls.SimInspiralChooseFDWaveform(m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, s1[0], s1[1], s1[2], s2[0], s2[1], s2[2], 1e6 * lal.PC_SI, iota, phi_ref,
                                            0.0, 0.0, 0.0, 1.0 / it.T_GEN, it.FMIN, it.FS_GEN / 2, it.FMIN, None, ap)
    H = fp * hp.data.data + fc * hc.data.data
    n = int(it.FS_GEN * it.T_GEN)
    if len(H) < n // 2 + 1: H = np.r_[H, np.zeros(n // 2 + 1 - len(H))]
    ht = np.fft.irfft(H[: n // 2 + 1], n=n); ipk = int(np.argmax(np.abs(ht))); ht = np.roll(ht, int(round(tc * it.FS_GEN)) - ipk)
    w = ht[:it.FS_GEN].copy(); r = int(0.04 * it.FS_GEN); w[:r] *= 0.5 * (1 - np.cos(np.pi * np.arange(r) / r)); r = int(0.02 * it.FS_GEN); w[-r:] *= 0.5 * (1 + np.cos(np.pi * np.arange(r) / r))
    return (np.fft.rfft(w) / it.FS_GEN)[: g.NB]


def build(approx, n=600, seed=77, chi_max=0.8, path=None):
    r = np.random.default_rng(seed)
    mt = r.uniform(30, 100, n); q = r.uniform(0.25, 1.0, n); m1 = mt / (1 + q); m2 = mt - m1
    def spin():
        c = r.uniform(-1, 1, n); ph = r.uniform(0, 2 * np.pi, n); a = chi_max * r.uniform(0, 1, n) ** (1 / 3) if approx != "D" else r.uniform(-0.5, 0.5, n)
        return (np.zeros(n), np.zeros(n), a) if approx == "D" else (a * np.sqrt(1 - c ** 2) * np.cos(ph), a * np.sqrt(1 - c ** 2) * np.sin(ph), a * c)
    s1, s2 = spin(), spin(); cosi = r.uniform(-1, 1, n); th = np.arccos(r.uniform(-1, 1, n)); phs = r.uniform(0, 2 * np.pi, n); psi = r.uniform(0, np.pi, n)
    fp, fc = it.antenna(th, phs, psi); tc = r.uniform(0.55, 0.88, n); H = np.zeros((n, g.NB), np.complex64); ok = np.ones(n, bool)
    for i in range(n):
        try: H[i] = waveform(approx, m1[i], m2[i], [s[i] for s in s1], [s[i] for s in s2], np.arccos(cosi[i]), fp[i], fc[i], tc[i])
        except Exception: ok[i] = False
    H = H / np.max(np.abs(H)); out = dict(H=H[ok], tc=tc[ok], mtot=mt[ok], q=q[ok], mc=it.chirp_mass(m1, m2)[ok], cosi=cosi[ok])
    path = path or os.path.join(g.CACHE, f"lib_{approx}.npz"); np.savez_compressed(path, **out); return out


if __name__ == "__main__":
    for a in ("D", "Pv2", "XPHM"):
        lib = build(a, n=600, seed=77); print(a, lib["H"].shape)


def waveform_pol(approx, m1, m2, s1, s2, iota, tc, phi_ref=0.0):
    """Window-bin spectra of the two polarisations (Hp, Hc), same placement of the merger (at ``tc``) for both; for network injections (notebook 31)."""
    import lal, lalsimulation as ls
    ap = {"Pv2": ls.IMRPhenomPv2, "XPHM": ls.IMRPhenomXPHM, "D": ls.IMRPhenomD}[approx]
    hp, hc = ls.SimInspiralChooseFDWaveform(m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, s1[0], s1[1], s1[2], s2[0], s2[1], s2[2], 1e6 * lal.PC_SI, iota, phi_ref,
                                            0.0, 0.0, 0.0, 1.0 / it.T_GEN, it.FMIN, it.FS_GEN / 2, it.FMIN, None, ap)
    n = int(it.FS_GEN * it.T_GEN); pad = lambda H: np.r_[H, np.zeros(max(0, n // 2 + 1 - len(H)))][: n // 2 + 1]
    Hp, Hc = pad(hp.data.data), pad(hc.data.data); ht = np.fft.irfft(Hp, n=n); shift = int(round(tc * it.FS_GEN)) - int(np.argmax(np.abs(ht))); out = []
    for H in (Hp, Hc):
        w = np.roll(np.fft.irfft(H, n=n), shift)[:it.FS_GEN].copy(); r = int(0.04 * it.FS_GEN); w[:r] *= 0.5 * (1 - np.cos(np.pi * np.arange(r) / r)); r = int(0.02 * it.FS_GEN); w[-r:] *= 0.5 * (1 + np.cos(np.pi * np.arange(r) / r))
        out.append((np.fft.rfft(w) / it.FS_GEN)[: g.NB])
    return out
