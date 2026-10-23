"""Binary-black-hole (IMRPhenomD) signals for the one-second analysis windows of the project.

The phenomenological bursts of notebooks 02-07 are not astrophysical waveforms. This module builds a library of
inspiral-merger-ringdown signals with LALSimulation (IMRPhenomD, aligned spins), as the strain seen by a single
interferometer for a random sky position, polarisation and inclination, and stores them in the frequency bins of a 1 s
window (1 Hz spacing, 0..512 Hz) with the merger at a random time of the window. Signals are injected like every other
morphology (``gwosc_tools.inject(v, "imr", snr, ...)``): the library is scaled to the requested optimal SNR, so the
distance of the source is irrelevant and only the *shape* matters.

Needs `lalsuite` only to BUILD the libraries (python imr_tools.py); using them needs just the cached files cache/imr_library.npz (kind "imr", for training) and cache/imr_library_test.npz (kind "imr_test", waveforms never seen in training: use it to evaluate).

Prior of the library (detector-frame masses): total mass U(30, 100) Msun, mass ratio q = m2/m1 U(0.25, 1), aligned spins
chi_{1,2} U(-0.5, 0.5), cos(inclination) U(-1, 1), isotropic sky position and uniform polarisation angle; merger time
U(0.55, 0.88) s in the window (an extra shift of +-30 ms is applied at injection). The signal is generated from 20 Hz and
cropped to the window: for the lightest systems the early inspiral (below ~35 Hz) falls before the window and is cut
(with a 40 ms Hann ramp).
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gwosc_tools as g

FS_GEN, T_GEN, FMIN = 2048, 4.0, 20.0
MSUN_S = 4.925491e-6                       # G Msun / c^3 [s]


def antenna(theta, phi, psi):
    """Antenna pattern F+, Fx of an L-shaped interferometer for a source at (theta, phi) in the detector frame."""
    c = 0.5 * (1 + np.cos(theta) ** 2)
    fp = c * np.cos(2 * phi) * np.cos(2 * psi) - np.cos(theta) * np.sin(2 * phi) * np.sin(2 * psi)
    fc = c * np.cos(2 * phi) * np.sin(2 * psi) + np.cos(theta) * np.sin(2 * phi) * np.cos(2 * psi)
    return fp, fc


def chirp_mass(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def one_waveform(m1, m2, chi1, chi2, iota, fp, fc, tc, phi_ref=0.0):
    """Window-bin strain (513 complex values, continuous-FT convention, per unit 1 Mpc) of one signal."""
    import lal
    import lalsimulation as ls
    hp, hc = ls.SimInspiralChooseFDWaveform(m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0, 0, chi1, 0, 0, chi2, 1e6 * lal.PC_SI, iota, phi_ref,
                                            0.0, 0.0, 0.0, 1.0 / T_GEN, FMIN, FS_GEN / 2, FMIN, None, ls.IMRPhenomD)
    H = fp * hp.data.data + fc * hc.data.data
    ht = np.fft.irfft(H, n=int(FS_GEN * T_GEN))
    ipk = int(np.argmax(np.abs(ht)))                      # merger ~ peak of the envelope; put it at tc
    ht = np.roll(ht, int(round(tc * FS_GEN)) - ipk)
    w = ht[:FS_GEN].copy()                                # the 1 s window
    ramp = int(0.04 * FS_GEN); w[:ramp] *= 0.5 * (1 - np.cos(np.pi * np.arange(ramp) / ramp))
    ramp = int(0.02 * FS_GEN); w[-ramp:] *= 0.5 * (1 + np.cos(np.pi * np.arange(ramp) / ramp))
    return (np.fft.rfft(w) / FS_GEN)[: g.NB], w


def build_library(n=4000, seed=1234, path=None):
    r = np.random.default_rng(seed)
    m_tot = r.uniform(30, 100, n); q = r.uniform(0.25, 1.0, n)
    m1 = m_tot / (1 + q); m2 = m_tot - m1
    chi1 = r.uniform(-0.5, 0.5, n); chi2 = r.uniform(-0.5, 0.5, n); cosi = r.uniform(-1, 1, n)
    theta = np.arccos(r.uniform(-1, 1, n)); phis = r.uniform(0, 2 * np.pi, n); psi = r.uniform(0, np.pi, n)
    fp, fc = antenna(theta, phis, psi); tc = r.uniform(0.55, 0.88, n)
    H = np.zeros((n, g.NB), np.complex64)
    for i in range(n):
        H[i] = one_waveform(m1[i], m2[i], chi1[i], chi2[i], np.arccos(cosi[i]), fp[i], fc[i], tc[i])[0]
    H = H / np.max(np.abs(H))                              # overall scale is irrelevant (SNR normalisation)
    out = dict(H=H, tc=tc, m1=m1, m2=m2, mtot=m_tot, q=q, mc=chirp_mass(m1, m2), chi1=chi1, chi2=chi2, cosi=cosi, fp=fp, fc=fc)
    path = path or os.path.join(g.CACHE, "imr_library.npz")
    os.makedirs(os.path.dirname(path), exist_ok=True); np.savez_compressed(path, **out); return out


class IMRBank(g.SGBank):
    """Matched-filter bank of IMRPhenomD templates (non-spinning, face-on, phase-free) on a (total mass, mass ratio) grid.
    Same ``scan`` as the sine-Gaussian bank; the merger of every template sits at t = 0, so the best lag tau is the merger time."""

    def __init__(self, S1, mtots=(30, 35, 42, 50, 60, 72, 85, 100), qs=(1.0, 0.6, 0.3), band=g.BAND):
        self.grid = [(m, q) for q in qs for m in mtots]
        self.f0 = np.array([m for m, q in self.grid], float); self.Q = np.array([q for m, q in self.grid], float)   # f0 -> M_tot, Q -> q
        T = []
        for m, q in self.grid:
            m1 = m / (1 + q); H, _ = one_waveform(m1, m - m1, 0.0, 0.0, 0.0, 1.0, 0.0, 0.75)     # face-on, F+ = 1: h+ only
            T.append(H * np.exp(2j * np.pi * g.FREQ * 0.75))       # circular shift of the 1 s window: the merger sits at t = 0
        T = np.array(T) / np.sqrt(np.atleast_2d(S1)) * band * 2.0
        self.norm2 = np.sum(np.abs(T) ** 2, axis=1)
        self.T = (T / np.sqrt(self.norm2)[:, None]).astype(np.complex64)


if __name__ == "__main__":
    for n, seed, name in ((4000, 1234, "imr_library.npz"), (2000, 4321, "imr_library_test.npz")):      # train and held-out test libraries (same prior, different draws)
        lib = build_library(n=n, seed=seed, path=os.path.join(g.CACHE, name))
        print(name, lib["H"].shape, " chirp mass %.1f-%.1f Msun, total mass %.0f-%.0f Msun" % (lib["mc"].min(), lib["mc"].max(), lib["mtot"].min(), lib["mtot"].max()))
