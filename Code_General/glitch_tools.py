"""Controlled glitches for the study of notebook 21 (extensions 55, 56).

Three families of instrumental transients, defined directly in whitened units (1024 samples at 1024 Hz) and injected in the
frequency bins exactly like the signals of ``gwosc_tools`` (scaled to a given norm, which plays the role of the optimal SNR):

* ``blip``    one-sided (sharp onset, exponential decay) sinusoid, f = 80-300 Hz, decay 3-12 ms;
* ``whistle`` fast frequency sweep, f0 = 40-150 Hz, df/dt = 1500-5000 Hz/s (far above the linear-chirp prior of the project), 0.1-0.25 s;
* ``arch``    low-frequency arch (scattered-light-like): 28-55 Hz carrier with a second harmonic, sin^2 envelope of 0.3-0.6 s.

``GlitchBank`` is a phase-free matched-filter bank of such shapes (same ``scan`` as ``SGBank``), used as a glitch model."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gwosc_tools as g

T = (np.arange(g.NWIN) - g.NWIN // 2) / g.FS          # time axis centred on the event


def shape(kind, p):
    """Waveform on the centred time axis, event onset at t = 0."""
    if kind == "blip":
        w = np.where(T >= 0, np.exp(-np.clip(T, 0, None) / p["tau"]) * (1 - np.exp(-np.clip(T, 0, None) / 5e-4)), 0.0) * np.cos(2 * np.pi * p["f"] * T)
    elif kind == "whistle":
        D = p["dur"]; u = np.clip(T, 0, D); env = np.where((T >= 0) & (T <= D), np.sin(np.pi * u / D) ** 2, 0.0)
        w = env * np.cos(2 * np.pi * (p["f"] * T + 0.5 * p["fdot"] * T ** 2))
    elif kind == "arch":
        D = p["dur"]; u = np.clip(T, 0, D); env = np.where((T >= 0) & (T <= D), np.sin(np.pi * u / D) ** 2, 0.0)
        ph = 2 * np.pi * (p["f"] * T + 0.5 * p["fdot"] * T ** 2); w = env * (np.cos(ph) + 0.4 * np.cos(2 * ph))
    else:
        raise ValueError(kind)
    return w


def spectrum(kind, p):
    """Band-limited bins of the shape with its onset at t = 0 (circular), unit-free."""
    return g.to_bins(np.fft.ifftshift(shape(kind, p))[None])[0]


def draw(kind, rng, n):
    if kind == "blip":
        return [dict(f=rng.uniform(80, 300), tau=rng.uniform(3e-3, 12e-3)) for _ in range(n)]
    if kind == "whistle":
        return [dict(f=rng.uniform(40, 150), fdot=rng.uniform(1500, 5000), dur=rng.uniform(0.10, 0.25)) for _ in range(n)]
    return [dict(f=rng.uniform(28, 55), fdot=rng.uniform(-120, 120), dur=rng.uniform(0.3, 0.6)) for _ in range(n)]


def inject_glitch(v_noise, kind, norm, rng):
    """Add glitches of the given family scaled to ``norm`` (norm of the whitened band bins) at random onset times in 0.25-0.75 s."""
    n = len(v_noise); W = np.empty((n, g.NB), complex); norm = np.broadcast_to(np.asarray(norm, float), (n,))
    for i, p in enumerate(draw(kind, rng, n)):
        w = spectrum(kind, p) * np.exp(-2j * np.pi * g.FREQ * rng.uniform(0.25, 0.75)); W[i] = w * norm[i] / np.sqrt(np.sum(np.abs(w) ** 2))
    return g.to_series(g.to_bins(v_noise) + W)


class GlitchBank(g.SGBank):
    """Phase-free bank of blip / whistle / arch templates (same ``scan`` as the sine-Gaussian bank; f0 label = carrier, Q label = family index)."""

    def __init__(self, band=g.BAND):
        P = []
        for f in (80, 120, 170, 230, 300):
            for tau in (3e-3, 6e-3, 12e-3): P.append(("blip", dict(f=f, tau=tau), 0))
        for f in (40, 80, 120):
            for fdot in (1500, 3000, 5000):
                for dur in (0.12, 0.2): P.append(("whistle", dict(f=f, fdot=fdot, dur=dur), 1))
        for f in (30, 40, 50):
            for fdot in (-100, 0, 100):
                for dur in (0.35, 0.55): P.append(("arch", dict(f=f, fdot=fdot, dur=dur), 2))
        self.grid = [(p["f"], c) for _, p, c in P]; self.f0 = np.array([a for a, _ in self.grid], float); self.Q = np.array([c for _, c in self.grid], float)
        T_ = np.array([2.0 * spectrum(k, p) for k, p, _ in P]) * band           # analytic-signal convention of SGBank (factor 2 on the positive frequencies)
        self.norm2 = np.sum(np.abs(T_) ** 2, axis=1); self.T = (T_ / np.sqrt(self.norm2)[:, None]).astype(np.complex64)
