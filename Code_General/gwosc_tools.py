"""Shared functions of the GWOSC burst-detection project (notebooks 01-05).

Conventions (everything lives in one 1 s analysis window, sampled at 1024 Hz):

* A raw 4096 s chunk (4096 Hz) is whitened with its own median-Welch PSD, band-passed to [30, 500] Hz and
  down-sampled to 1024 Hz. The whitened series ``v`` is scaled so that, in the band, the noise has unit
  variance per real degree of freedom.
* Frequency bins of a window: ``V_k = rfft(v)_k / sqrt(512)``, k = 0..512 (1 Hz spacing). For stationary Gaussian
  noise Re V_k and Im V_k are N(0, 1) in the band, hence E|V_k|^2 = 2.
* Inner product <a, b> = Re sum_k A_k conj(B_k) (band bins). The optimal SNR of a signal is
  rho^2 = <h, h> = 4 int |h(f)|^2 / S_n(f) df, computed with the PSD of the chunk.
"""
import hashlib
import json
import os
import numpy as np
from scipy.signal import welch
from scipy.signal.windows import tukey

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
RESULTS = os.path.join(HERE, "results")
FIGS = os.path.join(HERE, "figures")

FS_RAW = 4096
FS = 1024                      # sampling rate of the analysis windows
NWIN = 1024                    # samples per window (1 s)
NB = NWIN // 2 + 1             # frequency bins 0..512 Hz
FREQ = np.arange(NB, dtype=float)
FBAND = (30, 500)
BAND = (FREQ >= FBAND[0]) & (FREQ <= FBAND[1])
NBAND = int(BAND.sum())


def band_mask(lo, hi):
    return (FREQ >= lo) & (FREQ <= hi)

TRIM = 20                      # seconds removed at each edge of a chunk (filter/taper transients)

# time-ordered chunk indices (see results/chunks.json): interleaved over the whole run
SPLIT = {"train": [0, 2, 4, 6], "val": [1, 5], "test": [3, 7]}

COLORS = {"H1": "#1f77b4", "L1": "#ff7f0e", "freq": "#1f77b4", "bayes": "#d62728", "ml": "#2ca02c",
          "grey": "#7f7f7f", "purple": "#9467bd"}


def style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.figsize": (7, 4), "axes.grid": True, "grid.alpha": 0.3, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110,
                         "savefig.bbox": "tight"})


def savefig(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=150)
    fig.savefig(os.path.join(FIGS, name + ".pdf"))


def save_json(obj, name):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


def load_json(name):
    with open(os.path.join(RESULTS, name)) as fh:
        return json.load(fh)


def chunks():
    return load_json("chunks.json")


# --------------------------------------------------------------------------------------------------------------
# Whitening and window construction
# --------------------------------------------------------------------------------------------------------------
def welch_psd(x, fs=FS_RAW, nperseg_s=4.0, average="median"):
    """One-sided PSD (strain^2/Hz). The median average is robust to glitches (bias corrected by scipy)."""
    return welch(x, fs=fs, nperseg=int(nperseg_s * fs), window="hann", average=average)


def whiten_chunk(x, psd=None, band=FBAND, fs=FS_RAW, psd_kw=None):
    """Whiten a raw chunk and return (v, f_psd, S): v is the band-passed, down-sampled (1024 Hz) series."""
    x = np.asarray(x, dtype=np.float64)
    if psd is None:
        psd = welch_psd(x, fs=fs, **(psd_kw or {}))
    f_psd, S = psd
    n = len(x)
    X = np.fft.rfft(x * tukey(n, alpha=0.01))
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    X = X / np.sqrt(np.interp(f, f_psd, S) * fs / 2.0)
    X[(f < band[0]) | (f > band[1])] = 0.0
    m = n // (fs // FS)
    v = 0.5 * np.fft.irfft(X[: m // 2 + 1], n=m)       # factor 1/2: sum of squares is preserved by the decimation
    return v.astype(np.float32), f_psd, S


def psd_1hz(f_psd, S):
    """PSD sampled on the 1 Hz bins of the analysis windows."""
    return np.interp(FREQ, f_psd, S)


def make_windows(v, trim=TRIM):
    """Non-overlapping 1 s windows of the whitened series (first/last ``trim`` seconds removed)."""
    v = v[trim * FS: len(v) - trim * FS]
    n = len(v) // NWIN
    return v[: n * NWIN].reshape(n, NWIN)


def load_chunk(i, psd_kw=None, band=FBAND, cache=True):
    """Windows of chunk ``i`` (index in chunks.json): returns dict(v, t0, S1, f_psd, S, gps)."""
    meta = chunks()[i]
    tag = "" if (psd_kw is None and band == FBAND) else "_" + hashlib.md5(str((psd_kw, band)).encode()).hexdigest()[:8]
    fn = os.path.join(CACHE, f"win_{meta['gps_start']}{tag}.npz")
    if cache and os.path.exists(fn):
        d = np.load(fn)
        return {k: d[k] for k in d.files}
    x = np.load(os.path.join(CACHE, "raw", meta["file"]))
    v, f_psd, S = whiten_chunk(x, psd_kw=psd_kw, band=band)
    w = make_windows(v)
    out = {"v": w, "S1": psd_1hz(f_psd, S), "f_psd": f_psd, "S": S,
           "gps": meta["gps_start"] + TRIM + np.arange(len(w), dtype=float)}
    if cache:
        os.makedirs(CACHE, exist_ok=True)
        np.savez_compressed(fn, **out)
    return out


def load_coinc_chunk(k, det):
    """Whitened windows of chunk ``k`` of results/coinc_chunks.json for one detector ("H1" or "L1"): dict(v, S1, f_psd, S, gps).
    Windows with the same index are simultaneous in the two detectors."""
    meta = load_json("coinc_chunks.json")[k]
    fn = os.path.join(CACHE, f"win_{det}_{meta['gps_start']}.npz")
    if os.path.exists(fn):
        d = np.load(fn); return {kk: d[kk] for kk in d.files}
    x = np.load(os.path.join(CACHE, "raw", meta["files"][det]))
    v, f_psd, S = whiten_chunk(x); w = make_windows(v)
    out = {"v": w, "S1": psd_1hz(f_psd, S), "f_psd": f_psd, "S": S, "gps": meta["gps_start"] + TRIM + np.arange(len(w), dtype=float)}
    np.savez_compressed(fn, **out); return out


def load_split(split, **kw):
    """Concatenate the windows of a split; returns v (n, 1024), chunk id per window, list of chunk dicts."""
    vs, ids, ch = [], [], []
    for i in SPLIT[split]:
        d = load_chunk(i, **kw)
        vs.append(d["v"]); ids.append(np.full(len(d["v"]), i)); ch.append(d)
    return np.concatenate(vs), np.concatenate(ids), ch


def to_bins(v, band=BAND):
    """Window series (..., 1024) -> complex band bins V (..., 513); out-of-band bins are zeroed."""
    V = np.fft.rfft(v, axis=-1) / np.sqrt(NWIN / 2.0)
    return V * band


def to_series(V):
    return np.fft.irfft(V * np.sqrt(NWIN / 2.0), n=NWIN, axis=-1).astype(np.float32)


# --------------------------------------------------------------------------------------------------------------
# Burst waveforms: exact Fourier transforms of Gaussian-enveloped signals (continuous FT, signal centred at t = 0)
# --------------------------------------------------------------------------------------------------------------
def _cgauss(f, f0, fd, sigma):
    """FT of exp(-t^2/2 sigma^2) exp(i(2 pi f0 t + pi fd t^2))."""
    a = 1.0 / sigma ** 2 - 2j * np.pi * fd
    beta = 2 * np.pi * (f0 - f)
    return np.sqrt(2 * np.pi / a) * np.exp(-beta ** 2 / (2 * a))


def sine_gaussian_ft(f, f0, Q, phi=0.0, positive_only=False):
    """h(t) = exp(-t^2/2 s^2) cos(2 pi f0 t + phi), s = Q / (2 pi f0)."""
    s = Q / (2 * np.pi * f0)
    out = 0.5 * np.exp(1j * phi) * _cgauss(f, f0, 0.0, s)
    if not positive_only:
        out = out + 0.5 * np.exp(-1j * phi) * _cgauss(f, -f0, 0.0, s)
    return out


def gaussian_pulse_ft(f, sigma):
    """h(t) = exp(-t^2 / 2 sigma^2)."""
    return _cgauss(f, 0.0, 0.0, sigma)


def chirp_ft(f, f0, fdot, sigma, phi=0.0):
    """h(t) = exp(-t^2/2 sigma^2) sin(2 pi f0 t + pi fdot t^2 + phi)  (instantaneous frequency f0 + fdot t)."""
    return (np.exp(1j * phi) * _cgauss(f, f0, fdot, sigma) - np.exp(-1j * phi) * _cgauss(f, -f0, -fdot, sigma)) / 2j


MORPHS = ("sg", "pulse", "chirp")
MORPHS_ALL = MORPHS + ("imr",)      # "imr": IMRPhenomD binary-black-hole signals (library built by imr_tools.py, needs lalsuite once)

_IMR_LIB = {}


def imr_library(split="train"):
    """Cache of IMRPhenomD waveforms in the frequency bins of a 1 s window (built by `python imr_tools.py`).
    Two disjoint libraries with the same prior: "train" (injection kind "imr", used to train networks) and "test"
    (kind "imr_test", waveforms never seen in training: use it for every evaluation)."""
    if split not in _IMR_LIB:
        fn = os.path.join(CACHE, "imr_library.npz" if split == "train" else "imr_library_test.npz")
        if not os.path.exists(fn):
            raise FileNotFoundError(f"{fn} is missing: run `python imr_tools.py` (needs `pip install lalsuite`)")
        d = np.load(fn); _IMR_LIB[split] = {k: d[k] for k in d.files}
    return _IMR_LIB[split]


def draw_params(kind, rng, n):
    """Random morphology parameters (dict of arrays). Priors are the injection priors of the whole project."""
    p = {"t0": rng.uniform(0.25, 0.75, n), "phi": rng.uniform(0, 2 * np.pi, n)}
    if kind == "sg":
        p["f0"] = np.exp(rng.uniform(np.log(50), np.log(400), n))
        p["Q"] = rng.uniform(4, 15, n)
    elif kind == "pulse":
        p["sigma"] = np.exp(rng.uniform(np.log(0.8e-3), np.log(5e-3), n))
    elif kind == "chirp":
        p["f0"] = rng.uniform(50, 150, n)
        p["fdot"] = rng.uniform(150, 600, n)
        p["sigma"] = rng.uniform(0.03, 0.08, n)
    elif kind in ("imr", "imr_test"):
        lib = imr_library("test" if kind == "imr_test" else "train"); idx = rng.integers(0, len(lib["H"]), n)
        p["lib"] = idx; p["tc"] = lib["tc"][idx]; p["t0"] = rng.uniform(-0.03, 0.03, n)       # t0 = extra time shift of the merger
    else:
        raise ValueError(kind)
    return p


def waveform_bins(kind, p, S1, band=BAND):
    """Whitened frequency bins (n, 513) of unit-amplitude waveforms placed at p['t0'] inside the window.
    V_k = 2 h(f_k) / sqrt(S_n(f_k)) so that the noise bins are N(0, 1) per real component."""
    n = len(p["t0"])
    f = FREQ[None, :]
    if kind == "sg":
        H = sine_gaussian_ft(f, p["f0"][:, None], p["Q"][:, None], p["phi"][:, None])
    elif kind == "pulse":
        H = gaussian_pulse_ft(f, p["sigma"][:, None]) * np.ones((n, 1))
    elif kind == "chirp":
        H = chirp_ft(f, p["f0"][:, None], p["fdot"][:, None], p["sigma"][:, None], p["phi"][:, None])
    elif kind in ("imr", "imr_test"):
        H = imr_library("test" if kind == "imr_test" else "train")["H"][p["lib"]].astype(np.complex128)
    H = H * np.exp(-2j * np.pi * f * p["t0"][:, None])
    return 2.0 * H / np.sqrt(np.atleast_2d(S1)) * band


def snr_of(V):
    """Optimal SNR of whitened bins."""
    return np.sqrt(np.sum(np.abs(V) ** 2, axis=-1))


def inject(v_noise, kind, snr, rng, S1, params=None, band=BAND):
    """Add a burst of the given morphology at optimal SNR ``snr`` (scalar or array) to noise windows.
    Returns (series, params, snr_true). ``S1`` can be (513,) or (n, 513)."""
    n = len(v_noise)
    p = draw_params(kind, rng, n) if params is None else params
    W = waveform_bins(kind, p, S1, band)
    snr = np.broadcast_to(np.asarray(snr, dtype=float), (n,)).copy()
    W = W * (snr / snr_of(W))[:, None]
    X = to_bins(v_noise, band) + W
    return to_series(X), p, snr


# --------------------------------------------------------------------------------------------------------------
# Matched-filter bank of sine-Gaussians (frequentist statistic and Bayesian likelihood on a grid)
# --------------------------------------------------------------------------------------------------------------
class SGBank:
    """Complex whitened templates T(f0, Q) (positive-frequency part, phase-free) on a grid."""

    def __init__(self, S1, f0s=None, Qs=None, band=BAND):
        self.f0s = np.geomspace(40, 450, 40) if f0s is None else np.asarray(f0s)
        self.Qs = np.array([3.0, 6.0, 10.0, 16.0]) if Qs is None else np.asarray(Qs)
        self.grid = [(f, q) for q in self.Qs for f in self.f0s]
        self.f0 = np.array([g[0] for g in self.grid])
        self.Q = np.array([g[1] for g in self.grid])
        T = 2.0 * sine_gaussian_ft(FREQ[None, :], self.f0[:, None], self.Q[:, None], 0.0, positive_only=True)
        T = T / np.sqrt(S1)[None, :] * band
        self.norm2 = np.sum(np.abs(T) ** 2, axis=1)
        self.T = (T / np.sqrt(self.norm2)[:, None]).astype(np.complex64)   # unit-norm templates

    def scan(self, X, tau_range=(0.15, 0.85), batch=64, sigma_prior=None, return_cube=False):
        """X: complex bins (n, 513). Returns dict with rho2max, tau, f0, Q and (optional) ln Bayes factor
        and the rho^2 cube (n, nT, ntau).
        rho^2(tau, f0, Q) = |sum_k X_k conj(T_k) e^{2 pi i k tau}|^2 : -2 ln(L(0)/L(max over amplitude, phase))."""
        X = np.asarray(X, dtype=np.complex64)
        i0, i1 = int(tau_range[0] * NWIN), int(tau_range[1] * NWIN)
        taus = np.arange(i0, i1) / NWIN
        n = len(X)
        out = {"rho2": np.zeros(n), "tau": np.zeros(n), "f0": np.zeros(n), "Q": np.zeros(n)}
        if sigma_prior is not None:
            out["lnBF"] = np.zeros(n)
        cubes = []
        for a in range(0, n, batch):
            xb = X[a:a + batch]
            Z = xb[:, None, :] * np.conj(self.T)[None]                     # (b, nT, 513)
            full = np.zeros(Z.shape[:2] + (NWIN,), dtype=np.complex64)
            full[..., :NB] = Z
            z = np.fft.ifft(full, axis=-1)[..., i0:i1] * NWIN
            rho2 = (np.abs(z) ** 2).astype(np.float32)                     # (b, nT, ntau)
            flat = rho2.reshape(len(xb), -1)
            j = flat.argmax(1)
            it, itau = np.unravel_index(j, rho2.shape[1:])
            out["rho2"][a:a + batch] = flat.max(1)
            out["tau"][a:a + batch] = taus[itau]
            out["f0"][a:a + batch] = self.f0[it]
            out["Q"][a:a + batch] = self.Q[it]
            if sigma_prior is not None:
                s2 = sigma_prior ** 2
                lnl = 0.5 * flat * s2 / (1 + s2) - np.log1p(s2)
                m = lnl.max(1)
                out["lnBF"][a:a + batch] = m + np.log(np.mean(np.exp(lnl - m[:, None]), axis=1))
            if return_cube:
                cubes.append(rho2)
        out["taus"] = taus
        if return_cube:
            out["cube"] = np.concatenate(cubes)
        return out


# --------------------------------------------------------------------------------------------------------------
# Excess-power statistic (morphology-agnostic baseline)
# --------------------------------------------------------------------------------------------------------------
DURATIONS_MS = (10, 25, 50, 100, 200)


def sliding_energy(v, lengths=None):
    """Sliding sums of v^2 for several durations; returns {L: (n, nwin-L+1)}."""
    lengths = [int(d * FS / 1000) for d in DURATIONS_MS] if lengths is None else lengths
    c = np.concatenate([np.zeros((len(v), 1)), np.cumsum(v.astype(np.float64) ** 2, axis=1)], axis=1)
    return {L: c[:, L:] - c[:, :-L] for L in lengths}


class ExcessPower:
    """max over time and duration of the z-scored band energy; mu/sigma per duration fitted on background."""

    def fit(self, v_bkg):
        self.par = {}
        for L, E in sliding_energy(v_bkg).items():
            e = E[:, int(0.15 * FS): int(0.85 * FS)].ravel()
            self.par[L] = (np.median(e), 1.4826 * np.median(np.abs(e - np.median(e))))
        return self

    def stat(self, v):
        best = None
        for L, E in sliding_energy(v).items():
            mu, sd = self.par[L]
            z = ((E[:, int(0.15 * FS): int(0.85 * FS)] - mu) / sd).max(1)
            best = z if best is None else np.maximum(best, z)
        return best


# --------------------------------------------------------------------------------------------------------------
# Detection-performance helpers
# --------------------------------------------------------------------------------------------------------------
def threshold_at_far(bkg_stat, far):
    """Empirical (1 - far) quantile of the background statistic, per 1 s window."""
    return np.quantile(bkg_stat, 1.0 - far)


def efficiency(sig_stat, thr):
    k = int(np.sum(sig_stat > thr)); n = len(sig_stat)
    p = k / n
    return p, np.sqrt(max(p * (1 - p), 1e-12) / n)


def roc(bkg_stat, sig_stat, n=400):
    thr = np.unique(np.quantile(bkg_stat, 1 - np.logspace(-4, 0, n)))
    far = np.array([(bkg_stat > t).mean() for t in thr])
    eff = np.array([(sig_stat > t).mean() for t in thr])
    return far, eff


def auc(bkg_stat, sig_stat):
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([bkg_stat, sig_stat]))
    n0, n1 = len(bkg_stat), len(sig_stat)
    return (r[n0:].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


# --------------------------------------------------------------------------------------------------------------
# Helpers for sets of windows coming from different chunks (each chunk has its own PSD and template bank)
# --------------------------------------------------------------------------------------------------------------
def banks_for(chunk_dicts, ids, **kw):
    return {i: SGBank(d["S1"], **kw) for i, d in zip(ids, chunk_dicts)}


def scan_mixed(series, cid, banks, band=BAND, **kw):
    """Run the bank scan on windows of several chunks. ``cid`` gives the chunk of each window."""
    cid = np.asarray(cid)
    out = None
    for c in np.unique(cid):
        m = cid == c
        o = banks[c].scan(to_bins(series[m], band), **kw)
        if out is None:
            out = {k: np.zeros((len(series),) + a.shape[1:], dtype=a.dtype) for k, a in o.items() if k != "taus"}
            out["taus"] = o["taus"]
        for k, a in o.items():
            if k != "taus":
                out[k][m] = a
    return out


def exact_template_stat(X, p, S1, band=BAND):
    """Neyman-Pearson (oracle) statistic: |z|^2 of the phase-free sine-Gaussian template at the TRUE (t0, f0, Q)."""
    T = 2 * sine_gaussian_ft(FREQ[None, :], p["f0"][:, None], p["Q"][:, None], 0.0, positive_only=True)
    T = T * np.exp(-2j * np.pi * FREQ[None, :] * p["t0"][:, None]) / np.sqrt(np.atleast_2d(S1)) * band
    T /= np.sqrt(np.sum(np.abs(T) ** 2, axis=1))[:, None]
    return np.abs(np.sum(X * np.conj(T), axis=1)) ** 2


# --------------------------------------------------------------------------------------------------------------
# Bootstrap error bars (notebook 11)
# --------------------------------------------------------------------------------------------------------------
def snr_level(eff, snrs, level):
    """SNR at which a (monotonised) efficiency curve crosses ``level``; NaN if it never does."""
    e = np.maximum.accumulate(np.asarray(eff, float))
    return float(np.interp(level, e, np.asarray(snrs, float))) if e.max() >= level else np.nan


def bootstrap_rho(bkg, sig_by_snr, snrs, far, levels=(0.5, 0.9), B=500, seed=0):
    """Bootstrap of rho_level at a given FAP: resamples the background (threshold uncertainty) and, independently,
    the injections at every SNR. ``sig_by_snr`` is a list of statistic arrays, one per SNR.
    Returns (point estimates, array B x len(levels))."""
    rng = np.random.default_rng(seed)
    thr0 = threshold_at_far(bkg, far)
    point = [snr_level([(s > thr0).mean() for s in sig_by_snr], snrs, lv) for lv in levels]
    out = np.full((B, len(levels)), np.nan)
    for b in range(B):
        thr = threshold_at_far(rng.choice(bkg, len(bkg)), far)
        eff = [(rng.choice(s, len(s)) > thr).mean() for s in sig_by_snr]
        out[b] = [snr_level(eff, snrs, lv) for lv in levels]
    return np.array(point), out


def ci(x, q=(16, 84)):
    """Percentile interval of a bootstrap sample, ignoring NaN."""
    return tuple(float(v) for v in np.nanpercentile(x, q))


# --------------------------------------------------------------------------------------------------------------
# Signal-consistency test (notebook 16): Allen chi^2 of the best template and the re-weighted ("new") SNR
# --------------------------------------------------------------------------------------------------------------
def template_index(bank, f0, Q):
    """Index in ``bank.grid`` of the best template returned by ``bank.scan`` (matched on the (f0, Q) labels)."""
    key = {(round(float(a), 6), round(float(b), 6)): i for i, (a, b) in enumerate(bank.grid)}
    return np.array([key[(round(float(a), 6), round(float(b), 6))] for a, b in zip(f0, Q)])


def allen_chi2(bank, X, tau, idx, p=8):
    """chi^2 of the best-fit template: the template is split in ``p`` frequency bands of equal template power, z_i is the
    complex matched-filter output of band i at the best lag ``tau`` (fraction of the window); chi^2 = sum_i |z_i - z/p|^2 / (1/p).
    For Gaussian noise and an exact template chi^2 ~ chi^2 with 2 (p-1) degrees of freedom. Returns chi^2 / (2 (p-1))."""
    X = np.asarray(X, np.complex64); T = bank.T[idx]                              # (n, 513) unit-norm templates
    P = np.abs(T) ** 2; band = np.minimum((np.cumsum(P, axis=1) * p - 1e-9).astype(int), p - 1)
    prod = X * np.conj(T) * np.exp(2j * np.pi * FREQ[None, :] * np.asarray(tau)[:, None])
    z = prod.sum(1); chi = np.zeros(len(X))
    for i in range(p):
        zi = (prod * (band == i)).sum(1); chi += np.abs(zi - z / p) ** 2 * p
    return chi / (2 * (p - 1))


def new_snr(rho2, chi2_red):
    """Re-weighted SNR (Usman et al.): rho / [(1 + c^3)/2]^(1/6) for c = chi^2/dof > 1, rho otherwise."""
    rho = np.sqrt(rho2)
    return np.where(chi2_red > 1, rho / ((1 + chi2_red ** 3) / 2) ** (1 / 6), rho)
