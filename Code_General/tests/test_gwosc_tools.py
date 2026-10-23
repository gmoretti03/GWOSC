"""Automatic checks of the core of the pipeline (extension 44). Run from the repository root:
    python -m pytest Code_General/tests -q
Only synthetic data and light cached arrays are used; nothing is downloaded."""
import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import gwosc_tools as g


@pytest.fixture(scope="module")
def S1():
    return np.full(g.NB, 4.0)                                  # flat toy PSD on the 1 Hz bins


def gauss(n, rng):
    return g.to_series((rng.normal(size=(n, g.NB)) + 1j * rng.normal(size=(n, g.NB))) * g.BAND)


def test_gaussian_bins_have_unit_variance_in_band():
    rng = np.random.default_rng(0); V = g.to_bins(gauss(4000, rng))
    assert abs(V.real[:, g.BAND].var() - 1) < 0.02 and abs(V.imag[:, g.BAND].var() - 1) < 0.02
    assert np.all(V[:, ~g.BAND] == 0)


@pytest.mark.parametrize("kind", ["sg", "pulse", "chirp"])
def test_injection_has_the_requested_optimal_snr(kind, S1):
    rng = np.random.default_rng(1); n = 50
    ser, p, snr = g.inject(np.zeros((n, g.NWIN), np.float32), kind, 8.0, rng, S1)
    rho = g.snr_of(g.to_bins(ser))
    assert np.allclose(rho, 8.0, rtol=2e-3)                    # float32 round trip only


def test_injection_is_additive_and_noise_statistics_are_kept(S1):
    rng = np.random.default_rng(2); v = gauss(200, rng)
    ser, _, _ = g.inject(v, "sg", 6.0, np.random.default_rng(3), S1)
    zero, _, _ = g.inject(np.zeros_like(v), "sg", 6.0, np.random.default_rng(3), S1)
    assert np.allclose(ser - v, zero, atol=1e-4)


def test_sg_bank_noise_maximum_is_chi2_2_per_template_and_templates_unit_norm(S1):
    B = g.SGBank(S1); assert np.allclose(np.sum(np.abs(B.T) ** 2, axis=1), 1, atol=1e-4)
    rng = np.random.default_rng(4); o = B.scan(g.to_bins(gauss(300, rng)))
    assert 5 < np.median(o["rho2"]) < 25                       # max over ~160 templates x lags of chi^2_2 variables (mean 2)


def test_bank_recovers_injected_signal(S1):
    rng = np.random.default_rng(5); B = g.SGBank(S1)
    ser, p, _ = g.inject(np.zeros((40, g.NWIN), np.float32), "sg", 12.0, rng, S1)
    o = B.scan(g.to_bins(ser))
    assert np.median(np.sqrt(o["rho2"])) > 9.5 and np.median(np.abs(o["tau"] - p["t0"])) < 0.01


def test_oracle_statistic_equals_rho2_for_noiseless_exact_template(S1):
    rng = np.random.default_rng(6); ser, p, snr = g.inject(np.zeros((30, g.NWIN), np.float32), "sg", 9.0, rng, S1)
    stat = g.exact_template_stat(g.to_bins(ser), p, S1)
    assert np.allclose(stat, 81.0, rtol=0.05)


def test_threshold_efficiency_and_auc_helpers():
    rng = np.random.default_rng(7); bkg = rng.normal(size=20000); thr = g.threshold_at_far(bkg, 1e-2)
    assert abs((bkg > thr).mean() - 1e-2) < 1e-3
    eff, err = g.efficiency(rng.normal(3, 1, 2000), thr); assert abs(eff - 0.750) < 0.04 and err < 0.02          # 1 - Phi(2.326 - 3)
    assert abs(g.auc(rng.normal(size=5000), rng.normal(size=5000)) - 0.5) < 0.03


def test_bootstrap_rho_and_snr_level():
    snrs = [3, 4, 5, 6, 7, 8]; assert g.snr_level([0, .1, .4, .6, .9, 1], snrs, .5) == pytest.approx(5.5)
    assert np.isnan(g.snr_level([0, 0, 0, .1, .2, .3], snrs, .5))
    rng = np.random.default_rng(8); bkg = rng.normal(size=4000)
    sig = [rng.normal(s * 0.6, 1, 300) for s in snrs]
    p, b = g.bootstrap_rho(bkg, sig, snrs, 1e-2, B=50, seed=0); lo, hi = g.ci(b[:, 0])
    assert lo <= p[0] <= hi


def test_allen_chi2_is_about_one_for_signal_and_larger_for_a_glitch(S1):
    rng = np.random.default_rng(9); B = g.SGBank(S1)
    ser, _, _ = g.inject(gauss(80, rng), "sg", 14.0, rng, S1); o = B.scan(g.to_bins(ser))
    c_sig = g.allen_chi2(B, g.to_bins(ser), o["tau"], g.template_index(B, o["f0"], o["Q"]))
    spike = gauss(80, rng); spike[:, 500] += 40.0; o = B.scan(g.to_bins(spike))
    c_gl = g.allen_chi2(B, g.to_bins(spike), o["tau"], g.template_index(B, o["f0"], o["Q"]))
    assert np.median(c_sig) < 2 and np.median(c_gl) > np.median(c_sig)


def test_splits_are_by_chunk_and_disjoint():
    s = g.SPLIT; allc = s["train"] + s["val"] + s["test"]
    assert len(allc) == len(set(allc)) == 8 and set(s["train"]).isdisjoint(s["test"])
