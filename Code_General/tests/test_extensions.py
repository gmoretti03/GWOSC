"""Checks of the helpers added in the second round of extensions (glitch_tools, pe_tools, network_tools, precession_tools). Needs lalsuite."""
import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
pytest.importorskip("lalsimulation")
import gwosc_tools as g, glitch_tools as gt, pe_tools as pe, network_tools as nt, precession_tools as pt

S1 = np.full(g.NB, 4.0)


@pytest.mark.parametrize("kind", ["blip", "whistle", "arch"])
def test_glitch_injection_has_the_requested_norm(kind):
    rng = np.random.default_rng(0); ser = gt.inject_glitch(np.zeros((20, g.NWIN), np.float32), kind, 9.0, rng)
    assert np.allclose(g.snr_of(g.to_bins(ser)), 9.0, rtol=2e-3)


def test_glitch_bank_templates_are_unit_norm_and_find_a_blip():
    b = gt.GlitchBank(); assert np.allclose(np.sum(np.abs(b.T) ** 2, axis=1), 1, atol=1e-4)
    rng = np.random.default_rng(1); ser = gt.inject_glitch(np.zeros((30, g.NWIN), np.float32), "blip", 12.0, rng)
    assert np.median(np.sqrt(b.scan(g.to_bins(ser))["rho2"])) > 8


def test_exact_model_likelihood_peaks_near_the_truth():
    M, q, c1, c2, tau = 60.0, 0.7, 0.1, -0.1, 0.75
    X = 10.0 * pe.template(M, q, c1, c2, S1) * np.exp(-2j * np.pi * g.FREQ * tau)           # noiseless signal of norm 10 (phase-free template)
    good = pe.lnprob([M, q, c1, c2, tau], X, S1); bad = pe.lnprob([40.0, 0.4, 0.3, 0.3, tau + 0.05], X, S1)
    assert np.isfinite(good) and good > bad + 20
    assert pe.lnprob([120.0, q, c1, c2, tau], X, S1) == -np.inf                                 # outside the prior


def test_peaks_recover_the_lag_of_an_imr_signal():
    import imr_tools as it
    bank = it.IMRBank(S1); T = bank.T[5]; X = 12.0 * T * np.exp(-2j * np.pi * g.FREQ * 0.5)
    tau, z = nt.peaks(bank, X[None]); assert tau.shape == (1, len(bank.T)) and abs(int(tau[0, 5]) - int(0.5 * g.NWIN)) <= 1 and abs(abs(z[0, 5]) - 12.0) < 0.2


def test_polarisation_waveforms_are_consistent_with_the_single_detector_library_function():
    import imr_tools as it
    Hp, Hc = pt.waveform_pol("D", 40.0, 30.0, [0, 0, 0.2], [0, 0, -0.1], 0.0, 0.7)
    H, _ = it.one_waveform(40.0, 30.0, 0.2, -0.1, 0.0, 1.0, 0.0, 0.7)
    assert np.allclose(Hp, H, rtol=1e-3, atol=1e-6 * np.abs(H).max())                          # fp=1, fc=0 gives h_plus
