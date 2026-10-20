# Detecting short gravitational-wave bursts in real LIGO noise

### Frequentist, Bayesian and machine-learning analysis of simulated bursts injected into GWOSC O1 data

Frequentist, Bayesian and machine-learning analysis of simulated bursts (sine-Gaussian, Gaussian pulse, linear chirp) injected into
**real** LIGO-Hanford (H1) strain from the first observing run (O1), public data from GWOSC.
Final project for the *Statistics and Data Analysis* course (Università di Milano-Bicocca, A.Y. 2025–2026), Giacomo Moretti.
The complete report is `Docs/report/main.pdf`.

The organisation follows the earlier projects on LHCb open data and on AFM force curves: one shared data set, one notebook per
method, one common set of figures of merit, a report that says where each number comes from.

> **Research question.** At a fixed false-alarm probability on real LIGO noise, how do a frequentist matched-filter bank, a Bayesian
> analysis of the same likelihood and machine-learning detectors compare in detection efficiency, in the quality of their uncertainty
> statements, and in robustness to the non-Gaussian noise and to signal morphologies they were not designed for?

## Repository layout

```
Code_General/
  00_waveform_exploration.ipynb          exploratory prototype: GW150914, PSD, whitening, Q-transform, first injections
  01_data_whitening_validation.ipynb     8 O1 chunks, PSD, whitening, 1 s windows, Gaussianity / stationarity tests, split by chunk
  02_waveforms_and_injections.ipynb      exact analytic burst transforms, optimal SNR, validation of the injection chain
  03_frequentist_analysis.ipynb          matched-filter bank, calibration, efficiency / ROC, MLE + Fisher, toy MC, systematics
  04_bayesian_analysis.ipynb             exact grid posterior + emcee, Bayes factors, prior dependence, SBC, credible vs confidence
  05_ml_detection.ipynb                  features + Ridge/Lasso/RF/GBT, 1-D CNN, ensemble, MC dropout, autoencoder, calibration
  gwosc_tools.py                         functions shared by notebooks 01-05
  download_data.py                       downloads the 8 raw chunks (cache/raw/) and writes results/chunks.json
  results/                               summary JSON of every notebook (tracked)
  figures/                               figures of every notebook, PNG + PDF (tracked)
  cache/                                 raw strain, whitened windows, scans, trained models (NOT tracked, regenerable)
Docs/report/                             the report (main.tex, sections/, biblio.bib, figures/, main.pdf)
environment.yml                          conda environment of all the code
```

## Data

Eight chunks of 4096 s of H1 strain at 4096 Hz from O1 (Sept. 2015 – Jan. 2016), inside the H1 science segments and at least 2000 s from
every catalogued event, selected and downloaded by `Code_General/download_data.py` (about 0.5 GB in float32, needs internet access to
`gwosc.org`). The split is **by chunk** (train 0,2,4,6 · validation 1,5 · test 3,7), never by window.

## Environment

```
conda env create -f environment.yml
conda activate gwosc-burst
python -m ipykernel install --user --name cern
```

All notebooks use the kernel `cern`. Then, from `Code_General/`:

```
python download_data.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=cern 0N_*.ipynb    # N = 1 ... 5, in order
```

Run times on a 10-core laptop: 01 ≈ 1 min, 02 ≈ 1 min, 03 ≈ 10–15 min, 04 ≈ 20 min, 05 ≈ 30–40 min with training from scratch (a few minutes
with the saved models). All random numbers use fixed seeds; long blocks cache their output and resume on re-run. Notebooks 03–05 read the
whitened windows cached by notebook 01 (rebuilt automatically from `cache/raw/` if missing); notebook 04 reads `results/03_summary.json`.

The report is compiled with `tectonic -X compile main.tex` from `Docs/report/`.

## Pipeline

| Notebook | Content | Output |
|---|---|---|
| `01` | PSD (median Welch) of each chunk, whitening, 30–500 Hz band, 1 s windows; tests of the Gaussian-noise assumption | `results/01_summary.json` |
| `02` | sine-Gaussian / pulse / chirp as exact Fourier transforms, analytic optimal SNR, injections as whitened bins, validation (non-central χ²) | `results/02_summary.json` |
| `03` | 160-template bank, profile-likelihood statistic, look-elsewhere (N_eff), calibration on unseen epochs, efficiency vs oracle and excess power, `iminuit` MLE vs Fisher, Rician toy MC with Neyman/Feldman–Cousins belts, Barlow systematics | `results/03_summary.json` |
| `04` | analytic amplitude marginalisation, exact grid posterior, `emcee` cross-check, Bayes factors and prior dependence, simulation-based calibration, credible vs confidence intervals | `results/04_summary.json` |
| `05` | 22 features with Ridge/Lasso/RF/GBT (bias–variance, importance, PCA), 1-D CNN, 5-member ensemble, MC dropout, autoencoder, calibration, comparison on the same injections | `results/05_summary.json` |

## Main results

All SNRs are **optimal SNRs**; the false-alarm probability (FAP) is per one-second window of single-detector data.

| Statistic (FAP 10⁻², validation background) | sine-Gaussian ρ₅₀ / ρ₉₀ | pulse ρ₅₀ / ρ₉₀ | chirp ρ₅₀ / ρ₉₀ |
|---|---|---|---|
| matched-filter bank (frequentist) | **4.87** / **6.53** | 5.34 / 6.82 | 6.28 / 8.21 |
| excess power | 6.54 / 9.10 | 5.77 / 7.54 | 10.10 / 12.21 |
| gradient boosting (22 features) | 5.61 / 7.30 | 5.70 / 7.24 | 6.06 / 7.73 |
| CNN ensemble (5) | 5.05 / 7.05 | **4.21** / **5.90** | **5.28** / **6.92** |
| CNN trained on sine-Gaussians only | 5.09 / 6.87 | 6.50 / 12.93 | 6.80 / 10.82 |
| autoencoder (noise only), local max | 6.89 / 8.80 | 6.75 / 8.53 | 8.30 / 11.08 |

(ρ₅₀, ρ₉₀: SNR at 50 % and 90 % efficiency on test-chunk injections.)

- **Real noise.** Gaussian in the bulk (N(0,1) down to ~10⁻⁴), with a glitch floor of a few 10⁻³ per second and strong epoch-to-epoch variations.
  It sets the price of a low FAP: lowering it from 10⁻² to 10⁻³ moves ρ₅₀ of the bank from ≈5.5 to ≈11, and the threshold calibrated on the
  training chunks gives an observed FAP 4.5× below nominal on the test chunks.
- **Frequentist.** The look-elsewhere effect is quantified by N_eff ≈ 6800 independent trials; the oracle with known parameters needs ρ₅₀ = 3.0
  against ≈5 for the bank; the MLE saturates the Fisher bound for ρ ≳ 10; the profile-likelihood interval of the amplitude covers 39 % at s = 0
  (Neyman / Feldman–Cousins cover 68 % everywhere); the dominant systematic is the PSD estimator (+0.025 on ε(ρ=8)).
- **Bayesian.** The grid posterior is validated against a closed form, `emcee` and simulation-based calibration (uniform ranks in Gaussian and in real
  noise); the Bayes factor ranks windows like the frequentist statistic (same efficiency within 0.01), depends on the prior as −2 ln σ_ρ and exceeds 100
  on 0.2–0.9 % of pure-noise windows because of glitches; credible and confidence intervals agree for ρ ≳ 3.
- **Machine learning.** The CNN ensemble matches the bank on sine-Gaussians and is ≈1 SNR unit better on the pulses and chirps it was trained on; a CNN
  trained on one morphology loses efficiency on the others; an autoencoder trained on noise adds nothing to an energy statistic. Calibration is good
  (ECE ≲ 3 %), the ensemble is better than a single network in every proper score, and rejecting the 20 % most uncertain windows raises the AUC from 0.992 to 0.9997.

## Limits

- **Single detector.** No coincidence and no time slides: FAPs are per window, not rates; a second detector would remove most glitches.
- **Injections at a given SNR.** Not converted into distances; the three morphologies are phenomenological.
- **Background statistics.** 1.6·10⁴ training windows limit the lowest usable FAP to ~10⁻³; the 10⁻³ efficiencies are uncertain.
- **Eight chunks.** The epoch dependence is measured with two test chunks.
- **Grid resolution.** The Bayesian grid under-resolves f₀ and t₀ for ρ ≳ 15 (SBC coverage of those two parameters is 0.50–0.60, not the model).
- **ML numbers** are efficiencies for the training injection prior; parameter estimation with ML was not attempted.

## Conventions

Notebooks, comments and documentation are in English. Every notebook states at the beginning what it does and what it does **not** do, and ends
with the output produced and the next notebook. `cache/` is a regenerable cache; `results/` is the source of the numbers quoted here.

## License

Released under the [MIT License](LICENSE). The LIGO strain data are public (GWOSC) and are subject to their own
[data-use terms](https://gwosc.org/terms/).
