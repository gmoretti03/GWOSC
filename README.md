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
  06_sequence_models_hypothesis_testing.ipynb   Transformer / GRU detectors, masked-patch pre-training, 4-class morphology vs frequentist and Bayesian model comparison
  07_tuning_sequence_models.ipynb        stress test of the Transformer-vs-CNN result: fresh injections, learning rate, patch size, convolutional stem
  08_realistic_inspiral_signals.ipynb    binary-black-hole (IMRPhenomD) injections: chirp-mass mapping of the linear chirp, IMR template bank, detection of realistic signals
  09_two_detector_coincidence.ipynb      H1+L1: coincident statistic, time-slide background down to FAP 1e-6, zero-lag check, efficiency vs single detector
  10_open_set_goodness_of_fit.ipynb      flagging signals that belong to none of the hypotheses: matched-filter residual, network open-set scores
  11_bootstrap_error_bars.ipynb          bootstrap errors on rho50/rho90 and paired model comparisons (nb 06-07 detectors)
  12_threshold_convention.ipynb          bank rho50 with thresholds from train / validation / pooled / test background; fixes the report convention
  13_real_events.ipynb                   GW150914 / GW151012 / GW151226 in H1+L1 through the unchanged banks and coincident statistic
  14_continuous_search.ipynb             mini continuous search on 2.25 h of H1+L1: sliding windows, clustering, time-slide FAR in yr^-1, candidate list, injection recovery
  15_cnn_vs_bank_ablation.ipynb          why the CNN beats the bank: Gaussian vs real noise, 640-template bank, amplitude veto
  16_chi2_veto_ranking.ipynb             Allen chi2 and new SNR for the bank; logistic / boosted ranking statistics
  17_hard_negative_mining.ipynb          fine-tuning the CNN with hard negatives vs control (no measurable gain)
  18_glitch_families.ipynb               unsupervised families of high-statistic noise windows; which ones alarm bank / chi2 / CNN
  19_unfolding.ipynb                    unfolding of the total-mass distribution: inversion, IBU, OmniFold on IMR bank output
  20_prior_mismatch_base_rate.ipynb     CNN and bank outside the training prior; PPV of an alarm vs astrophysical base rate
  21_injected_glitches_robust_bayes.ipynb  labelled injected glitches (blip, whistle, arch) and a glitch-aware Bayes factor
  22_imr_parameter_estimation.ipynb      Mc, M, q, merger time of IMR signals: bank MLE, grid posterior, CNN regression with Gaussian head
  23_background_rigour.ipynb             leave-one-chunk-out epoch dependence, time-local background, GPD tail extrapolation, Feldman-Cousins rate limits
  24_rare_event_mc.ipynb                 subset simulation of the Gaussian-noise tail of the bank statistic down to FAP 1e-10
  25_robustness_conformal_interpretability.ipynb  systematics, conformal thresholds/sets, saliency, latency, FGSM attacks on the CNN
  26_scaling_augmentation_adaptation.ipynb  CNN scaling with training windows, augmentation, label noise, H1->L1 transfer
  27_bank_design_local_whitening_psd.ipynb  bank design (number/placement of templates), time-local whitening, PSD-estimation uncertainty
  28_physical_units_range.ipynb          h_rss benchmarks, ringdowns/cusps/white-noise bursts, physical BBH injections: sensitive range and volume
  29_precession_higher_modes.ipynb       IMRPhenomPv2 / XPHM libraries (precession, higher modes) vs the detectors
  30_seeds_paired_tests.ipynb            5 CNN / 3 hybrid / 3 Transformer seeds, seed scatter, permutation and Welch tests, 5-CNN ensemble
  31_coherent_network.ipynb              coherent (lambda_1) vs incoherent vs Fisher statistics, physical network injections (antenna patterns, delay), 10 background chunks
  32_bayesian_extensions.ipynb           exact-model MCMC (emcee), two-detector signal-vs-glitch evidence, SBC of the grid posterior, hierarchical population inference
  33_unsupervised_flow_vae.ipynb         RealNVP flow and VAE on noise only as anomaly detectors
  34_other_architectures.ipynb           ResNet-1D, TCN, 2-D CNN on spectrograms, EdgeConv graph network
  35_data_quality_flags.ipynb            GWOSC CAT1/CAT2 flags as vetoes: dead time vs glitch floor
  36_o2_o3_generalisation.ipynb          unchanged pipeline on O2, O3a, O3b chunks (download_runs.py)
  37_limits_low_mass_bns_nsbh_ecc.ipynb  SNR retained by the 1 s / 30-500 Hz window (low masses, BNS), NSBH and eccentric libraries vs the detectors
  38_virgo_snr_budget.ipynb              SNR budget and sensitive volume of HLV against HL (O2 August 2017); download_virgo.py
  39_transformer_tuning_round2.ipynb     Transformer depth / width / 30 epochs (variants tf16_d6/w128/long_fresh of seq_models.py)
  pe_tools.py, network_tools.py          likelihood/sampler and network injection helpers of notebook 32
  download_coinc_extra.py                6 more H1+L1 chunks -> results/coinc_chunks_extra.json
  precession_tools.py                    waveform generation of notebook 29 (needs lalsuite to build)
  glitch_tools.py                        glitch shapes, injection and GlitchBank of notebook 21
  download_events.py                     downloads the three event chunks (NaN gaps handled in nb 13)
  make_summary_figures.py                summary rho50 figure and pipeline diagram of the report (fig13_*)
  gwosc_tools.py                         functions shared by notebooks 01-10
  seq_models.py                          models, dataset and resumable training loop of notebooks 07-08 (run it to train the variants in parallel processes)
  imr_tools.py                           IMRPhenomD signal libraries (train and held-out test) and the IMR template bank (needs lalsuite to build the libraries)
  open_set.py                            family fits and goodness-of-fit scores of notebook 10
  download_data.py                       downloads the 8 raw H1 chunks (cache/raw/) and writes results/chunks.json
  download_coinc_data.py                 downloads the 6 H1+L1 chunks of notebook 09 and writes results/coinc_chunks.json
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

## Tests and reproducibility

`python -m pytest Code_General/tests -q` (19 checks: whitening, exact SNR of injections, bank, oracle, bootstrap, χ², splits, glitch/PE/network helpers; synthetic data only); `bash Code_General/run_all.sh` regenerates data, libraries and every notebook in order; `python Code_General/make_checksums.py --check` verifies the raw strain files against `results/data_checksums.json`.

## Environment

```
conda env create -f environment.yml
conda activate gwosc-burst
python -m ipykernel install --user --name cern
```

All notebooks use the kernel `cern`. Then, from `Code_General/`:

```
python download_data.py
python download_coinc_data.py        # only for notebook 09 (H1+L1 chunks)
python imr_tools.py                  # only for notebooks 08-10 (IMRPhenomD libraries, needs lalsuite: pip install lalsuite)
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=cern NN_*.ipynb      # NN = 01 ... 10, in order
```

Run times on a 10-core laptop on AC power: 01 ≈ 1 min, 02 ≈ 1 min, 03 ≈ 10–15 min, 04 ≈ 20 min, 05 ≈ 30–40 min with training from scratch (a few minutes
with the saved models), 06 ≈ 1–2 h, 07 ≈ 3 h of training on four processes, 08 ≈ 10 min plus 20 min of training, 09 ≈ 10 min, 10 ≈ 10 min. All random numbers use fixed seeds; long blocks cache their output and resume on re-run. Notebooks 03–05 read the
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
| `06` | patch Transformer and GRU against the CNN (efficiency at FAP 10⁻², 10⁻³, glitches, attention), masked-patch pre-training and label efficiency, 4-class (noise / sg / pulse / chirp) detectors against frequentist and Bayesian model comparison, an out-of-family burst | `results/06_summary.json`, `figures/fig06_*` |
| `07` | tuning of the sequence models, one lever at a time (fresh injections each epoch, learning rate, patch size, convolutional stem); diagnosis by training vs validation loss; same test injections and thresholds as 05–06; four-class networks with fresh injections | `results/07_summary.json`, `figures/fig07_*` |
| `08` | mapping of the linear chirp to a binary chirp mass; library of IMRPhenomD signals (4000 train + 2000 held-out); IMR matched-filter bank and oracle; detection of realistic signals by every method; mass dependence | `results/08_summary.json`, `figures/fig08_*` |
| `09` | H1+L1 chunks; Fisher combination of per-detector p-values; time slides ($6.6\cdot10^7$ pairs) down to FAP 10⁻⁶; zero-lag calibration; network-SNR efficiency, coincident against single detector | `results/09_summary.json`, `figures/fig09_*` |
| `10` | goodness-of-fit residual of the best-fit family, maximum softmax, energy score and Mahalanobis distance of the networks; stochastic bursts and IMR signals as out-of-family | `results/10_summary.json`, `figures/fig10_*` |
| `11` | bootstrap of ρ₅₀/ρ₉₀ (resampling background and injections), paired differences between detectors | `results/11_summary.json` |
| `12` | which background sets the threshold: bank ρ₅₀ with train / validation / pooled / test thresholds | `results/12_summary.json` |
| `13` | the three O1 events: IMR / sine-Gaussian bank SNR, best-fit mass, H–L delay, p-values, coincident FAP | `results/13_summary.json` |
| `14` | sliding-window search with trigger clustering; FAR in yr⁻¹ from time slides of the same chunks (2.08 yr); zero-lag 62 vs 63.1 expected; injection recovery | `results/14_summary.json` |
| `15` | ablation of the CNN-vs-bank gap: real vs Gaussian noise (bank 5.38 vs CNN 5.63 in Gaussian), richer bank (no gain), amplitude veto (helps at 1e-3 only) | `results/15_summary.json` |
| `16` | Allen χ² and re-weighted SNR; learned ranking statistics: at FAP 10⁻³ the χ² veto takes the bank from 7.76 to 6.35 (CNN 5.48), no gain at 10⁻² | `results/16_summary.json` |
| `17` | hard-negative mining vs control: mining − control = +0.10 (−0.15, +0.33) at FAP 10⁻², no effect | `results/17_summary.json` |
| `18` | clustering of the 4 % highest-statistic windows: no real structure except a small extreme family (real glitches) that the χ² test and the CNN reject | `results/18_summary.json` |
| `19` | unfolding of the detected total-mass spectrum: inversion unstable, IBU 3–5 iterations best, OmniFold close but biased; bootstrap spread < bias | `results/19_summary.json` |
| `20` | prior mismatch: bank robust, CNN fails at f0 400–500 Hz (ρ₅₀ 8.6 vs 5.1) and on loud short pulses (eff 0.02 at ρ=40); base rate: PPV 0.001 (FAP 10⁻², 1/day) vs 0.90 (FAP 10⁻⁶) | `results/20_summary.json` |
| `21` | labelled glitches: bank/Bayes/CNN alarm on 97 % of blips at ρ=8; glitch-aware Bayes factor cuts blip alarms to 0.35 and whistles to 0.04 at a cost of 0.08 in signal efficiency; χ² is not a glitch veto | `results/21_summary.json` |
| `22` | IMR parameter estimation: bank/posterior ≈7 % error in Mc and 1 ms in time, CNN 10–17 % and 15–85 ms; grid posterior overconfident (68 % interval covers 9–32 %), CNN calibrated (0.65–0.82) | `results/22_summary.json` |
| `23` | background rigour: LOO observed/nominal FAP 0.3–0.9 (chunk 2: 4.3), local thresholds cut the spread 0.013 → 0.001, GPD ξ=0.91 unsafe below 1e-3, FC 90 % limit 13 signals | `results/23_summary.json` |
| `24` | subset simulation: bank threshold agrees with 1−(1−e^{−λ/2})^{N_eff} (N_eff=6794) to 0.1–2 down to FAP 1e-10 | `results/24_summary.json` |
| `25` | CNN robust to PSD estimate and calibration but not to a 60 Hz line (FAP 0.4 % → 17 %) nor FGSM (eff 1.0 → 0.33 at 0.1σ); conformal guarantee fails across epochs; CNN 20× faster than the bank per batch | `results/25_summary.json` |
| `26` | CNN is data-limited (ρ₅₀ 6.5→4.5 from 10³ to 1.6·10⁴ windows, no saturation); flip/shift augmentation no gain; H1-trained CNN transfers to L1 (5.55) while scratch on 2 L1 chunks fails (8.45) | `results/26_summary.json` |
| `27` | bank design: ≥80 templates all equal (fitting factor 0.97→0.996, ρ₅₀ 5.6), random = regular; time-local whitening does not cure epoch dependence; PSD from ≥256 s is negligible, 32 s inflates FAP ×1.5–2.3 | `results/27_summary.json` |
| `28` | h_rss50 ≈ 2.5·10⁻²³ Hz⁻¹ᐟ² (H1, optimal orientation); CNN ≥ bank on unseen shapes except broadband bursts; BBH range 1.3 Gpc (IMR bank) / 1.6 Gpc (CNN with IMR), Euclidean | `results/28_summary.json` |
| `29` | precessing (Pv2) and higher-mode (XPHM) signals cost ≤0.3 in ρ₅₀ for all detectors in the 30–100 M☉ window (fitting factor 0.97 → 0.93) | `results/29_summary.json` |
| `30` | seed scatter of the fresh CNN is 0.40 at FAP 10⁻² (0.03 at 10⁻³); architecture gaps (hybrid +1.0, Transformer +2.0) significant; 5-CNN ensemble 3.79 / 5.32 | `results/30_summary.json` |
| `31` | coherent λ₁ does not beat the Fisher statistic (ρ_net50 10.2 vs 9.4 at FAP 10⁻⁴); with physical antenna patterns ρ_net50 at FAP 10⁻⁶ is ≈15, not 11 | `results/31_summary.json` |
| `32` | exact-IMRPhenomD MCMC calibrated (68 % coverage 0.6–0.8) with 5–8 % error in Mc; two-detector evidence odds no better than min ρ² (AUC 0.85 at norm 8); SBC KS 0.05 (f0, Q), finer grid no help; hierarchical fit recovers (45, 5, 0.30) as (44.8, 4.7, 0.30) while the naive fit gives μ=49.6 | `results/32_summary.json` |
| `33` | noise-only models: VAE local error ρ₅₀ 6.7/6.5/8.5 (sg/pulse/chirp, IMR 7.7), flow 7.5/7.2/9.3; ELBO useless; alarms only 20–26 % shared with the bank | `results/33_summary.json` |
| `34` | ResNet-1D 5.01, TCN 5.09, 2-D CNN 5.20, EdgeConv graph network 5.48 (FAP 1e-2, one seed): none beats the 1-D CNN (4.31 ± 0.40) | `results/34_summary.json` |
| `35` | CAT2 veto: 0.9 % dead time removes 40 % of the loudest bank windows, LOO FAP spread 0.74 → 0.15, threshold at 1e-3 110 → 57; no effect on the CNN | `results/35_summary.json` |
| `36` | O2/O3 transfer: O1 thresholds give FAP 0.9–2.1 % at nominal 1 %; ρ₅₀ unchanged (bank 5.55, CNN 3.7–3.9 with O1 threshold); O3 glitches (kurtosis 10⁵–10⁶) make the bank threshold at 10⁻³ unusable (>16), CNN robust | `results/36_summary.json` |
| `37` | 1 s/30–500 Hz keeps 93–97 % of the SNR for ≥10 M☉ components but 49 % at 3+3 and 8.7 % for a BNS; NSBH costs the IMR bank 5.5 in ρ₅₀ (CNN with IMR: 0.9); eccentricity ≤ 0.9 | `results/37_summary.json` |
| `38` | adding Virgo (O2): V1 median ASD 3.7× H1; 9.6 % of the network SNR²; ideal volume +6.3 % (HLV/HL); no 3-detector search | `results/38_summary.json` |
| `39` | Transformer: depth 6 → 6.23, width 128 → 6.11, 30 epochs → 5.81 (FAP 1e-2; baseline 6.29, CNN 4.31) | `results/39_summary.json` |

## Main results

All SNRs are **optimal SNRs**; the false-alarm probability (FAP) is per one-second window (of H1 alone, except in notebook 09).

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
  It sets the price of a low FAP: lowering it from 10⁻² to 10⁻³ moves ρ₅₀ of the bank from ≈5.5 to ≈11 with the training-background threshold (4.9 → 6.8 with the validation one: notebook 12), and the threshold calibrated on the
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
- **Sequence models and hypothesis testing.** With the same data and budget a patch Transformer (114 k parameters) and a bidirectional GRU are *worse* than the 17.8 k-parameter CNN
  (ρ₅₀ at FAP 10⁻² on sine-Gaussian / pulse / chirp: CNN ensemble 5.05 / 4.21 / 5.28, GRU 6.27 / 5.16 / 6.36, Transformer 6.38 / 4.88 / 7.35) and do not reject glitches better; masked-patch
  pre-training on whitened noise learns almost nothing (it helps only at ≈1000 labels per class). For the morphology, Bayesian model comparison is the best classifier
  (3-way accuracy 0.97 at ρ = 8) among the networks trained on the static set, the CNN is close at high SNR, the Transformer last (but see notebook 07: with fresh injections the CNN ties with it, 0.977 at ρ = 8); no family decision flags a burst that belongs to none of the three families.
- **Tuning the sequence models (notebook 07).** The patch Transformers *under-fit* (training ≈ validation loss ≈ 0.3, against 0.15–0.17 for the CNN) and no single lever fixes them: fresh injections,
  a 3× learning rate, patches of 8 samples and the 25-epoch run all leave mean ρ₅₀ at 6.3–6.6 (FAP 10⁻²). A **convolutional stem** followed by two Transformer blocks is the lever that works
  (5.7 static, 5.4 with fresh injections) but does not overtake the CNN. The biggest effect is on the CNN itself: **fresh injections at every epoch** cut its mean ρ₅₀ from 5.1–5.3 to 4.6 (FAP 10⁻²) and
  from 6.2–6.3 to 5.65 (FAP 10⁻³), better than the 5-member static ensemble of notebook 05 (4.85, 5.79); the CNN of notebooks 05–06 was data-limited. Two seeds for the main variants, ≤ 12 epochs for the sequence models. With fresh injections the 4-class CNN reaches 0.977 three-way accuracy at ρ = 8, as good as or slightly better than the Bayesian model comparison (0.969).
- **Realistic signals (notebook 08).** The sweep rate of the linear chirp corresponds to a binary of 2–30 M☉ but without acceleration or merger. On IMRPhenomD binary-black-hole signals (held-out waveforms, total mass 30–100 M☉)
  the sine-Gaussian bank needs ρ₅₀ = 7.4 (FAP 10⁻²) against 5.4 for an IMR template bank and 3.0 for the oracle; the CNNs trained on the phenomenological morphologies generalise (5.5–6.2), the hybrid Transformer does not (7.6);
  a CNN trained **with** IMR signals reaches **4.45** (efficiency 0.98 at ρ = 8, against 0.91 for the bank), at no cost on the three phenomenological morphologies.
- **Two detectors (notebook 09).** With H1+L1, a Fisher combination of per-detector p-values and time slides the false-alarm floor moves from 10⁻³ to 10⁻⁶ per window; the noises of the two detectors are independent (zero-lag against slides of the same chunks: ratio 0.6–1.1).
  Network-SNR ρ₅₀ at FAP 10⁻² / 10⁻³ / 10⁻⁴ / 10⁻⁵ / 10⁻⁶ on binary-black-hole injections: CNN 4.1 / 6.8 / 8.2 / 9.7 / 11.0, IMR bank 7.4 / 7.6 / 8.5 / 9.7 / 11.0, sine-Gaussian bank 12.4 / 12.9 / 14.3 / 16.6 / 20.4. A plain sum of the statistics is dominated by glitches of one detector and is not used.
- **Open set (notebook 10).** The residual energy after subtracting the best-fit family flags stochastic bursts (AUROC 0.94 at ρ = 8, ≥ 0.998 at ρ ≥ 12); the energy score of the 4-class CNN flags binary-black-hole chirps (0.93 / 0.98 / 0.99 at ρ = 8 / 12 / 16);
  the Mahalanobis distance of the features does not work (AUROC falls with SNR).

## Extensions (notebooks 11–39)

Second round of work, one backlog idea per result (see the table above and `CLAUDE.md` for the numbers):

- **Rigour.** Bootstrap errors on ρ₅₀ and paired tests (nb 11, 30); seed scatter of the fresh CNN is 0.40 at FAP 10⁻² (5 seeds); convention = **validation background** for every thresholded comparison (nb 12); leave-one-chunk-out calibration, time-local background, GPD tail, Feldman–Cousins limits (nb 23); subset simulation to FAP 10⁻¹⁰ (nb 24).
- **Why the CNN wins and where it fails.** Ablation (nb 15), χ² and ranking statistics (nb 16), hard negatives and glitch families (nb 17–18), prior mismatch and base rate (nb 20), labelled glitches and glitch-aware Bayes factor (nb 21), robustness, conformal sets, saliency, latency, FGSM (nb 25), scaling/augmentation/H1→L1 transfer (nb 26), other architectures (nb 34), Transformer tuning (nb 39), noise-only flow and VAE (nb 33).
- **Physics and data.** Real events (nb 13), continuous search (nb 14), physical units and range (nb 28), precession/higher modes (nb 29), limits for low masses/BNS/NSBH/eccentricity (nb 37), coherent statistic with physical network injections and 10 background chunks (nb 31), Virgo SNR budget (nb 38), GWOSC data-quality flags (nb 35), O2 and O3 (nb 36).
- **Parameters and populations.** IMR parameter estimation with CNN regression, grid posterior and exact-model MCMC (nb 22, 32), unfolding and hierarchical population inference (nb 19, 32), SBC (nb 32), bank design, local whitening and PSD uncertainty (nb 27).
- **Negative or limited results worth keeping:** hard-negative mining, flip/shift augmentation, coherent λ₁ without sky scan, time-local whitening, a denser bank, and extra Transformer depth/width give no measurable gain; the CNN is brittle outside its training ranges.

## Limits

- **Detectors.** Most notebooks use H1 alone; notebooks 09, 14 and 31 add L1 (incoherent combination and a peak-based coherent statistic, no sky scan, no χ² vetoes; 6–12 chunks), notebook 38 only an SNR budget for Virgo. FAPs are per window, not astrophysical rates.
- **Injections at a given SNR.** Converted into distances only in notebook 28 (Euclidean, single H1); the three morphologies are phenomenological; the main IMRPhenomD library has aligned spins, total masses 30–100 M☉; precessing, higher-mode, NSBH and eccentric signals are tested in notebooks 29 and 37, BNS and masses < 10 M☉ are out of reach of one-second windows.
- **Background statistics.** 1.6·10⁴ training windows limit the lowest usable FAP to ~10⁻³; the 10⁻³ efficiencies are uncertain.
- **Eight chunks.** The epoch dependence is measured with two test chunks.
- **Grid resolution.** The Bayesian grid under-resolves f₀ and t₀ for ρ ≳ 15 (SBC coverage of those two parameters is 0.50–0.60, not the model).
- **ML numbers** are efficiencies for the training injection prior (the CNN fails outside it, notebook 20); most variants have one seed and the seed scatter at FAP 10⁻² is 0.4.

## Conventions

Notebooks, comments and documentation are in English. Every notebook states at the beginning what it does and what it does **not** do, and ends
with the output produced and the next notebook. `cache/` is a regenerable cache; `results/` is the source of the numbers quoted here.

## License

Released under the [MIT License](LICENSE). The LIGO strain data are public (GWOSC) and are subject to their own
[data-use terms](https://gwosc.org/terms/).
