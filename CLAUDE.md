# GWOSC

Burst-signal detection in public LIGO data (GWOSC, O1, H1) — University of Milano-Bicocca, *Statistics and Data Analysis*
(A.Y. 2025–2026). Scientific goal: detect short, unmodelled gravitational-wave bursts (sine-Gaussian, Gaussian pulse,
linear chirp) injected into **real** H1 noise with three families of methods — frequentist (matched-filter bank,
profile likelihood), Bayesian (exact grid posterior, MCMC, Bayes factors) and machine learning (feature models, CNN,
autoencoder) — and compare them on the same data at the same false-alarm probability.

## Repository layout

```
Code_General/
    00_waveform_exploration.ipynb          exploratory prototype (GW150914, PSD, whitening, Q-transform, first injections)
    01_data_whitening_validation.ipynb     stage 1: 8 O1 chunks, PSD, whitening, 1 s windows, Gaussianity/stationarity tests, split
    02_waveforms_and_injections.ipynb      stage 2: exact analytic burst transforms, optimal SNR, injection-chain validation
    03_frequentist_analysis.ipynb          stage 3: matched-filter bank, calibration, efficiency/ROC, MLE+Fisher, toy MC, systematics
    04_bayesian_analysis.ipynb             stage 4: grid posterior + emcee, Bayes factors, prior dependence, SBC, credible vs confidence
    05_ml_detection.ipynb                  stage 5: features+Ridge/Lasso/RF/GBT, 1-D CNN, ensemble, MC dropout, autoencoder, calibration
    06_sequence_models_hypothesis_testing.ipynb   stage 6: Transformer/GRU detectors, masked-patch pre-training, 4-class morphology vs frequentist/Bayesian model comparison
    07_tuning_sequence_models.ipynb        stage 7: stress test of the Transformer-vs-CNN result (fresh injections, lr, patch size, conv stem)
    08_realistic_inspiral_signals.ipynb    stage 8: IMRPhenomD (binary black hole) injections, IMR template bank, chirp-mass mapping of the linear chirp, detection of realistic signals
    09_two_detector_coincidence.ipynb      stage 9: H1+L1 coincidence, time slides down to FAP 1e-6, zero-lag check
    10_open_set_goodness_of_fit.ipynb      stage 10: flagging out-of-family signals (matched-filter residual, network open-set scores)
    11_bootstrap_error_bars.ipynb          extension: bootstrap errors on rho50/rho90, paired tests (stats of nb 06-07)
    12_threshold_convention.ipynb          extension: bank rho50 vs background used for the threshold (train/val/pooled/test)
    13_real_events ... 39_transformer_tuning_round2   second-round extensions (see the "Notebooks 11-12" section below: each notebook has its own entry, bold number = notebook). New helpers:
    glitch_tools.py, pe_tools.py, network_tools.py, precession_tools.py, make_summary_figures.py, make_checksums.py, run_all.sh, tests/test_gwosc_tools.py (12 tests);
    downloads: download_events.py, download_coinc_extra.py, download_runs.py, download_virgo.py (results/events_chunks.json, coinc_chunks_extra.json, runs_chunks.json, virgo_chunk.json)
    seq_models.py                          models, dataset and resumable training loop of notebooks 07-08 (CLI: python seq_models.py <variant> ... trains in parallel processes)
    imr_tools.py                           IMRPhenomD libraries (train "imr" and held-out "imr_test") and IMRBank; needs lalsuite only to build (python imr_tools.py)
    open_set.py                            FamilyBank fits and goodness-of-fit residual of notebook 10
    gwosc_tools.py                         functions shared by notebooks 01-10 (whitening, waveforms, injection, SGBank, statistics, imr_library, load_coinc_chunk)
    download_data.py                       downloads the 8 raw chunks into cache/raw/ and writes results/chunks.json
    download_coinc_data.py                 downloads the 6 H1+L1 chunks of notebook 09 (parallel, with retries) and writes results/coinc_chunks.json
    results/                               light results TRACKED by git (summary JSON of every notebook)
    figures/                               figures of every notebook (PNG + PDF), tracked
    cache/                                 raw strain, whitened windows, scans, trained models (git-ignored, regenerable)
Docs/report/                               LaTeX report (main.tex, biblio.bib, figures/ copied from Code_General/figures)
environment.yml                            conda environment (Jupyter kernel used: `cern`)
README.md                                  public description, main results and limits
```

Interpreter: the conda env `cern` of `environment.yml` (on the author's laptop it no longer exists: the working env is `/opt/homebrew/Caskroom/miniforge/base/envs/vulcanus/bin/python`, and the registered kernel `cern` points to the missing one, so run nbconvert with `--ExecutePreprocessor.kernel_name=python3` and reset the notebook kernelspec to `cern` afterwards). The project is a pipeline of Jupyter notebooks
run in order; there are no build/test scripts. Run a notebook with
`jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=cern <nb>.ipynb` from `Code_General/`.
When running python scripts, add `Code_General` to `sys.path` (do not rely on the working directory with `python -I`).

## Data

Eight 4096 s chunks of H1 strain (4096 Hz) from O1, inside H1 science segments and ≥ 2000 s away from every catalogued event.
`download_data.py` selects them (evenly spread over the run) and stores them as float32 `.npy`. The split is **by chunk**,
interleaved in time (`gwosc_tools.SPLIT`): train 0,2,4,6 · val 1,5 · test 3,7. Never split by window: neighbouring windows share noise.

## Conventions of the pipeline (important)

- Analysis unit: **1 s window**, 1024 samples at 1024 Hz, band 30–500 Hz, whitened with the median-Welch PSD of its own chunk.
- Frequency bins `V_k = rfft(v)_k / sqrt(512)`: in the band Re and Im are N(0,1) for Gaussian noise; the optimal SNR is
  `rho^2 = sum_k |h_k|^2`. All SNRs in the project are optimal SNRs; signals are injected **in the frequency domain** as
  `V_k = 2 h(f_k)/sqrt(S_n(f_k))` (exact, no filter transients).
- Detection statistic (frequentist) = `max rho^2` over a 160-template sine-Gaussian bank × all time shifts; Bayesian
  statistic = prior-weighted average of the same cube (analytic amplitude marginalisation, `sigma_rho = 10`).
- False-alarm probabilities are per 1 s window of single-detector data, **not** astrophysical rates. H1 only.
- The real noise is non-Gaussian (glitches) and non-stationary (epoch 2 is very glitchy): thresholds are always set on real
  background, and calibration is checked on chunks never used for the threshold.

## Code conventions

- Notebooks, comments and documentation are in **English**; keep this language for consistency.
- Every final notebook starts with a markdown cell that states explicitly what it does and what it does **not** do, and ends
  with a cell describing the output produced and the next expected notebook — keep this pattern in new notebooks.
- Light results go in `Code_General/results/` (tracked); anything big or regenerable stays in `cache/`.
- All random numbers use fixed seeds. Long blocks cache their output in `cache/` or `results/` and resume on re-run.
- `cache/` is entirely regenerable (`download_data.py`, `download_coinc_data.py`, `python imr_tools.py`, then notebooks 01 → 10): treat it as a cache, not a source of truth.

## Notebook 06 (done): what it found, and caveats

- Transformer encoder (patches of 16 samples, 114 k parameters), GRU baseline (88 k) and CNN (17.8 k) on the same split and injections. **The CNN wins**: at FAP 1e-2 the CNN ensemble has rho50 = 5.05 / 4.21 / 5.28 (sg / pulse / chirp), the Transformer 6.38 / 4.88 / 7.35, the GRU 6.27 / 5.16 / 6.36; the gap grows at FAP 1e-3. Pre-training (masked patches on noise) learned almost nothing (error ratio 0.995 vs the zero predictor), helps at 1000 labels/class (AUC 0.76 vs 0.54) and not with the full set.
- Transformers/GRU were trained for at most 12 epochs (CPU budget, 2 seeds), not tuned (depth, width, patch size): conclusions are for this budget. One 25-epoch seed gave the same validation loss as the 12-epoch one.
- Multiclass: Bayesian model comparison (3 families, sigma_rho = 10) is the best morphology classifier (3-way accuracy 0.969 at rho = 8); CNN 4-class close at high SNR, GRU next, Transformer last. Frequentist argmax of rho^2 is 0.05 worse; the p-value rule saturates at 1/(N+1) of the validation background. No method can flag a signal outside the three families (stochastic burst is called sine-Gaussian).
- Everything is cached in `cache/models/` (weights) and `cache/stats06.pkl`; delete the pickle if models change.

Possible follow-ups: tune the Transformer (smaller patches, convolutional stem, more data), an open-set / goodness-of-fit head for unknown morphologies, more background (more chunks) for FAP < 1e-3.

## Notebook 07 (done): tuning of the sequence models

- Diagnosis: the patch Transformers/GRU of notebook 06 *under-fit* (train ≈ val loss ≈ 0.3); the CNN with static data over-fits (train 0.07–0.09, val 0.17–0.19) and fresh injections cure it (val 0.117). Models are cached as `cache/models/v07_*`, logits in `cache/stats07.pkl`.
- One lever at a time, mean rho50 over the 3 morphologies at FAP 1e-2 / 1e-3 (test injections identical to 05-06, thresholds on validation noise): Transformer nb06 6.20/7.32; fresh injections 6.29/7.25; lr 3e-3 6.36/7.68; patch 8 6.59/7.58 (4x cost); **conv stem + 2 blocks 5.66/6.60 (static), 5.36/6.14 (fresh)**; CNN static 5.1-5.3/6.2-6.3; **CNN with fresh injections 4.6/5.65** (a single net beats the 5-member static ensemble of nb05: 4.85/5.79). Seed-to-seed scatter of the mean rho50: 0.23/0.14 (static CNN), 0.06/0.02 (fresh CNN); most variants have one seed.
- **Use the fresh-injection CNN as the reference ML detector in any future comparison.** Notebooks 05-06 are unchanged (their CNN is the data-limited one).
- Training budget: 12 epochs for the sequence models (23-81 min each on a laptop on battery), 20 for the CNN (17 min); 4 parallel processes ~1.5 h. A Mac on battery power runs ~3x slower than the isolated timing; the M4 has only 4 fast cores, so >4 processes do not help.
- Possible follow-ups: second seed for the hybrid/Transformer, fresh injections for the multiclass networks of 06, longer windows (where attention may matter).

## Notebooks 08-10 (done): realistic signals, two detectors, open set

- **08.** The linear chirp mimics a binary of Mc ~ 2-30 Msun only locally (no acceleration, no merger). IMRPhenomD library: total mass 30-100 Msun, aligned spins; **always evaluate on the held-out library (injection kind `"imr_test"`)**, train on `"imr"` (`g.MORPHS_ALL`). rho50 (FAP 1e-2 / 1e-3, val thresholds): oracle 3.00/3.50, **CNN trained with IMR 4.45/5.91**, IMR bank (24 templates, fitting factor 0.93) 5.40/6.42, CNN fresh no IMR 5.52/7.16, sine-Gaussian bank (FF 0.65) 7.38/11.05, hybrid fresh 7.55/9.41, excess power 8.28/10.16. No cost on the 3 morphologies. Mass dependence: burst methods worst at 30-45 Msun.
- **09.** 6 H1+L1 chunks (bkg 0,1,3,4; test 2,5). Coincident statistic = Fisher combination -ln p_H - ln p_L of per-detector empirical p-values (a plain or capped sum is dominated by H1 glitches: do not use); banks also require |tau_H - tau_L| <= 15 ms. Time slides give 6.6e7 pairs -> FAP 1e-6. Zero-lag vs slides of the **same** chunks ratio 0.6-1.1 (independent noises); vs slides of other epochs 1.3-5 (epoch dependence). Network-SNR rho50 at FAP 1e-2..1e-6 (IMR injections): CNN 4.1/6.8/8.2/9.7/11.0, IMR bank 7.4/7.6/8.5/9.7/11.0, SG bank 12.4/12.9/14.3/16.6/20.4; single H1 cannot go below 1e-3.
- **10.** Out-of-family signals (stochastic burst, IMR) vs in-family, detected signals only. Matched-filter residual (+-60 ms around the best fit) AUROC 0.935/0.998/1.000 (rho 8/12/16) for stochastic bursts; energy score of the 4-class CNN best for IMR (0.93/0.98/0.99); Mahalanobis on penultimate features fails (AUROC falls with SNR).
- **Correction of notebook 06 (found in 07):** 4-class networks trained with fresh injections: CNN 0.977 three-way accuracy at rho=8 vs Bayes 0.969 (ties at low SNR). "Bayes is the best classifier" holds for the static-data networks only. nb07 also has 2 seeds for hybrid/Transformer-fresh variants (hybrid 5.64/6.79 static, 5.33/6.21 fresh).
- Environment: `lalsuite` was pip-installed into the `vulcanus` env (added to environment.yml). Training on AC power is ~3x faster than on battery.

Possible follow-ups: coherent two-detector statistic with chi^2 vetoes and sky localisation, precessing/higher-mode waveforms and lower masses with longer windows, an explicit unknown class in the networks, a seed sweep and depth/width sweep for the Transformer.

## Notebooks 11-12 (done): rigour and threshold convention

- **11.** Bootstrap (background + injections) on cached logits of nb06-07: 1-sigma half-width of mean rho50 0.05-0.11; fresh CNN beats static CNN by 0.76 (FAP 1e-2) and Transformer loses to fresh CNN by 1.71 (all significant); different seeds of one configuration are not distinguishable. Curves never reaching 50% are censored at SNR 16. Helpers `bootstrap_rho`, `snr_level`, `ci` in gwosc_tools.
- **12.** **Convention: every cross-method comparison uses the VALIDATION background for thresholds** (training-background numbers only in the calibration discussion). Bank sg rho50 at FAP 1e-2/1e-3: train 5.49/10.78, val 4.94/6.82, test 4.89/14.2 (large epoch dependence at 1e-3; glitchy chunk 2 in train). Report has the new Sec. "Which background sets the threshold?", an executive summary + glossary (`sections/summary.tex`) and fig13 (summary rho50, pipeline) made by `make_summary_figures.py`. The report compiles with `tectonic -X compile main.tex` from `Docs/report` (tectonic is in the `vulcanus` conda env; 59 pages).
- **13.** Real events (H1+L1, GWOSC files have NaN gaps: use the common clean stretch). IMR bank: GW150914 rho 19.2/13.3 (net 23.4, M 72, dt 5.9 ms, loudest window of its chunk), GW151012 net 9.1 (FAP~2e-4 coincident, IMR bank only; sg bank misses it), GW151226 NOT recovered (outside bank, only 540 s clean). `download_events.py` -> `results/events_chunks.json`.
- **14.** Mini continuous search (H1+L1 test chunks 2,5; 0.5 s step; IMR bank; Fisher of nb13 p-values; clustering +-1 s; slides of the same chunks = 2.08 yr). Zero-lag 62 vs 63.1 expected: no foreground. Loudest S=9.8, FAR ~7e3/yr, made of H1 glitches. Injections: rho_net 15 recovered at FAR<1e3/yr 100%, never below 1/yr. **Slide bug to remember: paired windows are aligned, gate on |tau_H - tau_L(shifted index)| only (wrapped pairs were wrongly rejected, background underestimated x4).**
- **15.** Ablation (mean rho50 sg/pulse/chirp, val thresholds, FAP 1e-2/1e-3): bank160 real 5.58/7.71; bank640 5.54/7.61 (richer bank does NOT help); bank+amplitude veto 5.66/6.87 (kills loud pulses); CNN fresh real 4.50/5.48; Gaussian noise: bank 5.38/6.00, CNN (not retrained) 5.63/6.30 -> the CNN advantage lives in real-noise tails, not in detection power per se.
- **16.** chi2 (8 bands, `allen_chi2`, `new_snr`, `template_index` in gwosc_tools) + ranking stats (train on train chunks, val thresholds, test injections). Mean rho50 FAP 1e-2/1e-3: bank 5.51/7.76; chi2 veto c<=3 5.44/6.35; new SNR 5.63/6.72; logistic 5.54/6.46; GBT 5.38/6.34; CNN fresh 4.50/5.48. Gain only at 1e-3; learned combination no better than the chi2 veto.
- **Tests/repro (44).** `Code_General/tests/test_gwosc_tools.py` (19 tests incl. `tests/test_extensions.py`, `python -m pytest Code_General/tests -q`; pytest is not in the vulcanus env: install with `pip install --target <dir> pytest` and set PYTHONPATH), `run_all.sh`, `make_checksums.py` -> `results/data_checksums.json`.
- **17.** Hard-negative mining (top 10% training noise x5, 8 epochs) vs control: mean rho50 4.46/5.84 vs 4.37/5.80; difference +0.10 (-0.15,+0.33) / +0.04: NO effect.
- **18.** Glitch families (4% highest bank stat of val+test, spectrogram+PCA+kmeans): silhouette 0.08 -> no real classes; extreme family (20 windows) = real glitches: bank alarms 95%, chi2<3 removes ~95% of them, CNN alarms 5%. Outlier handling matters: without removing the extreme 3% silhouette picks singleton clusters.
- **19.** Unfolding (sim: train IMR lib, uniform M, rho^-4; pseudo-data: test lib with peak at 45): chi2 vs truth naive 1216, inversion 4862 (negative bins), IBU 3/5/10 it. 199/141/268, OmniFold 468 (bias in the 55-65 bin). Bootstrap std (3%) < bias (11-19%): prior dependence dominates. New report section `unfolding.tex`.
- **20.** Prior mismatch: bank robust (rho50<=7.0, eff 1.0 at rho 40 everywhere); CNN brittle: sg f0 400-500 rho50 8.6 vs 5.1, pulses 5-15 ms 9.3 vs 6.7, eff at rho=40 only 0.02 for 0.3-0.8 ms pulses (input clip at +-12, training SNR<=15). **The CNN advantage holds only inside the training injection prior.** Base rate (illustrative 1/day): PPV 0.001 / 0.01 / 0.90 for FAP 1e-2 / 1e-3 / 1e-6.
- **21.** `glitch_tools.py` (blip/whistle/arch + GlitchBank 51 templates). At norm 8, alarm fraction (bank / Bayes / robust Bayes / CNN): blip .98/.97/.35/.98, whistle .47/.38/.04/.52, arch .52/.52/.19/.53; signals (sg/pulse/chirp) mean .95/.96/.88/.98. chi2<3 is NOT a glitch veto (also rejects loud signals). Robust BF (eta=0.1) barely changes B>100 on REAL noise (0.17%->0.15%): real false alarms are not these families. Upper bound (model knows the families).
- **22.** IMR PE (203-template non-spinning bank; grid posterior; CNN Gaussian head, max-pool vs flatten). Median |dMc|/Mc: bank 0.07, CNN 0.09-0.17; merger time 1 ms (bank) vs 15-85 ms (CNN; max-pool head has no position). Grid posterior 68% coverage only 0.09-0.32 (model mismatch + grid); CNN 0.65-0.82. New report section `parameters.tex`. Weights `cache/models/v22_*`.
- **23.** LOO chunk calibration: observed/nominal FAP 0.3-0.9 except chunk 2 (4.3/4.9). Local threshold (+-150-600 s): spread across chunks 0.013 -> ~0.001. GPD xi=0.91: extrapolation below 1e-3 unsafe (underestimates x9 at 1e-4). FC 90% UL for n=62,b=63.1: 13.0 signals. Items done so far: 1,3,4,7,8,9,18,24,25,29,33,39,44,49,52,55,56,58,59,63-66.
- **24.** Subset simulation (pCN, p0=0.1, 4 runs): Gaussian tail of the bank agrees with the N_eff=6794 formula to <=0.6 (FAP>=1e-6) and <=2 (1e-10). Item 51 done.
- **25.** Robustness of the fresh CNN: PSD estimate / amplitude +-10% benign; **60 Hz line norm 3 -> CNN FAP 17% (bank unchanged)**; FGSM eps 0.1 sigma drops eff(rho=10) 1.0 -> 0.33; conformal detection threshold ratio 0.5-1.9 (CNN) / 0.5-3.5 (bank) across epochs; 4-class conformal sets size ~1, conditional coverage depends on SNR; saliency mass in +-4sigma_t 0.30 vs 0.10 uniform; latency CNN 0.17 ms/window (batch) vs bank 3.5 ms. Bank scans only lag 0.15-0.85 s: edge injections invisible to it by design.
- **26.** Scaling (mean rho50 FAP1e-2/1e-3): static n=1000/4000/16224: 6.70/6.16/5.34 | 9.61/7.39/6.48; fresh: 6.53/5.83/4.49 | 9.33/7.11/5.67 -> data-limited, no saturation. Augmentation (flip+shift) no gain at 16k; label noise 5% harmless, 20% +0.4/+1.4. H1->L1 zero-shot 5.55/6.97, fine-tune 5.49, scratch on 2 L1 chunks 8.45. Items done: + 28, 60, 62.
- **27.** Bank design: FF 0.78/0.91/0.97/0.985/0.996 for 20/40/80/160/640 templates; rho50(1e-2) 6.24/5.80/5.61/5.57/5.57; random placement as good as regular. Local (512 s block) whitening: no change in epoch dependence. PSD from T s: FAP 1.46e-2 (32 s), 1.1e-2 (64), <=0.96e-2 (>=256). Items 40, 43, 57 done. NOTE: raw strain is float32 -> `welch_psd` on float32 underflows to 0 (cast to float64 first).
- **28.** `one_waveform` amplitude needs x FS_GEN (=2048) to be physical (library is normalised anyway). h_rss50 (1e-22) sg Q=9: bank 0.27-0.38, CNN 0.25-0.38; cusp/ringdown: CNN better than bank (rho50 3.8-5.0 vs 5.1-5.8); wide white-noise burst: CNN fails. BBH Euclidean range (FAP 1e-2/1e-3): IMR bank 1.31/1.10 Gpc, CNN+IMR 1.57/1.26 Gpc; VT 9.3/16.3 Gpc^3 yr. Items 14, 15, 16 done. New report section `physical.tex`.
- **29.** Precessing (Pv2, |chi|<=0.8) and HM (XPHM) vs aligned PhenomD, 600 each, networks NOT retrained: rho50 change <=0.3 (IMR bank 5.30/5.57/5.59; CNN+IMR 4.44/4.44/4.52); bank FF median 0.971/0.937/0.931. `precession_tools.py`, libraries `cache/lib_*.npz`. Item 10 done.
- **30.** Seeds: CNN fresh x5 FAP1e-2 4.31+-0.40 (3.72-4.63; **earlier 2-seed scatter 0.06 was too optimistic**), 1e-3 5.67+-0.03; hybrid x3 5.35+-0.05/6.29+-0.16; Transformer x3 6.29+-0.05/7.47+-0.20; 5-CNN logit ensemble 3.79/5.32. Arch gaps significant (perm p=0.018). Models cache `v07_*_s2..4`. Item 2 done.
- **31.** 10 bkg chunks (4 old + 6 new `x0..x5` from `download_coinc_extra.py`), 1.64e8 slide pairs. Raw lambda_1 / incoherent sums are dominated by single huge glitches (threshold 3e8): equalise |z|^2 -> -2 ln p per detector first. Network rho50 FAP 1e-4: Fisher 9.36, incoherent 9.77, coherent 10.20; FAP 1e-6: only Fisher reaches 50% (15.0). Physical antenna patterns => ~15 not 11 (nb09 injections optimistic). Items 17, 19, 21, 22, 37 done (22/37 only in this reduced form). `precession_tools.waveform_pol` gives hp, hc.
- **32.** emcee with the EXACT PhenomD model (analytic complex amplitude: 1 ms/likelihood; `pe_tools.py`): Mc err 4.7/6.3/6.8% (rho 8/12/16), coverage68 0.6-0.8 (30 inj); grid posterior of nb22 was overconfident. Two-detector Bayes odds C-vs-G: AUC 0.85 (norm 8) = min rho^2; profile odds NOT calibrated on noise (8% > ln100). SBC: KS 0.05-0.06 f0,Q; tau rank bias (mean 0.61); 640 grid no better. Population: hierarchical with response+selection recovers (45,5,0.30) -> (44.8,4.7,0.30); naive mu=49.6. Items 35, 36, 38, 54 done.
- **33.** RealNVP on 16 hand-built features + conv VAE (latent 16), noise only. rho50 FAP 1e-2 sg/pulse/chirp/imr: flow 7.45/7.24/9.25/9.45; VAE local 6.74/6.52/8.50/7.69; VAE ELBO ~12 (useless). 20-26% of bank alarms also flagged. Items 31, 53 done.
- **35.** GWOSC H1_CBC_CAT2 flag (gwosc.timeline.get_segments works): 0.9% dead time (7.4% chunk 2) removes ~40% of the top-1% bank windows; LOO FAP ratio range 0.30-4.3 -> 0.76-1.21 at 1e-2; pooled threshold at 1e-3: 110 -> 57. CNN unaffected. Item 42 done. USE CAT2 in any template-statistic search.
- **36.** O2/O3 (3 chunks each of O2, O3a, O3b; `download_runs.py`, `results/runs_chunks.json`): ASD x0.91/0.65/0.61 of O1; O1 thresholds give observed FAP 0.9-2.1% at nominal 1% (bank 1.25/1.64/0.88%, CNN 1.6/2.1/1.5%); rho50 with O1 threshold unchanged (bank 5.55, CNN 3.7-3.9); own-run 1e-3 threshold of the bank in O3 > 16 (giant glitches: stat up to 7e5). Item 41 done.
- **34.** ResNet-1D 5.01/5.77, TCN 5.09/5.87, 2-D CNN (spectrogram) 5.20/5.97, EdgeConv GNN on 64 pixels 5.48/6.46 (1 seed each, 12 epochs); none beats 1-D CNN (4.31+-0.40/5.67). Items 26, 50 done.
- **37.** Limits: SNR retained (band + last 1 s): 3+3 Msun 0.49, 5+5 0.73, 7.5+7.5 0.87, 10+10 0.93, >=15 0.96-0.97; BNS 0.087 (undetectable here). NSBH: IMR bank rho50 10.8 vs 5.3 control; CNN+IMR 5.44 vs 4.56. Eccentric (EccentricFD, inspiral only) <=0.9 loss for IMR bank. Items 11, 12, 13 done (analysis of limits, not new 4-8 s / 16 kHz pipelines). The `duration_band_s` column of the retention table is an FFT-wrap artefact for M>=30: do not use.
- **38.** Virgo (reduced: SNR budget only, `download_virgo.py`, chunk 1186897920 H1/L1/V1): V1 median ASD 3.7x H1, 9.6% of network SNR^2, ideal volume HLV/HL = 1.063. Item 20 done in reduced form (no three-detector background/search).
- **39.** Transformer tuning round 2 (1 seed): depth 6: 6.23/7.21; width 128: 6.11/6.97; 30 epochs: 5.81/7.15 (val loss 0.255) vs baseline 6.29/7.47, CNN 4.31/5.67. Not run: contrastive pretraining, overlapping windows. Item 27 done (partly).
- Backlog of extensions (memory) is being done in order; slides come last.

## Known limits to keep in the conclusions

(Updated after notebooks 11-39: see their entries above; limits below still hold unless an entry says otherwise.)
H1 alone except in notebooks 09/14/31 (two-detector, incoherent or peak-based coherent, no sky scan, no chi^2 vetoes); injections are scaled to a *given
optimal SNR*, not placed at physical distances; a single detector's ~16 000 background windows limit its lowest usable FAP to ~10⁻³
(time slides reach 10⁻⁶ per window, still not an astrophysical rate); the three morphologies are phenomenological, the IMRPhenomD
library is aligned-spin only with total masses 30-100 Msun; ML numbers are efficiencies *for the training injection prior*.
