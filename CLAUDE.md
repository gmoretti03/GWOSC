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
    gwosc_tools.py                         functions shared by notebooks 01-05 (whitening, waveforms, injection, SGBank, statistics)
    download_data.py                       downloads the 8 raw chunks into cache/raw/ and writes results/chunks.json
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
- `cache/` is entirely regenerable (`download_data.py`, then notebooks 01 → 06): treat it as a cache, not a source of truth.

## Notebook 06 (done): what it found, and caveats

- Transformer encoder (patches of 16 samples, 114 k parameters), GRU baseline (88 k) and CNN (17.8 k) on the same split and injections. **The CNN wins**: at FAP 1e-2 the CNN ensemble has rho50 = 5.05 / 4.21 / 5.28 (sg / pulse / chirp), the Transformer 6.38 / 4.88 / 7.35, the GRU 6.27 / 5.16 / 6.36; the gap grows at FAP 1e-3. Pre-training (masked patches on noise) learned almost nothing (error ratio 0.995 vs the zero predictor), helps at 1000 labels/class (AUC 0.76 vs 0.54) and not with the full set.
- Transformers/GRU were trained for at most 12 epochs (CPU budget, 2 seeds), not tuned (depth, width, patch size): conclusions are for this budget. One 25-epoch seed gave the same validation loss as the 12-epoch one.
- Multiclass: Bayesian model comparison (3 families, sigma_rho = 10) is the best morphology classifier (3-way accuracy 0.969 at rho = 8); CNN 4-class close at high SNR, GRU next, Transformer last. Frequentist argmax of rho^2 is 0.05 worse; the p-value rule saturates at 1/(N+1) of the validation background. No method can flag a signal outside the three families (stochastic burst is called sine-Gaussian).
- Everything is cached in `cache/models/` (weights) and `cache/stats06.pkl`; delete the pickle if models change.

Possible follow-ups: tune the Transformer (smaller patches, convolutional stem, more data), an open-set / goodness-of-fit head for unknown morphologies, more background (more chunks) for FAP < 1e-3.

## Known limits to keep in the conclusions

Single detector (no coincidence, no time slides); injections are scaled to a *given optimal SNR*, not placed at physical
distances; background of ~16 000 windows limits the lowest usable FAP to ~10⁻³; the three morphologies are phenomenological
burst models, not astrophysical waveforms; ML numbers are efficiencies *for the training injection prior*.
