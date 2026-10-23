"""Shared code of the tuning study of the sequence models (notebook 07).

Contents: the dataset of notebooks 05-06 (same seeds), the model builders (patch transformer with a configurable patch size,
hybrid convolutional stem + transformer, 1-D CNN of notebook 05), a resumable training loop with optional *fresh injections at
every epoch* and the configuration of every variant of the study.

Run from the command line to train a list of variants (used to train in parallel processes):
    python seq_models.py hyb_static hyb_fresh ...
Everything is cached in cache/models/ (weights + history JSON) and resumes if the files exist.
"""
import json
import math
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gwosc_tools as g

SEED = 2026
SNR_RANGE = (3.0, 15.0)
KID = {"sg": 1, "pulse": 2, "chirp": 3}
MODELS = os.path.join(g.CACHE, "models")
os.makedirs(MODELS, exist_ok=True)
D = 64


def clip(a):
    return np.clip(a, -12, 12).astype(np.float32)


def make_signals(v, S1_, morph_pool, r, snr_range=SNR_RANGE):
    n = len(v); kinds = r.choice(morph_pool, n); snr = r.uniform(*snr_range, n)
    out = np.empty_like(v)
    for k in morph_pool:
        m = kinds == k
        if m.any():
            out[m] = g.inject(v[m], k, snr[m], r, S1_[m])[0]
    return out, kinds, snr


# --------------------------------------------------------------------------------------------------------------
# data (identical to notebooks 05/06: same seeds, same injection prior)
# --------------------------------------------------------------------------------------------------------------
def load_data():
    CH = {i: g.load_chunk(i) for i in range(len(g.chunks()))}
    data = {sp: g.load_split(sp)[:2] for sp in ("train", "val", "test")}
    S1 = {sp: np.stack([CH[i]["S1"] for i in data[sp][1]]) for sp in data}
    DS = {}
    for sp in ("train", "val"):
        v, cid = data[sp]
        sig, kinds, snr = make_signals(v, S1[sp], list(g.MORPHS), np.random.default_rng(SEED + len(sp)))
        DS[sp] = dict(noise=v, sig=sig, snr=snr)
        DS[sp]["y4"] = np.r_[np.zeros(len(v), int), [KID[k] for k in kinds]]
        DS[sp]["X"] = np.concatenate([clip(v), clip(sig)])[..., None]
        DS[sp]["yb"] = np.r_[np.zeros(len(v)), np.ones(len(v))].astype(np.float32)
    return CH, data, S1, DS


SNRS = [3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16]
NINJ = 500


def make_test_sets(data, S1):
    """The test injections of notebooks 05/06 (seed 777): {(morphology, snr): (series, chunk ids)}."""
    vte, cte = data["test"]; S1te = S1["test"]; r = np.random.default_rng(777); TEST = {}
    for kind in g.MORPHS:
        for snr in SNRS:
            idx = r.choice(len(vte), NINJ, replace=False)
            TEST[(kind, snr)] = (g.inject(vte[idx], kind, float(snr), r, S1te[idx])[0], cte[idx])
    return TEST


# --------------------------------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------------------------------
def _tf():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers as L
    return tf, keras, L


def _pos_layer():
    tf, keras, L = _tf()

    class Pos(L.Layer):
        def build(self, s):
            self.p = self.add_weight(shape=s[1:], initializer=keras.initializers.RandomNormal(stddev=0.02), name="pos")

        def call(self, x):
            return x + self.p
    return Pos


def _block(h, d, heads, mlp, drop):
    tf, keras, L = _tf()
    a = L.LayerNormalization()(h); a = L.MultiHeadAttention(heads, d // heads, dropout=drop)(a, a); h = h + L.Dropout(drop)(a)
    f = L.LayerNormalization()(h); f = L.Dense(d)(L.Dense(mlp, activation="gelu")(f)); return h + L.Dropout(drop)(f)


def _head(h, drop=0.1, nout=1):
    tf, keras, L = _tf()
    x = L.Concatenate()([L.GlobalMaxPooling1D()(h), L.GlobalAveragePooling1D()(h)])
    x = L.Dropout(drop)(L.Dense(64, activation="gelu")(x)); return L.Dense(nout)(x)


def build_tf(patch=16, d=D, depth=3, heads=4, mlp=128, seed=0, nout=1):
    """Patch transformer (as notebook 06, with a configurable patch size)."""
    tf, keras, L = _tf(); tf.keras.utils.set_random_seed(seed)
    inp = L.Input((g.NWIN, 1)); h = L.Conv1D(d, patch, strides=patch)(inp); h = _pos_layer()()(h)
    for _ in range(depth): h = _block(h, d, heads, mlp, 0.1)
    return keras.Model(inp, _head(L.LayerNormalization()(h), nout=nout))


def build_hybrid(d=D, depth=2, heads=4, mlp=128, seed=0, nout=1):
    """Convolutional stem (the backbone of the CNN of notebook 05, 8x down-sampling) + transformer on 128 tokens."""
    tf, keras, L = _tf(); tf.keras.utils.set_random_seed(seed)
    inp = L.Input((g.NWIN, 1)); h = inp
    for ch, k, pool in ((16, 15, 4), (32, 9, 2), (d, 5, 1)):
        h = L.Conv1D(ch, k, padding="same")(h); h = L.BatchNormalization()(h); h = L.Activation("gelu")(h)
        if pool > 1: h = L.MaxPool1D(pool)(h)
    h = _pos_layer()()(h)
    for _ in range(depth): h = _block(h, d, heads, mlp, 0.1)
    return keras.Model(inp, _head(L.LayerNormalization()(h), nout=nout))


def build_cnn(seed=0, drop=0.2, nout=1):
    """The 1-D CNN of notebook 05 (binary logit)."""
    tf, keras, L = _tf(); tf.keras.utils.set_random_seed(seed)
    return keras.Sequential([L.Input((g.NWIN, 1)),
        L.Conv1D(16, 15, padding="same"), L.BatchNormalization(), L.ReLU(), L.MaxPool1D(4),
        L.Conv1D(32, 9, padding="same"), L.BatchNormalization(), L.ReLU(), L.MaxPool1D(4),
        L.Conv1D(64, 5, padding="same"), L.BatchNormalization(), L.ReLU(),
        L.GlobalMaxPooling1D(), L.Dense(32, activation="relu"), L.Dropout(drop), L.Dense(nout)])


# --------------------------------------------------------------------------------------------------------------
# the variants of the study
# --------------------------------------------------------------------------------------------------------------
VARIANTS = {
    # name: (builder kwargs, fresh injections each epoch, epochs, peak lr, seeds)
    "cnn_static":      (dict(arch="cnn"),               False, 20, 2e-3, (0, 1)),
    "cnn_fresh":       (dict(arch="cnn"),               True,  20, 2e-3, (0, 1)),
    "tf16_static_lr3": (dict(arch="tf", patch=16),      False, 12, 3e-3, (0,)),
    "tf16_fresh":      (dict(arch="tf", patch=16),      True,  12, 1e-3, (0, 1)),
    "tf8_static":      (dict(arch="tf", patch=8),       False, 12, 1e-3, (0,)),
    "hyb_static":      (dict(arch="hyb"),               False, 12, 1e-3, (0, 1)),
    "hyb_fresh":       (dict(arch="hyb"),               True,  12, 1e-3, (0, 1)),
    # four-class (noise / sine-Gaussian / pulse / chirp) networks trained with fresh injections (nout = 4)
    # binary CNN trained with fresh injections of the three phenomenological morphologies AND of IMRPhenomD signals (notebook 08)
    "cnn_fresh_imr":   (dict(arch="cnn", pool="all"),   True,  20, 2e-3, (0, 1)),
    "cnn4_fresh":      (dict(arch="cnn", nout=4),       True,  20, 2e-3, (0, 1)),
    # notebook 33: depth / width / training length of the patch Transformer (the levers not covered by notebook 07)
    "tf16_d6_fresh":   (dict(arch="tf", patch=16, depth=6),                 True, 12, 1e-3, (0,)),
    "tf16_w128_fresh": (dict(arch="tf", patch=16, d=128, heads=4, mlp=256), True, 12, 1e-3, (0,)),
    "tf16_long_fresh": (dict(arch="tf", patch=16),                          True, 30, 1e-3, (0,)),
    "tf16_4_fresh":    (dict(arch="tf", patch=16, nout=4), True, 12, 1e-3, (0,)),
    "hyb4_fresh":      (dict(arch="hyb", nout=4),       True,  12, 1e-3, (0,)),
}


def builder(spec, seed):
    spec = dict(spec); arch = spec.pop("arch"); spec.pop("pool", None)
    return {"cnn": build_cnn, "tf": build_tf, "hyb": build_hybrid}[arch](seed=seed, **spec)


def train_variant(name, DSd, S1_train, seed=0, patience=6, bs=128, verbose=True):
    """Resumable training. Returns (model, history dict). The best-validation weights are restored and cached."""
    tf, keras, L = _tf()
    spec, fresh, epochs, lr, _ = VARIANTS[name]
    fn = os.path.join(MODELS, f"v07_{name}_s{seed}.weights.h5"); hf = os.path.join(MODELS, f"v07_{name}_s{seed}_hist.json")
    m = builder(spec, seed)
    if os.path.exists(fn) and os.path.exists(hf):
        m.load_weights(fn); return m, json.load(open(hf))
    nout = spec.get("nout", 1)
    ykey = "yb" if nout == 1 else "y4"
    X, y = DSd["train"]["X"], DSd["train"][ykey]; XV, yv = DSd["val"]["X"], DSd["val"][ykey]
    loss = keras.losses.BinaryCrossentropy(from_logits=True) if nout == 1 else keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    noise = DSd["train"]["noise"]; steps = int(math.ceil(len(X) / bs))
    sched = keras.optimizers.schedules.CosineDecay(0.0, epochs * steps, alpha=0.02, warmup_target=lr, warmup_steps=2 * steps)
    m.compile(keras.optimizers.AdamW(sched, weight_decay=1e-4), loss)
    r = np.random.default_rng(10_000 + seed)
    hist = dict(loss=[], val_loss=[], seconds=[]); best, best_w, bad = np.inf, None, 0
    n_noise = len(noise)
    for ep in range(epochs):
        t0 = time.time()
        if fresh:                                                     # new burst morphology / SNR / parameters / phase every epoch
            sig, kinds_, _ = make_signals(noise, S1_train, list(g.MORPHS_ALL if spec.get("pool") == "all" else g.MORPHS), r)
            X = np.concatenate([clip(noise), clip(sig)])[..., None]
            if nout > 1: y = np.r_[np.zeros(len(noise), int), [KID[k] for k in kinds_]]
        h = m.fit(X, y, validation_data=(XV, yv), epochs=1, batch_size=bs, verbose=0, shuffle=True)
        l, vl = float(h.history["loss"][0]), float(h.history["val_loss"][0])
        hist["loss"].append(l); hist["val_loss"].append(vl); hist["seconds"].append(time.time() - t0)
        if verbose:
            print(f"  {name} s{seed} ep {ep + 1:2d}/{epochs}  loss {l:.4f}  val {vl:.4f}  ({hist['seconds'][-1]:.0f} s)", flush=True)
        if vl < best: best, best_w, bad = vl, m.get_weights(), 0
        else:
            bad += 1
            if bad >= patience: break
    m.set_weights(best_w); m.save_weights(fn); json.dump(hist, open(hf, "w")); return m, hist


if __name__ == "__main__":
    tf, keras, L = _tf()
    tf.config.threading.set_intra_op_parallelism_threads(int(os.environ.get("TF_THREADS", "3")))
    tf.config.threading.set_inter_op_parallelism_threads(2)
    names = sys.argv[1:] or list(VARIANTS)
    CH, data, S1, DS = load_data()
    for name in names:
        for s in VARIANTS[name][4]:
            t0 = time.time(); m, h = train_variant(name, DS, S1["train"], seed=s)
            print(f"== {name} seed {s}: {len(h['loss'])} epochs, best val {min(h['val_loss']):.4f}, {time.time() - t0:.0f} s, {m.count_params():,} params", flush=True)
