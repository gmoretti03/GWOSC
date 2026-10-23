"""Download the H1+L1 chunks used by the two-detector (coincidence / time-slide) study of notebook 09.

A chunk is a 4096 s stretch, aligned to the 4096 s GWOSC file grid, fully inside the science segments of BOTH H1 and L1 and
at least EXCL seconds from every catalogued O1 event. The chunks 0 and 6 of the single-detector study qualify and are kept
(their H1 file is reused); the others are chosen evenly over the run. Raw strain is cached as float32 .npy in cache/raw/
(git-ignored); the list is written to results/coinc_chunks.json (tracked).

Usage: python download_coinc_data.py
"""
import json
import os
import time
import numpy as np
from gwosc import timeline, datasets
from gwpy.timeseries import TimeSeries

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "cache", "raw")
O1 = (1126051217, 1137254417)
CHUNK = 4096
EXCL = 2000.0
N_CHUNKS = 6
KEEP = [1126076416, 1135579136]          # chunks 0 and 6 of results/chunks.json


def intersect(a, b):
    out = []
    for s1, e1 in a:
        for s2, e2 in b:
            s, e = max(s1, s2), min(e1, e2)
            if e - s >= CHUNK: out.append((s, e))
    return out


def candidates():
    events = [datasets.event_gps(e) for e in datasets.find_datasets(type="event", segment=O1)]
    both = intersect(timeline.get_segments("H1_DATA", *O1), timeline.get_segments("L1_DATA", *O1))
    out = []
    for a, b in both:
        s = int(np.ceil(a / CHUNK) * CHUNK)
        while s + CHUNK <= b:
            if all(abs(s + CHUNK / 2 - g) > EXCL + CHUNK / 2 for g in events): out.append(s)
            s += CHUNK
    return out


def fetch(det, s, attempts=6):
    f = os.path.join(RAW, f"{det}_{s}.npy")
    for k in range(attempts):                       # the GWOSC server sometimes times out: retry
        if os.path.exists(f): break
        try:
            ts = TimeSeries.fetch_open_data(det, s, s + CHUNK, cache=True)
            assert abs(1.0 / ts.dt.value - 4096) < 1e-6 and len(ts) == CHUNK * 4096
            np.save(f, np.asarray(ts.value, dtype=np.float32))
        except Exception as e:                      # noqa: BLE001
            print(f"  {det} {s}: attempt {k + 1} failed ({type(e).__name__}); retrying", flush=True); time.sleep(20)
    if not os.path.exists(f): raise RuntimeError(f"could not download {det} {s}")
    return os.path.basename(f)


def main():
    os.makedirs(RAW, exist_ok=True)
    cands = candidates(); print(len(cands), "candidate chunks with H1 and L1 in science mode")
    others = [c for c in cands if c not in KEEP]
    idx = np.linspace(0, len(others) - 1, N_CHUNKS - len(KEEP)).round().astype(int)
    chosen = sorted(set(KEEP + [others[i] for i in idx]))
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(d, s) for s in chosen for d in ("H1", "L1")]
    with ThreadPoolExecutor(max_workers=4) as ex:           # the server is the bottleneck: use several connections
        names = list(ex.map(lambda j: fetch(*j), jobs))
    meta = []
    for s in chosen:
        meta.append({"gps_start": int(s), "duration": CHUNK, "files": {d: n for (d, ss), n in zip(jobs, names) if ss == s}}); print("ok", s, flush=True)
    with open(os.path.join(HERE, "results", "coinc_chunks.json"), "w") as fh:
        json.dump(meta, fh, indent=1)


if __name__ == "__main__":
    main()
