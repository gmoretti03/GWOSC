"""Download the noise-only H1 strain chunks used by the whole project (GWOSC, O1, 4096 Hz).

Chunks are 4096 s long, aligned to the 4096 s GWOSC file grid, fully inside H1 science segments (H1_DATA) and
at least EXCL seconds away from any catalogued O1 event. Raw strain is cached as float32 .npy in cache/raw/
(git-ignored); the chunk list is written to results/chunks.json (tracked).

Usage: python download_data.py
"""
import json
import os
import numpy as np
from gwosc import timeline, datasets
from gwpy.timeseries import TimeSeries

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "cache", "raw")
O1 = (1126051217, 1137254417)
CHUNK = 4096
EXCL = 2000.0
N_CHUNKS = 8


def candidate_chunks():
    events = [datasets.event_gps(e) for e in datasets.find_datasets(type="event", segment=O1)]
    segs = timeline.get_segments("H1_DATA", *O1)
    out = []
    for a, b in segs:
        s = int(np.ceil(a / CHUNK) * CHUNK)
        while s + CHUNK <= b:
            if all(abs(s + CHUNK / 2 - g) > EXCL + CHUNK / 2 for g in events):
                out.append(s)
            s += CHUNK
    return out


def main():
    os.makedirs(RAW, exist_ok=True)
    cands = candidate_chunks()
    print(len(cands), "candidate chunks")
    idx = np.linspace(0, len(cands) - 1, N_CHUNKS).round().astype(int)
    chosen = [cands[i] for i in idx]
    meta = []
    for s in chosen:
        f = os.path.join(RAW, f"H1_{s}.npy")
        if not os.path.exists(f):
            ts = TimeSeries.fetch_open_data("H1", s, s + CHUNK, cache=True)
            assert abs(1.0 / ts.dt.value - 4096) < 1e-6 and len(ts) == CHUNK * 4096
            np.save(f, np.asarray(ts.value, dtype=np.float32))
        meta.append({"detector": "H1", "gps_start": int(s), "duration": CHUNK, "file": os.path.basename(f)})
        print("ok", s)
    with open(os.path.join(HERE, "results", "chunks.json"), "w") as fh:
        json.dump(meta, fh, indent=1)


if __name__ == "__main__":
    main()
