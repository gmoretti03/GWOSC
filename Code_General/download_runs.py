"""Download H1 chunks of O2 and O3 for the generalisation study of notebook 36 (extension 41): N_PER_RUN chunks of 4096 s per run, evenly spread
over the run, inside H1 science segments and >= 2000 s from every catalogued event, aligned to the 4096 s grid. Raw float32 .npy go to cache/raw/ (git-ignored);
the list is written to results/runs_chunks.json (tracked). Usage: python download_runs.py"""
import json, os
import numpy as np
from gwosc import timeline, datasets
import download_coinc_data as d
RUNS = {"O2": (1164556817, 1187733618), "O3a": (1238166018, 1253977218), "O3b": (1256655618, 1269363618)}; N_PER_RUN = 3; EXCL = 2000.0
def main():
    here = os.path.dirname(os.path.abspath(__file__)); os.makedirs(d.RAW, exist_ok=True); meta = []
    for run, (a, b) in RUNS.items():
        events = [datasets.event_gps(e) for e in datasets.find_datasets(type="event", segment=(a, b))]
        cands = []
        for s0, e0 in timeline.get_segments("H1_DATA", a, b):
            s = int(np.ceil(s0 / d.CHUNK) * d.CHUNK)
            while s + d.CHUNK <= e0:
                if all(abs(s + d.CHUNK / 2 - g_) > EXCL + d.CHUNK / 2 for g_ in events): cands.append(s)
                s += d.CHUNK
        idx = np.linspace(0, len(cands) - 1, N_PER_RUN + 2).round().astype(int)[1:-1]; chosen = [cands[i] for i in idx]; print(run, len(cands), "candidates ->", chosen, flush=True)
        for s in chosen: meta.append({"run": run, "gps_start": int(s), "duration": d.CHUNK, "file": d.fetch("H1", s)})
    json.dump(meta, open(os.path.join(here, "results", "runs_chunks.json"), "w"), indent=1); print("done", len(meta))
if __name__ == "__main__": main()
