"""Download N_EXTRA more H1+L1 chunks (extension 21: longer two-detector background) that are NOT in results/coinc_chunks.json.
Same selection rules as download_coinc_data.py (4096 s inside both science segments, >= 2000 s from every catalogued event); the new chunks
are chosen evenly among the remaining candidates and written to results/coinc_chunks_extra.json. Usage: python download_coinc_extra.py"""
import json, os
import numpy as np
import download_coinc_data as d
N_EXTRA = 6
def main():
    here = os.path.dirname(os.path.abspath(__file__)); old = {c["gps_start"] for c in json.load(open(os.path.join(here, "results", "coinc_chunks.json")))}
    cands = [c for c in d.candidates() if c not in old and all(abs(c - o) > d.CHUNK for o in old)]; print(len(cands), "remaining candidates")
    idx = np.linspace(0, len(cands) - 1, N_EXTRA).round().astype(int); chosen = sorted(set(cands[i] for i in idx))
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(det, s) for s in chosen for det in ("H1", "L1")]
    with ThreadPoolExecutor(max_workers=4) as ex: names = list(ex.map(lambda j: d.fetch(*j), jobs))
    meta = [{"gps_start": int(s), "duration": d.CHUNK, "files": {det: n for (det, ss), n in zip(jobs, names) if ss == s}} for s in chosen]
    json.dump(meta, open(os.path.join(here, "results", "coinc_chunks_extra.json"), "w"), indent=1); print("done", len(meta))
if __name__ == "__main__": main()
