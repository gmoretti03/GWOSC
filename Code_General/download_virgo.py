"""Download one 4096 s stretch with H1, L1 and V1 in science mode in the Virgo part of O2 (1 Aug 2017 onwards) for notebook 38 (extension 20), away from events.
Writes results/virgo_chunk.json. Usage: python download_virgo.py"""
import json, os
import numpy as np
from gwosc import timeline, datasets
import download_coinc_data as d
A, B = 1185580818, 1187733618                       # 1-25 Aug 2017
def main():
    here = os.path.dirname(os.path.abspath(__file__)); events = [datasets.event_gps(e) for e in datasets.find_datasets(type="event", segment=(A, B))]
    segs = {k: timeline.get_segments(f"{k}_DATA", A, B) for k in ("H1", "L1", "V1")}; both = d.intersect(d.intersect(segs["H1"], segs["L1"]), segs["V1"]); cands = []
    for a, b in both:
        s = int(np.ceil(a / d.CHUNK) * d.CHUNK)
        while s + d.CHUNK <= b:
            if all(abs(s + d.CHUNK / 2 - g_) > 2000 + d.CHUNK / 2 for g_ in events): cands.append(s)
            s += d.CHUNK
    print(len(cands), "candidate chunks"); s = cands[len(cands) // 2]; files = {k: d.fetch(k, s) for k in ("H1", "L1", "V1")}
    json.dump(dict(gps_start=int(s), duration=d.CHUNK, files=files), open(os.path.join(here, "results", "virgo_chunk.json"), "w"), indent=1); print("done", s)
if __name__ == "__main__": main()
