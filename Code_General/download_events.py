"""Download 4096 s of H1 and L1 strain around three O1 events (GW150914, GW151012, GW151226) for notebook 13.
The window is aligned to the 4096 s grid of the GWOSC files; raw float32 .npy go to cache/raw/ (git-ignored) and the list
to results/events_chunks.json (tracked). Usage: python download_events.py"""
import json, os, time
import numpy as np
from gwosc import datasets
from gwpy.timeseries import TimeSeries
HERE = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(HERE, "cache", "raw")
EVENTS = ["GW150914-v3", "GW151012-v3", "GW151226-v2"]; CHUNK = 4096
def fetch(det, s, attempts=6):
    f = os.path.join(RAW, f"{det}_{s}.npy")
    for k in range(attempts):
        if os.path.exists(f): break
        try:
            ts = TimeSeries.fetch_open_data(det, s, s + CHUNK, cache=True)
            assert abs(1.0 / ts.dt.value - 4096) < 1e-6 and len(ts) == CHUNK * 4096
            np.save(f, np.asarray(ts.value, dtype=np.float32))
        except Exception as e:
            print(f"  {det} {s}: attempt {k + 1} failed ({type(e).__name__}: {e}); retrying", flush=True); time.sleep(20)
    if not os.path.exists(f): raise RuntimeError(f"could not download {det} {s}")
    return os.path.basename(f)
if __name__ == "__main__":
    out = []
    for ev in EVENTS:
        t = datasets.event_gps(ev); s = int(np.floor(t / CHUNK) * CHUNK)
        if t - s < 300 or s + CHUNK - t < 300: s -= 1024 if t - s < 300 else -1024     # keep the event away from the chunk edges
        print(ev, t, s, flush=True)
        out.append(dict(event=ev.split("-")[0], gps=float(t), gps_start=s, duration=CHUNK, files={d: fetch(d, s) for d in ("H1", "L1")}))
    with open(os.path.join(HERE, "results", "events_chunks.json"), "w") as f: json.dump(out, f, indent=1)
    print("done")
