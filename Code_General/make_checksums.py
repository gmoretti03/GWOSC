"""SHA-256 of the raw strain files in cache/raw/ -> results/data_checksums.json (tracked), to verify a re-download. Usage: python make_checksums.py [--check]"""
import hashlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); RAW = os.path.join(HERE, "cache", "raw"); OUT = os.path.join(HERE, "results", "data_checksums.json")
def sha(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
cur = {f: sha(os.path.join(RAW, f)) for f in sorted(os.listdir(RAW)) if f.endswith(".npy")}
if "--check" in sys.argv:
    ref = json.load(open(OUT)); bad = [f for f in ref if cur.get(f) != ref[f]]; print("mismatch or missing:", bad if bad else "none"); sys.exit(1 if bad else 0)
json.dump(cur, open(OUT, "w"), indent=1); print(len(cur), "files hashed")
