#!/usr/bin/env python3
"""Summarize + compare network-interference tolerance for StreamInfer and sglang.

Usage: python parse_results.py <RESULTS_BASE>
Scans <RESULTS_BASE>/<system>/<condition>/result.json for system in
{streaminfer, sglang} and condition in {none, aws-ring, azure-ring, aws-single},
extracts output-token throughput + ITL, and reports throughput degradation and ITL
inflation vs each system's own no-interference baseline. Writes interference.csv and
(if matplotlib) interference.png.
"""
import sys, os, re, csv, json

BASE = sys.argv[1] if len(sys.argv) > 1 else "results"
_NL = os.environ.get("NUM_LAYERS", "")
LAYERS_LABEL = f"{_NL} of 36 layers" if _NL else "reduced layers"

# condition tag -> display label (also fixes the plotting/order)
CONDS = [("none", "baseline"), ("single-link", "single-link"),
         ("single-link-2x", "single-link 2x"), ("all-links", "all links"),
         ("bidir-all-links", "bidir. all links")]

def parse_streaminfer(path):
    try: txt = open(path, errors="ignore").read()
    except FileNotFoundError: return None
    def g(p):
        m = re.search(p, txt); return float(m.group(1)) if m else None
    tput = g(r"token_throughput:\s*([0-9.]+)\s*tokens/s")
    if tput is None: return None
    return dict(tput=tput, itl_mean=g(r"itl_latency_mean:\s*([0-9.]+)\s*ms"),
                itl_p99=g(r"itl_latency_p99:\s*([0-9.]+)\s*ms"))

# server-side per-10s stats line, e.g.:
# [ts] from Detokenizer Manager, Throughput: 24320.7 tokens/s, In-flight requests: 2495,
#   Waiting requests: 0, ITL mean=104.38 ms, median=104.48 ms, p99=292.87 ms, samples=240349
_DETOK_RE = re.compile(
    r"from Detokenizer Manager,\s*Throughput:\s*([0-9.]+)\s*tokens/s.*?"
    r"ITL mean=([0-9.]+)\s*ms,\s*median=[0-9.]+\s*ms,\s*p99=([0-9.]+)\s*ms")

def parse_sglang(path):
    # result.json (bench_serving client) gates on "the run completed", but the metrics
    # come from the SERVER's Detokenizer Manager lines: under overload the client-side
    # ITL/throughput also count detokenizer->client delivery stalls and drain time, and
    # deviate from what the server actually sustained. Average every parsable line;
    # itl_p99 is the mean of the per-window p99s, not a whole-run p99.
    try: d = json.load(open(path))
    except Exception: return None
    if not isinstance(d, dict) or "output_throughput" not in d: return None
    def f(k):
        try: return float(d.get(k))
        except Exception: return None
    client = dict(tput=f("output_throughput"), itl_mean=f("mean_itl_ms"), itl_p99=f("p99_itl_ms"))
    try: log_txt = open(os.path.join(os.path.dirname(path), "server_head.log"), errors="ignore").read()
    except FileNotFoundError: return client
    windows = [tuple(map(float, m)) for m in _DETOK_RE.findall(log_txt)]
    if not windows: return client
    def avg(i): return sum(w[i] for w in windows) / len(windows)
    return dict(tput=avg(0), itl_mean=avg(1), itl_p99=avg(2))

# keys must match the results sub-dirs written by run_head_*.sh (streaminfer / sglang);
# "sglang EP" is only the display label.
PARSERS = {"streaminfer": parse_streaminfer, "sglang": parse_sglang}
LAB = {"streaminfer": "StreamInfer", "sglang": "sglang EP"}

data = {}
for system, parser in PARSERS.items():
    for tag, _ in CONDS:
        res = parser(os.path.join(BASE, system, tag, "result.json"))
        if res: data.setdefault(system, {})[tag] = res

def s(x, w=8, p=0): return (f"{x:.{p}f}" if isinstance(x, (int, float)) else "-").rjust(w)
def ratio(a, b): return (a / b) if (isinstance(a, (int, float)) and isinstance(b, (int, float)) and b) else None

print(f"\nNetwork-interference tolerance  —  gptoss {LAYERS_LABEL}, sharegpt, fixed rate")
print("(tput=output tok/s; itl in ms; tput_drop=throughput drop vs baseline; ITLx=ITL inflation vs baseline)\n")
hdr = f"{'system':>11} | {'condition':>15} | {'tput':>8} {'itl':>7} | {'tput_drop':>9} {'ITLx':>6}"
print(hdr); print("-" * len(hdr))
rows = []
for system in PARSERS:
    d = data.get(system, {})
    base = d.get("none")
    for tag, label in CONDS:
        r = d.get(tag)
        drop = itlx = None
        if r and base and tag != "none":
            drop = ratio(base["tput"] - r["tput"], base["tput"])
            itlx = ratio(r.get("itl_mean"), base.get("itl_mean"))
        print(f"{LAB[system]:>11} | {label:>15} | {s(r and r['tput'])} {s(r and r.get('itl_mean'),7)} | "
              f"{(f'{drop*100:.0f}%' if drop is not None else '-'):>9} {(f'{itlx:.2f}x' if itlx is not None else '-'):>6}")
        rows.append(dict(system=system, condition=tag,
                         tput=(r or {}).get("tput"), itl_mean=(r or {}).get("itl_mean"),
                         itl_p99=(r or {}).get("itl_p99"),
                         tput_drop_frac=drop, itl_inflation=itlx))
    print("-" * len(hdr))

if rows:
    p = os.path.join(BASE, "interference.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {p}")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import numpy as np
    systems = [sy for sy in PARSERS if sy in data]
    colors = {"streaminfer": "tab:blue", "sglang": "tab:orange"}
    labels = [lb for _, lb in CONDS]
    x = np.arange(len(CONDS)); w = 0.8 / max(1, len(systems))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    for i, sy in enumerate(systems):
        d = data[sy]
        tp = [(d.get(t) or {}).get("tput", 0) or 0 for t, _ in CONDS]
        it = [(d.get(t) or {}).get("itl_mean", 0) or 0 for t, _ in CONDS]
        off = (i - (len(systems) - 1) / 2) * w
        ax1.bar(x + off, tp, w, label=LAB[sy], color=colors.get(sy))
        ax2.bar(x + off, it, w, label=LAB[sy], color=colors.get(sy))
    for ax, ylab, title in ((ax1, "output throughput (tok/s)", "Throughput under interference"),
                            (ax2, "mean inter-token latency (ms)", "Latency under interference")):
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(ylab); ax.set_title(title); ax.grid(axis="y", alpha=.3); ax.legend()
    fig.suptitle(f"Network-interference tolerance — gptoss ({LAYERS_LABEL}, fake prefill, sharegpt)")
    fig.tight_layout()
    png = os.path.join(BASE, "interference.png"); fig.savefig(png, dpi=120); print(f"wrote {png}")
except Exception as e:
    print(f"(plot skipped: {e})")
