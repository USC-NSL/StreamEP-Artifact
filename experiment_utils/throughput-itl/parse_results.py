#!/usr/bin/env python3
"""Summarize + compare the throughput-vs-ITL sweep for StreamInfer and sglang.

Usage: python parse_results.py <RESULTS_BASE>
Scans <RESULTS_BASE>/<system>/sharegpt-<rate>rps/result.json for
system in {streaminfer, sglang}, extracts output-token throughput + ITL, prints a
combined table, writes comparison.csv, and (if matplotlib) comparison.png.
"""
import sys, os, re, glob, csv, json

BASE = sys.argv[1] if len(sys.argv) > 1 else "results"
_NL = os.environ.get("NUM_LAYERS", "")
LAYERS_LABEL = f"{_NL} of 36 layers" if _NL else "reduced layers"

def parse_streaminfer(path):
    # /run_once response text: "token_throughput: N tokens/s ... itl_latency_mean: N ms ..."
    try: txt = open(path, errors="ignore").read()
    except FileNotFoundError: return None
    def g(p):
        m = re.search(p, txt); return float(m.group(1)) if m else None
    tput = g(r"token_throughput:\s*([0-9.]+)\s*tokens/s")
    if tput is None: return None
    return dict(tput=tput, itl_mean=g(r"itl_latency_mean:\s*([0-9.]+)\s*ms"),
                itl_median=g(r"itl_latency_median:\s*([0-9.]+)\s*ms"),
                itl_p99=g(r"itl_latency_p99:\s*([0-9.]+)\s*ms"))

# server-side per-10s stats line, e.g.:
# [ts] from Detokenizer Manager, Throughput: 24320.7 tokens/s, In-flight requests: 2495,
#   Waiting requests: 0, ITL mean=104.38 ms, median=104.48 ms, p99=292.87 ms, samples=240349
_DETOK_RE = re.compile(
    r"from Detokenizer Manager,\s*Throughput:\s*([0-9.]+)\s*tokens/s.*?"
    r"ITL mean=([0-9.]+)\s*ms,\s*median=([0-9.]+)\s*ms,\s*p99=([0-9.]+)\s*ms")

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
    client = dict(tput=f("output_throughput"), itl_mean=f("mean_itl_ms"),
                  itl_median=f("median_itl_ms"), itl_p99=f("p99_itl_ms"))
    try: log_txt = open(os.path.join(os.path.dirname(path), "server_head.log"), errors="ignore").read()
    except FileNotFoundError: return client
    windows = [tuple(map(float, m)) for m in _DETOK_RE.findall(log_txt)]
    if not windows: return client
    def avg(i): return sum(w[i] for w in windows) / len(windows)
    return dict(tput=avg(0), itl_mean=avg(1), itl_median=avg(2), itl_p99=avg(3))

# keys must match the results sub-dirs written by run_head_*.sh (streaminfer / sglang);
# "sglang EP" is only the display label (see `lab` below).
PARSERS = {"streaminfer": parse_streaminfer, "sglang": parse_sglang}

data = {}
for system, parser in PARSERS.items():
    for d in sorted(glob.glob(os.path.join(BASE, system, "sharegpt-*rps"))):
        m = re.search(r"sharegpt-(\d+)rps", d)
        if not m: continue
        res = parser(os.path.join(d, "result.json"))
        if res: data.setdefault(system, {})[int(m.group(1))] = res

rates = sorted({r for s in data.values() for r in s})
def s(x): return f"{x:.0f}" if isinstance(x, (int, float)) else "-"

print(f"\nThroughput (tok/s) & ITL (ms) vs request rate  —  gptoss {LAYERS_LABEL}, sharegpt")
hdr = f"{'rate':>5} | {'SI tput':>8} {'SI itl':>7} | {'SG tput':>8} {'SG itl':>7}"
print(hdr); print("-" * len(hdr))
rows = []
for r in rates:
    si = data.get("streaminfer", {}).get(r); sg = data.get("sglang", {}).get(r)
    print(f"{r:>5} | {s(si and si['tput']):>8} {s(si and si['itl_mean']):>7} | "
          f"{s(sg and sg['tput']):>8} {s(sg and sg['itl_mean']):>7}")
    rows.append(dict(rate_rps=r,
        streaminfer_tput=(si or {}).get('tput'), streaminfer_itl_mean=(si or {}).get('itl_mean'), streaminfer_itl_p99=(si or {}).get('itl_p99'),
        sglang_tput=(sg or {}).get('tput'), sglang_itl_mean=(sg or {}).get('itl_mean'), sglang_itl_p99=(sg or {}).get('itl_p99')))
print("\n(SI=StreamInfer, SG=sglang; tput=output tok/s; itl in ms)")

if rows:
    p = os.path.join(BASE, "comparison.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {p}")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    colors = {"streaminfer": "tab:blue", "sglang": "tab:orange"}
    lab = {"streaminfer": "StreamInfer", "sglang": "sglang EP"}
    # single latency-throughput curve: token throughput (x) vs ITL (y), one point per
    # rate (labeled), connected in rate order. mean = solid, p99 = dashed.
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for sysn, d in data.items():
        rs = sorted(d)
        if not rs or not all(d[r].get('itl_mean') is not None for r in rs):
            continue
        c = colors.get(sysn)
        tp = [d[r]['tput'] for r in rs]
        itl = [d[r]['itl_mean'] for r in rs]
        ax.plot(tp, itl, "o-", color=c, label=lab.get(sysn, sysn))
        for r, x, y in zip(rs, tp, itl):
            ax.annotate(f"{r} rps", (x, y), textcoords="offset points",
                        xytext=(6, 5), fontsize=8, color=c)
    ax.set(xlabel="output throughput (tok/s)", ylabel="inter-token latency (ms)",
           title=f"Latency vs throughput — gptoss ({LAYERS_LABEL}, fake prefill, sharegpt)")
    ax.grid(alpha=.3); ax.legend()
    fig.tight_layout()
    png = os.path.join(BASE, "comparison.png"); fig.savefig(png, dpi=120); print(f"wrote {png}")
except Exception as e:
    print(f"(plot skipped: {e})")
