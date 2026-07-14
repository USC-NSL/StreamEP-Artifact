# Network-interference tolerance — StreamInfer vs. sglang EP

Measures how each system's serving performance holds up when the RoCE fabric is under
**trace-driven network interference** — the experiment that shows StreamInfer's
**barrier-free** design tolerating congestion that stalls a barrier-based baseline:

- **StreamInfer** — the system under evaluation (barrier-free distributed MoE).
- **sglang EP** (`sglang_dummy_prefill`) — expert-parallel sglang with fake prefill; its
  per-decode-step DP-attention all-gather is a **barrier** that network jitter stalls.

Both run gpt-oss on dummy weights, layer count cut to 18 (original model has 36) for the 8-GPU
testbed (original exp has 16), sharegpt lengths, at a single fixed rate (100 rps). For each interference condition we replay a real cloud-noise trace as
RDMA traffic on the datapath links (via [`interference_gen`](../../interference_gen)),
run the benchmark, and compare throughput drop / latency inflation vs. the no-interference
baseline.

The four interference **modes** (the paper's), all replaying the `aws_hpc_metal` trace as
**one-directional** RDMA flows between node pairs (n0..n3 = the 4 nodes in order, n0 = head):

| Condition | Flows | Meaning |
|---|---|---|
| `none` | — | baseline (no interference) |
| `single-link` | n0→n1 | one link stressed, normal trace intensity |
| `single-link-2x` | n0→n1 | same link, **doubled** trace intensity |
| `all-links` | n0→n1, n2→n3 | half the nodes drive the other half (one-directional) |
| `bidir-all-links` | n0↔n1, n2↔n3 | same links, **both directions** (two generator sets per link) |

> Scaled-down reproduction of the paper's 16-GPU congestion study — see
> [What was scaled down](#what-was-scaled-down). All modes scale the trace intensity by
> a factor of 6 (see [`config.sh`](config.sh)) to compensate for the half-scale
> victim's idle link headroom — [`scale_down.md`](scale_down.md) derives why.

> **RDMA interference required.** TCP backs off under congestion and never takes bandwidth
> from NCCL's RDMA traffic — so the generator pushes **RDMA**, via **UCX**. See
> [UCX / RDMA requirement](#ucx--rdma-requirement).

---

## Quick reproduction on the provided testbed (SPHERE)

On SPHERE everything is **already prepared** — 4 nodes (**sgpu6–sgpu9**, 8× L40S), both
conda envs (`streaminfer`, `sglang`), `gdrdrv`, and the pinned **UCX 1.18 bundle in
`~/ucx118`** on every node; [`config.sh`](config.sh) already points at the cluster. **You
only launch the two sweeps — one at a time — and read the plot.** Run from the artifact
root on the head (**sgpu6**).

### 1 · sglang baseline — launch on head only (it auto SSH-launches workers + the interference)

```bash
bash experiment_utils/interference-resist/run_head_sglang.sh
```

### 2 · StreamInfer — launch on the head (sgpu6) and each worker node (sgpu7-9)

```bash
# head — sgpu6
bash experiment_utils/interference-resist/run_head_streaminfer.sh
# workers — run on sgpu7, sgpu8 and sgpu9 (each in a shell on that node)
bash experiment_utils/interference-resist/run_worker_streaminfer.sh
```

Each head run launches the inference server, then for every condition starts the RDMA
interference, runs the fixed-rate benchmark, stops the interference, and records. The
interference generator is launched **from the head over SSH** — nothing extra to run on
the workers. Ctrl-C the StreamInfer workers once the head prints `Sweep done`.

### 3 · Read the result

Both head runs re-parse at the end; once **both** systems have run you have:

- **`results/interference.png`** — grouped bars: output throughput (left) and mean ITL
  (right) per condition, both systems.
- `results/interference.csv` and `results/{streaminfer,sglang}/<condition>/…` — raw numbers,
  including `results/*/<condition>/interference.log` (the interference status log).

You should see **StreamInfer** nearly flat across conditions while **sglang** loses
throughput and gains latency under the heavier modes.

> Run the two systems **one at a time** — each uses all 8 GPUs, and the interference
> generator shares the same NICs. The comparison plot is drawn by the parse step under
> each run's own conda env, which needs `matplotlib` (both envs have it on SPHERE). To
> (re)draw any time: `NUM_LAYERS=18 python experiment_utils/interference-resist/parse_results.py experiment_utils/interference-resist/results`

---

## Outputs

```
results/
├── streaminfer/<condition>/result.json       # /run_once response (embeds metrics)
├── streaminfer/<condition>/interference.log   # interference status: offered vs achieved per flow
├── streaminfer/<condition>/interference.counters.csv  # 2s IB bytes + ECN/CNP samples (contention proof)
├── sglang/<condition>/result.json             # sglang.bench_serving output JSON
├── interference.csv                           # one row per (system, condition)
└── interference.png                           # throughput + ITL bars, both systems
```

---

## UCX / RDMA requirement

The interference generator must transmit over **RDMA (rc)**, so **UCX is a hard
dependency** on every node. Getting UCX `rc` to connect on RoCE needs three things (all
already set up on SPHERE; encoded as overridable defaults in [`config.sh`](config.sh)'s
`UCX_*` exports and [`lib_interference.sh`](lib_interference.sh)'s `IFACE_UCX_ENV` — the
same knobs also patch `interference_gen/run_ring.sh` for anyone using that tool directly):

1. **UCX ≥ 1.18, uniform on all nodes.** Ubuntu's UCX 1.16 has a broken `tcp_sockcm`
   listener (the wireup can't complete). SPHERE bundles UCX 1.18 in `~/ucx118` on every
   node and loads it via `LD_LIBRARY_PATH` (`UCX_LIB_DIR`) — no root, no clobbering the
   system UCX. Elsewhere: install UCX ≥ 1.18 (`apt-get install libucx0` where the distro
   ships ≥1.18, or build from source) with the `ib`/`rdmacm` transport modules, plus
   `perftest` (`ib_write_bw`).
2. **A tcp connection-manager for the wireup** (`UCX_SOCKADDR_TLS_PRIORITY=tcp,sockcm,rdmacm`).
   RoCE `rdma_cm`'s private-data area is too small for UCX's endpoint address, so the
   listener wireup is rejected; the tiny wireup handshake goes over tcp_sockcm while the
   **bulk data still rides `rc` (RDMA)**. `UCX_NET_DEVICES` therefore lists the IB
   device(s) **and** a netdev: `mlx5_1:1,rocep225s0f1:1,ens1f1np1`.
3. **The right RoCE device + GID.** `UCX_IB_GID_INDEX=3` (RoCE v2, IPv4-mapped GID). Device
   names differ per node (mlx5_1 vs. rocep225s0f1 on sgpu9) — a comma-list lets each node
   pick whichever it has.

Verify RDMA interference is really flowing (not falling back to TCP): while interference runs,
the datapath **IB** counters climb — `cat /sys/class/infiniband/mlx5_1/ports/1/counters/port_xmit_data`
(4-byte words) — whereas the kernel netdev byte counter stays flat (RDMA bypasses it).

Every run is additionally **self-verifying**: each condition's `interference.log` records,
per flow, the senders' own *achieved vs. target* Gbps sampled live during the benchmark
(plus each source's whole-run average at stop), and `interference.counters.csv` samples the
head's IB TX/RX bytes and RoCE congestion counters (`np_cnp_sent`, `rp_cnp_handled`,
ECN-marked packets) every 2 s — rising CNP/ECN deltas during the benchmark are DCQCN
engaging, i.e. hardware-level proof the interference and NCCL actually contended for the
link rather than coexisting in spare headroom.

---

## Detailed configuration & general (any-cluster) reproduction

These scripts are tuned for **SPHERE**; on other hardware/networks treat them as a
**reference**. Every value is environment-overridable (or edit [`config.sh`](config.sh)).

### Knobs

| Knob | Meaning |
|---|---|
| `N_NODE`, `N_GPU_PER_NODE` | → `WORLD_SIZE` = total GPUs = the EP/DP degree |
| `WORKER_HOSTS` | worker hostnames → ranks 1..N (head = rank 0) |
| `HEAD_IP`, `HOST_IFNAME`, `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX` | inference-side RoCE/NCCL config (see throughput-itl) |
| `NUM_LAYERS` | gpt-oss layers (original 36; reduced for the scaled-down cluster) |
| `RATE`, `BENCH_TIME` | fixed request rate + seconds of load per condition |
| `CONDITIONS` | which interference conditions to sweep |
| `IFACE_NODES` | interference nodes as `host:datapath-ip`, indexed n0..n3 (n0 = head) |
| `LINK_CAP_GBPS` | RoCE link capacity — scales the trace's interference rates |
| `IFACE_STREAMS`, `IFACE_TRACE`, `IFACE_WARMUP` | parallel streams per flow; trace name; seconds to let flows settle before benchmarking |
| `IFACE_EXTRA_MULT` | global trace-intensity multiplier compensating for the scaled-down victim's link headroom — see [`scale_down.md`](scale_down.md) |
| `IFACE_MAX_OUTSTANDING`, `IFACE_BURST_BYTES` | sender pipeline depth per stream (64 msgs / 8 MB burst). The tool's shallow defaults collapse under congestion-inflated RTT — the generator would yield to NCCL exactly when it should contend |
| `UCX_LIB_DIR`, `UCX_NET_DEV_LIST`, `UCX_IB_GID_INDEX`, `UCX_SOCKADDR_TLS_PRIORITY` | the RDMA-mode UCX env (see above) |
| `UCX_IB_TRAFFIC_CLASS` | pinned to `0` = NCCL's default TC, so interference and NCCL share the same switch queue / DCQCN loop (UCX's `auto` TC could be queued separately by a DSCP-trusting switch) |

### What each run does

Both runners drive the same fixed-rate benchmark used by the [throughput-itl](../throughput-itl)
experiment (StreamInfer via `POST /run_once`, sglang via `sglang.bench_serving`), wrapped
per condition by [`lib_interference.sh`](lib_interference.sh): `iface_start` generates the
trace's rate schedule on the head (applying the per-mode intensity multiplier), deploys it
to the flow endpoints, and launches `IFACE_STREAMS` parallel `ucx_sender`→`ucx_receiver`
pairs per one-directional flow with the validated rc env; it waits until the head's IB
counter shows real RDMA traffic, the benchmark runs, then `iface_stop` kills the flows.
StreamInfer reuses one server across conditions; sglang launches a fresh server per
condition (and retries on its transient under-load crashes).

### Common issues on other clusters

- **UCX `rc` won't connect** — almost always UCX < 1.18 or a missing tcp connection-manager;
  re-read [UCX / RDMA requirement](#ucx--rdma-requirement). Confirm raw RDMA + `rdma_cm`
  work first with `ib_write_bw -R -x 3 -d <dev> <peer-ip>`.
- **`WARNING: no RDMA traffic`** — `iface_start` confirms interference via the head's IB
  counter; if the head isn't an endpoint of any flow, or its datapath device isn't
  `mlx5_1`/`rocep225s0f1`, adjust `_iface_head_ib_tx` and `UCX_NET_DEV_LIST`.
- **Interference too weak/strong** — it's trace-driven and scaled by `LINK_CAP_GBPS` (set it
  to your link's real capacity) times `IFACE_EXTRA_MULT`. The right multiplier depends on
  your victim's link utilization: measure the per-node IB TX during a no-interference run
  and size the multiplier to fill the headroom (the arithmetic is worked in
  [`scale_down.md`](scale_down.md)). Check each run's `interference.log` — if *achieved*
  tracks *target*, the generator is contending as configured.
- **A condition ran with no interference** — `iface_start` prints `WARNING ... mark suspect`
  if no RDMA traffic appears within ~75 s; check that run's `interference.log`.

---

## What was scaled down

| Aspect | Original (sphere-16 congestion) | Here |
|---|---|---|
| Cluster | 8 nodes / 16 L40S | 4 nodes / 8 L40S |
| Model depth | 36 layers | 18 layers (½) |
| Gate profiles | balanced profiles | none (minor effect on this metric) |
| Rate / duration | 400 rps / 100 s | 100 rps / 150 s (rate where both baselines are stable) |
| Interference modes | single-link, single-link-2x, all-links, bidirectional all-links | same 4 modes (n0→n1, node pairs) |
| Interference transport | RDMA (UCX rc) | RDMA (UCX rc) |
| Trace intensity | 1× (victim links already at ~75%+ utilization) | × `IFACE_EXTRA_MULT` to fill the half-scale victim's idle headroom — [`scale_down.md`](scale_down.md) |
