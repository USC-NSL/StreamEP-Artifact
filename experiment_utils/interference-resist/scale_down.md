# Why the interference intensity is scaled up at half cluster scale

The original paper experiment ran on **8 nodes × 2 L40S = 16 GPUs** serving the full
**36-layer** gpt-oss-120b (EP16). This artifact reproduces it on **4 nodes × 2 L40S =
8 GPUs** with **18 layers** (EP8) — the same scale-down used by the `throughput-itl`
reproduction. Everything else is unchanged: the same workload at the same **100 req/s**,
and the same RoCE fabric (**200 Gbps** links, ~**182 Gbps** achievable per `ib_write_bw`).

Half the GPUs running half the layers is just for scaling the VRAM usage down proportional to the cluster-size change. However, to reflect the same network pressure of the 16 GPU case, we need to consider:

- each token crosses **half as many MoE layers** (36 → 18), halving its
  dispatch/combine exchanges;
- a smaller share of those exchanges leave the node (`1 − 2/16 = 7/8` remote at
  16 GPUs vs. `1 − 2/8 = 3/4` at 8).

So at identical token throughput, the cluster generates **~2.3× less cross-node
traffic**, while the links stayed 200 Gbps. The cloud-noise trace that congested the
original cluster no longer makes a significant impact on this scaled-down cluster. So, we scale-up the interference level by a factor. This multiplicative factor is default to 6, where in this scaled-down cluster the sglang baseline shows the slowdown similar to the extent presented in the paper, while StreamInfer shows minimal performance difference under the same interference.

## Fidelity notes
In this reproduction, sglang's per-step GLOO barrier was moved to a dedicated
  control NIC (required for its load stability at this scale), sheltering it from the
  interference, while StreamInfer's Ray/ZMQ control plane still shares the interfered
  datapath.
