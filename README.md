# StreamEP-Artifact

For SOSP'26 artifact review, this repo (including submodules) contains

- StreamInfer: prototype distributed MoE decoding system implementing StreamEP. Currently only work with dummy model weights, yet it performs all model computations and data movements, and can replay authentic expert routing profiled from real model execution of each dataset.
- sglang_dummy_prefill: a fork of sglang with modifications to support decoding from dummy KV-Cache tensors. It can also replay expert routing from profiles. It also supports other minor options (e.g. skipping the 1st dense layer in GLM) for fair comparison with StreamInfer.
- experiment_utils: Instructions and scripts for artifact reviewers.

## Getting started

**On the provided testbed (SPHERE)** everything is already installed — go straight to
[`experiment_utils/`](experiment_utils/README.md) and run the two experiments below.

Note that the first main experiment in the paper draft (Figure 10) is about the peak throughput, and the throughput-itl experiment (Figure 11) is an extention of it, so we directly start with figure 11 for reproduction. To limit the total combinations of experiments, we only run GPT-OSS-120b on the SPHERE testbed, as the Delta testbed is not reservable. The next experiment we reproduce is the tolerance to network interference (Figure 13).

- Throughput vs. ITL sweep: [`experiment_utils/throughput-itl/`](experiment_utils/throughput-itl/README.md), corresponding to Figure 11 in the paper draft.
- Network-interference tolerance: [`experiment_utils/interference-resist/`](experiment_utils/interference-resist/README.md), corresponding to Figure 13 in the paper draft.

Each of the above have two scripts, one for StreamInfer and one for baseline, so there are 4 major experiment scripts to run. On sphere, each script is about 30 minutes, so about 2 hours are needed in total.

**On any other cluster**, install both systems on every node first:

- StreamInfer: see [`StreamInfer/readme.md`](StreamInfer/readme.md).
  [`experiment_utils/setup_node.sh`](experiment_utils/setup_node.sh) is the per-node
  script we used to provision SPHERE — it assumes that environment, so elsewhere treat
  it as a **reference** for the steps, not a turnkey installer.
- sglang_dummy_prefill (baseline): see [`sglang_dummy_prefill/readme.md`](sglang_dummy_prefill/readme.md)
  — installs into its **own** torch-2.8 conda env (separate from StreamInfer).