# N2/STO-3G benchmark

The benchmark answers two separate questions:

1. Is the readable implementation mathematically correct?
2. Under the same model and sampling protocol, is its end-to-end VMC time competitive with
   NetKet?

The first is checked by exact diagonalization, not by agreement between two stochastic runs.
The second uses a full fixed-spin, singles-only protocol because NetKet's public fermion-hop
sampler does not provide the point-group-preserving singles/doubles mixture used by the
recommended qcnqs run.

The qcnqs Ag matrix and the NetKet full-space sparse operator independently reproduce PySCF
FCI within `1.0e-13 Ha` and `2.0e-13 Ha`, respectively. This validates both integral
conventions before comparing either optimizer trajectory.

## Locked matched protocol

| Setting | qcnqs | NetKet |
|---|---:|---:|
| Sector | fixed `(N_alpha,N_beta)=(7,7)` | same |
| Sector size | 14,400 | 14,400 |
| Backflow hidden width | 20 | 20 |
| Trainable parameters | 6,300 | 6,300 |
| Initialization | orthogonal | orthogonal |
| Numeric type | FP64 | FP64 |
| Samples / chains | 4,096 / 1,024 | same |
| Chain length | 4 | 4 |
| Initial discarded sweeps | 16 | 16 |
| Proposals per sweep | 20 | 20 |
| Proposal | spin-conserving single hop | same |
| Optimizer | SGD, learning rate 0.05 | same |
| SR | sample-space MinSR | NTK SR |
| Diagonal shift | 0.05 | 0.05 |
| Updates | 300 | 300 |

The timer covers initialization-dependent first execution, all 300 updates, and the terminal
sampled estimate. Exact post-training model expectation is computed after the timer. Thus the
reported speed is an end-to-end workload measurement; it is not a cherry-picked steady-state
kernel time. Each framework uses its native PRNG, so equal seed labels identify independent
reproducible runs rather than byte-identical initial tensors.

## Reproduce

Generate and validate the common molecular input:

```bash
uv run python benchmark/generate_n2.py
uv run python benchmark/validate_n2.py \
  benchmark/data/N2.STO3G.RHF.FCIDUMP \
  --output benchmark/results/n2_validation.json
uv run python benchmark/validate_netket_operator.py \
  benchmark/data/N2.STO3G.RHF.FCIDUMP \
  --output benchmark/results/netket_operator_validation.json
```

Run qcnqs for each benchmark seed. qcnqs is a library, so configure the root
`main.py` dictionary rather than passing command-line flags. For the matched seed-3 run,
select the following values (all unspecified values remain at their defaults):

```python
PARAMETERS["system"]["kind"] = "n2"
PARAMETERS["system"]["n2"]["sector"] = "full"
PARAMETERS["sampling"]["single_probability"] = 1.0
PARAMETERS["runtime"].update(
    device="cuda",
    seed=3,
    output="benchmark/results/qcnqs_full_singles_seed3.json",
)
```

Then execute the ordinary Python script:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python main.py
```

Run the matched NetKet path:

```bash
CUDA_VISIBLE_DEVICES=1 uv run python benchmark/netket_n2.py \
  benchmark/data/N2.STO3G.RHF.FCIDUMP \
  --iterations 300 --samples 4096 --chains 1024 --discard 16 \
  --hidden-units 20 --seed 3 --chunk-size 512 --exact-model \
  --output benchmark/results/netket_full_seed3.json
```

The recommended point-group run is deliberately reported outside the matched table. Change
the two mode-specific entries in `main.py`, then run it in the same way:

```python
PARAMETERS["system"]["n2"]["sector"] = "point-group"
PARAMETERS["sampling"]["single_probability"] = 0.05
PARAMETERS["runtime"]["output"] = (
    "benchmark/results/qcnqs_point_group_seed3_exact.json"
)
```

Repeat all three runs with seeds 7 and 11, then build machine-readable statistics:

```bash
uv run python benchmark/summarize.py \
  benchmark/results/qcnqs_full_singles_seed3.json \
  benchmark/results/qcnqs_full_singles_seed7.json \
  benchmark/results/qcnqs_full_singles_seed11.json \
  benchmark/results/netket_full_seed3.json \
  benchmark/results/netket_full_seed7.json \
  benchmark/results/netket_full_seed11.json \
  benchmark/results/qcnqs_point_group_seed3_exact.json \
  benchmark/results/qcnqs_point_group_seed7_exact.json \
  benchmark/results/qcnqs_point_group_seed11_exact.json \
  --output benchmark/results/all_summary.json
```

## Checked environment and results

The recorded runs use an NVIDIA GeForce RTX 4090 D, Python 3.12, PyTorch 2.12.1+cu126,
PySCF 2.14, JAX 0.11.0, and NetKet 3.22.3. The common PySCF FCI reference is
`-107.65412244752476 Ha`.

Values are mean +/- sample standard deviation over seeds 3, 7, and 11. Energy errors are
absolute errors relative to the common FCI result.

| Implementation / mode | End-to-end time (s) | Sampled error (mHa) | Exact model error (mHa) |
|---|---:|---:|---:|
| qcnqs full, matched | 19.38 +/- 1.19 | 37.04 +/- 13.99 | 33.33 +/- 5.51 |
| NetKet full, matched | 339.94 +/- 24.03 | 29.72 +/- 4.49 | 32.17 +/- 5.56 |
| qcnqs Ag, recommended | 29.62 +/- 0.22 | 13.79 +/- 2.13 | 13.80 +/- 0.68 |

The two matched rows have comparable exact model errors, while qcnqs is 17.54x faster by
the mean end-to-end time. The Ag row is not used to calculate that speedup because it changes
both the physical sector and proposal mixture; it shows the benefit of the supported
point-group path under the same `H=20`, 300-update workload.

Generated FCIDUMP, JSON result, and checkpoint files are ignored by Git. Commands regenerate
all evidence from the locked source and dependency versions.
