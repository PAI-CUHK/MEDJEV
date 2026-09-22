# Benchmarks and Claims

## Current evidence level

The initial research archive demonstrates a runnable prototype and a testable candidate-query interface. It does not establish that dynamic candidates outperform fixed classification, that the system is clinically valid, or that an LLM's factuality is improved in real use.

The preliminary single-seed snapshot shown in the README is retained as context only:

| Route | Accuracy | Interpretation |
| --- | ---: | --- |
| Dynamic candidate route | 84.39% | One fixed experiment configuration |
| Fixed-label control | 84.81% | Comparable control in the same archive |

The small difference should not be described as superiority or non-inferiority. The exact dataset, split, seed, calibration, and checkpoint must accompany any future reported number.

## Selected archived highlights

These values are copied from the internal result summaries used to prepare this public repository. The raw datasets, checkpoints, and result archives are intentionally not included in Git, so these are documented snapshots rather than a fresh public rerun.

| Measure | Result | Scope |
| --- | ---: | --- |
| Dynamic route calibrated ECE | 0.0141 | Official MedNLI test, n=1,422, seed 17; temperature fitted on development calibration data |
| MEDJEV v2 unseen-query accuracy | 87.38% +/- 0.87% | Three seeds, direct runtime-query answers over unseen wording |
| MEDJEV v2 unseen-query NLL/log(K) | 0.4236 +/- 0.0439 | Same three-seed comparison, uncalibrated |
| Best v2 direct query family | 92.92% +/- 0.33% | `refute` query family; diagnostic slice, not an overall score |
| Shared/fresh timing ratio | 1.25x-1.82x | 96-query probe, 12 paired conditions, median timing ratio |
| Candidate-order probe | 0.00% flip rate, max probability delta 0 | 96-query seed-17 probe |

The v2 comparison reports the following direct-query means: original NLI 84.01% +/- 0.77%, `support` 88.28% +/- 0.65%, `refute` 92.92% +/- 0.33%, and `determined` 87.20% +/- 0.53%. The corresponding unseen-query mean is 87.38% +/- 0.87%. These are schema/query slices derived from the same MedNLI relation labels, not independent clinical tasks.

The shared-execution ratio is a backend timing observation, not a service-level throughput claim. The compared path can change numerical probabilities under BF16; observed category flips ranged from 0.00% to 2.08% and maximum probability drift ranged from 0.0193 to 0.0537 across the paired conditions. Any production claim requires synchronized GPU timing, warm-up policy, p50/p95/p99 latency, throughput, memory, and quality-drift reporting under one fixed workload.

The reported 87.38% unseen-query result is a useful research signal, but it must be read with the paired derived baseline in the same archive: the mapped fixed baseline reached 88.23% +/- 0.46% on the corresponding derived comparison. This prevents the result from being presented as proof of a new medical reasoning advantage.

## Required evaluation discipline

Before making a performance claim:

1. Freeze the task, candidate semantics, and evaluation metric.
2. Separate training, model-selection, calibration, and final test data.
3. Use multiple seeds or report the single-seed limitation clearly.
4. Compare against a fixed-label baseline and a simple non-neural baseline.
5. Check candidate permutation, evidence ablation, and input-order robustness.
6. Report confidence intervals and failure cases.
7. Keep clinical validity separate from benchmark accuracy.

## Result artifacts

Result files should be generated from a pinned source commit and a machine-readable manifest. Never select a final model based on the sealed test set. Never merge scores from different routes into one headline result.
