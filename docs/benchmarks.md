# Benchmarks and Claims

## Current evidence level

The initial research archive demonstrates a runnable prototype and a testable candidate-query interface. It does not establish that dynamic candidates outperform fixed classification, that the system is clinically valid, or that an LLM's factuality is improved in real use.

The preliminary single-seed snapshot shown in the README is retained as context only:

| Route | Accuracy | Interpretation |
| --- | ---: | --- |
| Dynamic candidate route | 84.39% | One fixed experiment configuration |
| Fixed-label control | 84.81% | Comparable control in the same archive |

The small difference should not be described as superiority or non-inferiority. The exact dataset, split, seed, calibration, and checkpoint must accompany any future reported number.

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
