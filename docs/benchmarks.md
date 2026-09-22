# Benchmarks and Claims

## Latest PubMedQA development snapshot

This is the current public performance snapshot. The 500-example PubMedQA test split was not accessed.

| Backbone | Method | Canonical | Unseen answer | Unseen decision | Permutation behavior |
| --- | --- | ---: | ---: | ---: | --- |
| Qwen3-0.6B | Direct | 57% | 55% | 55% | variable across permutations |
| Qwen3-0.6B | Direct SFT | 31% | 38% | 35% | variable across permutations |
| Qwen3-0.6B | Fixed | 34% | 34% | 34% | position-dependent |
| Qwen3-0.6B | MedJEV | 60% | 60% | 60% | invariant in all four permutations |
| Qwen3-1.7B | Direct | 60% | 53% | 51% | variable across permutations |
| Qwen3-1.7B | Direct SFT | 34% | 34% | 34% | variable across permutations |
| Qwen3-1.7B | Fixed | 31% | 31% | 33% | position-dependent |
| Qwen3-1.7B | MedJEV | 72% | 68% | 36% | invariant; decision schema exposes a semantic limitation |
| Qwen3.5-4B | Direct | 73% | 73% | 60% | variable across permutations |
| Qwen3.5-4B | Direct SFT | 87% | 88% | 77% | strong, but not invariant in all schemas |
| Qwen3.5-4B | Fixed | 28% | 28% | 29% | position-dependent |
| Qwen3.5-4B | MedJEV | 80% | 88% | 80% | invariant in all four permutations |

The primary metric is identity-restored accuracy averaged over four candidate permutations. The fixed development partition contains 50 examples, so each percentage point represents one example and confidence intervals are wide. Each completed matrix contains 25 development records per schema and permutation.

The strongest current result is Qwen3.5-4B MedJEV: 80% canonical, 88% unseen answer, and 80% unseen decision with invariance across all four permutations. Direct SFT reaches 87%/88% but remains variable across permutations. Qwen3-1.7B MedJEV reaches 72%/68% but drops to 36% on unseen decision, exposing a semantic limitation rather than a uniformly strong result.

## Claim boundaries

- These are development-only results and are not clinical validation.
- The 500-example PubMedQA test split was not accessed.
- The table is not a confidence interval or population-level estimate; the small development set makes percentage differences unstable.
- Permutation invariance is an interface/semantic robustness property, not proof of medical generalization.
- No claim is made that MedJEV is best on every metric or backbone.

## Reproduction and provenance

The public figure is generated from [the supplied source table](../figures/source_data/pubmedqa_dev_performance.csv) by [the local R script](../figures/scripts/pubmedqa_dev_performance.R). The plot source, exported derivatives, data profile, and audit are kept under `figures/`. The source snapshot was supplied as the latest project result; raw PubMedQA records, checkpoints, and private run artifacts remain external.

Before adding future numbers:

1. Freeze the task, candidate semantics, split, and metric.
2. Keep training, model selection, calibration, and test data separate.
3. Report candidate-permutation behavior alongside accuracy.
4. Preserve the raw source table and the plotting script.
5. Keep clinical validity separate from benchmark accuracy.
