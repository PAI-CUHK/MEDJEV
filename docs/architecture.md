# Architecture

MEDJEV is organized around one narrow contract: relate a supplied record to a supplied statement under an explicit candidate meaning set.

## Data flow

```text
record + statement + candidates
              |
              v
       semantic encoders
              |
              v
      candidate-conditioned
          decision head
              |
              v
       logits and probabilities
              |
              v
    calibration + scope metadata
              |
              v
          runtime result
```

## Layers

### Public contract

`medjev.api` defines `Candidate`, `Query`, and `DecisionEngine`. Query construction validates nonempty and unique candidate IDs and meanings. Runtime output preserves candidate identifiers and includes:

- per-candidate probabilities;
- the selected prediction;
- calibration status;
- execution backend;
- the scope statement that the output is an evidence relationship, not disease probability.

### Semantic models

- `medjev.model.DecisionHead` reads shared evidence with independent candidate queries.
- `medjev.direct_jev.DirectJEV` scores dynamic candidates with shared set context.
- `medjev.gpu` provides Transformer-backed training and inference.
- `medjev.native` provides native label-logit and prefix-cache baselines.

These are separate experimental routes. Their results must not be combined into one performance claim.

### Data and evaluation

`medjev.data` handles JSONL loading, grouped development partitioning, split overlap audits, and file hashes. `medjev.schemas` defines candidate semantics and calibration keys. `medjev.metrics` handles probabilities, temperature fitting, Brier/NLL/ECE-style metrics, and paired bootstrap utilities.

### Experimental signal extension

`sleepjev` is intentionally a separate package. It provides multi-resolution signal tokens, temporal/event indexes, sparse or dense readout, dynamic temporal queries, and fixed-head baselines. It shares the candidate-query philosophy but does not require the text package's datasets or checkpoints.

## Invariants

The test suite treats these as interface contracts:

1. Reordering candidates reorders the output mapping without changing the score associated with a meaning.
2. Queries remain isolated from one another.
3. Masked/padded evidence does not change a valid result.
4. Development calibration does not read the held-out test split.
5. Unknown or positive-unlabeled signal events are not silently treated as negative labels.

These invariants improve auditability but do not establish clinical validity.

## Extension points

New components should generally be added in this order:

1. Define a typed input/output contract.
2. Add a dataset adapter that keeps raw data external.
3. Add a baseline and a focused test.
4. Add an evaluation manifest with model revision, split, seed, and calibration scope.
5. Add documentation before adding a generated result artifact.
