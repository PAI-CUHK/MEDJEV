# JEV and System One terminology in MEDJEV

MEDJEV is an independent research implementation inspired by the public JEV / System One design pattern. It is intended to be discoverable by researchers looking for **JEV**, **System One**, **typed decisions**, **calibrated probabilities**, **clinical evidence reasoning**, **biomedical NLP**, and **sleep-signal decision models**.

MEDJEV is **not** the official TypeSafe AI Jev API, SDK, hosted service, or a reproduction of proprietary Jev weights. The shared terminology describes a research design pattern; it does not imply affiliation, endorsement, compatibility, or access to the official Jev model.

## What the JEV pattern means here

The public JEV description frames a decision model as a shared `state` evaluated by explicit typed questions. The answer is structured and probabilistic rather than a generated paragraph. MEDJEV adapts that pattern to biomedical evidence:

```text
state: clinical record, literature passage, or sleep-signal context
questions: explicit evidence-relationship or signal-semantic questions
answers: candidate decision + per-candidate probabilities + scope metadata
```

The repository uses the following question vocabulary in its demonstrations:

| Primitive | MEDJEV interpretation | Typical output |
| --- | --- | --- |
| `Choice` | Select one meaning from an explicit candidate set | `supported`, `contradicted`, or `unresolved` plus probabilities |
| `Score` | Place evidence on an ordered research rubric | Score, levels, and probabilities |
| `Noul` | Evaluate a bounded yes/no statement | Probability of the positive decision |

These names describe the public JEV-style interface pattern; they do not imply that MEDJEV is an official implementation, compatible SDK, hosted model, or reproduction of proprietary weights.

## MEDJEV and SLEEPJEV

- **MEDJEV** focuses on clinical text, biomedical literature, evidence–statement relationships, candidate-order behavior, and calibration scope.
- **SLEEPJEV** applies the related runtime-question pattern to overnight polysomnography and physiological signals. See the [companion repository](https://github.com/PAI-CUHK/SLEEPJEV).

## Primary references

- [TypeSafe System One overview](https://github.com/chujianyun/typesafe-wiki/blob/main/references/docs/Concepts/System%20One.md)
- [Jev SDK guide](https://www.jevtypesafe.org/docs/jev-sdk/)
- [jev-mcp: typed judgments and probabilities](https://github.com/jkudish/jev-mcp)
- [MEDJEV architecture](architecture.md)

## Directory-ready project description

Use the following wording when listing MEDJEV in a research directory or an independent JEV project gallery:

> Independent JEV-inspired typed decision research for clinical evidence and biomedical text, with runtime `Choice` / `Score` / `Noul`-style questions, explicit candidate probabilities, calibration metadata, and an experimental SLEEPJEV signal extension.

MEDJEV remains research software. Its benchmark snapshots are not clinical validation, and no patient data, hosted API credentials, model weights, or private experiment archives belong in this repository.
