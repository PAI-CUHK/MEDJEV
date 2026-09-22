# Data and Model Policy

## No raw clinical data in Git

Do not commit patient records, clinical notes, EDF signal files, protected health information, credentials, or access-controlled dataset copies. The `.gitignore` is a guardrail, not a substitute for review.

## Dataset records

Every dataset used in a published result should have a separate manifest containing:

| Field | Required content |
| --- | --- |
| `name` | Official dataset name |
| `version` | Release, commit, or access date |
| `source` | Stable URL or repository |
| `license` | Exact terms or access agreement |
| `task` | Intended task and label semantics |
| `split` | Train/development/calibration/test policy |
| `patient_disjointness` | Evidence for or limitation of subject-level separation |
| `sha256` | Hash of the local source archive when permitted |

## Model records

For every checkpoint record:

- model name and revision;
- tokenizer name and revision;
- architecture and trainable parameters;
- candidate schema and label mapping;
- seed and software environment;
- maximum sequence length and truncation policy;
- calibration data and temperature-fitting procedure;
- license and redistribution permission.

## Results

Generated predictions, plots, logs, and checkpoints should be stored in a separate results archive or release artifact. The source repository should contain scripts and manifests that explain how to reproduce them, not an unbounded collection of machine-specific outputs.

## Medical scope

MEDJEV returns a relationship to supplied evidence. It does not estimate disease prevalence, diagnosis, prognosis, treatment response, or patient risk unless a separate task definition, dataset, clinical validation protocol, and safety review establish those claims.
