# Release Checklist

Before creating the first public release:

- [ ] Confirm repository owner, URL, authors, contact, and citation metadata.
- [ ] Confirm the software license and contributor ownership.
- [ ] Review all external dataset, checkpoint, and third-party source terms.
- [ ] Run `git status --ignored` and inspect the complete staged file list.
- [ ] Verify there are no credentials, private notes, patient text, raw datasets, checkpoints, or machine-specific paths.
- [ ] Run `pytest -q` on supported Python versions.
- [ ] Run `ruff check src tests` and resolve or document findings.
- [ ] Build and inspect both sdist and wheel.
- [ ] Install the wheel in a clean environment and run the contract example.
- [ ] Tag the exact source commit used for every published result.
- [ ] Publish data/model manifests separately from restricted artifacts.
- [ ] Use conservative language for all medical and performance claims.

## GitHub settings

Enable branch protection, required CI checks, dependency alerts, secret scanning, issue templates, and a private security contact before inviting external contributions.
