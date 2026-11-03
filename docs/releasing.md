# Release procedure

The version is stored in `llm4ad/_version.py` and read by
runtime, packaging and docs. The existing primary BSD-2-Clause license and
bundled Apache attribution are retained. Resolve `THIRD_PARTY_NOTICES.md` gaps
before publishing a release.

1. Resolve provenance, correctness, documentation and retained experimental claims.
2. Obtain green hosted CI evidence for the declared matrix and feature tests;
   record skips, line/branch coverage and model/benchmark artifacts.
3. Select an unused final version, update the single version source, README,
   changelog, generated API/catalogue, and commit.
4. Build from that commit: `python -m build`, `python -m twine check dist/*`,
   `python scripts/check_distribution.py dist`.
5. Install the wheel and independently build/install the sdist in clean
   environments; run examples outside the checkout. Compare dependency count,
   install time and size under matched Python/platform/cache conditions.
6. Record commit, environment snapshot and artifact SHA-256 hashes. Exclude
   VCS/cache/private files from the source archive.
7. A maintainer creates the matching tag and uploads verified artifacts to PyPI.
   This repository does not automatically publish packages.
8. Point versioned hosted docs to that exact release.

No tag, package upload or external document/site edit is claimed by this working
tree. An uncommitted development version is not an immutable release.

## GitHub and package release records

Use [Optima-CityU/LLM4AD](https://github.com/Optima-CityU/LLM4AD) as the canonical
repository. The existing GitHub `v1.0.0` tag and PyPI versions are separate records;
use the source links in [CHANGELOG](changelog.md) rather than inventing missing
tags or assigning current changes to old package releases.

Maintain the unpublished section while developing. At release time, move verified
changes under the final version/date and include a concise GitHub release body:

- What changed and which users are affected.
- Score-sign, candidate-budget, dependency and checkpoint migration notes.
- Exact commit, package hashes and links to executed CI and validation artifacts.
- Remaining limitations, including experimental evidence not yet collected.

Check PyPI and existing tags immediately before selecting the final version.
Create the release from that immutable commit; the changelog, package metadata,
and documentation must identify the same version. Keep old
release records intact and avoid copying their outdated sandbox wording.
