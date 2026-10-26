# Licensing and provenance

The primary license of LLM4AD's own code is **BSD-2-Clause**, reproduced in
`LICENSE`. The distribution also contains an Apache-2.0 component; the package
metadata therefore describes the combined distribution as
`BSD-2-Clause AND Apache-2.0`. This does not relicense either component.

| Component | Attribution / source | License and verification |
| --- | --- | --- |
| `llm4ad/method/funsearch/programs_database.py` | Copyright 2023 DeepMind Technologies Limited; original source notice preserved | Apache-2.0; full text in `LICENSES/Apache-2.0.txt` |
| Co-Bench adapters | Sun et al., CO-Bench, arXiv:2504.04310; source references retained | Project adapter code uses the project license; downloaded datasets have their own upstream terms. Per-dataset provenance and redistribution audit remains pending. |
| Feynman SRSD equation definitions and sampling | Matsubara et al., arXiv:2206.10540; see source references | Exact source revision and copied-code provenance remain to be verified before a formal release. A citation is not evidence of license permission. |
| Scientific observation tables | See each evaluator's source references | Data provenance and redistribution terms remain to be verified before a formal release. |
| Installed optional libraries / models | Individually installed, not bundled as model weights | Governed by their respective licenses; installing an extra does not transfer ownership or change those terms. |

Source SPDX labels express the project's current distribution declarations;
they are not a claim that the pending provenance checks above have been completed.
Maintainers must resolve those checks before publishing a release.
Retain upstream copyright, license and modification notices when contributing.

Commercial use does not require contacting the team. Academic citation and
industrial collaboration are welcome, voluntary requests under the license.
