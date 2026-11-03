# Testing, CI and dependencies

Run `python -m unittest discover -s tests -v`. This intentionally does not collect
legacy task demonstration scripts or Python observation tables named `test*`.
The suite uses deterministic fake LLMs, real process boundaries and small task
fixtures. Co-Bench dataset downloads are replaced at the loader boundary; scoring
functions themselves execute. Optional method/benchmark tests report explicit skips when their dependencies are unavailable.
Search contracts cover budget exhaustion, malformed output, concurrent workers,
terminal replay failures, resource cleanup and checkpoint warm starts. Candidate
slots and checkpoint limits are explained in `reproducibility.md`.

The workflow `.github/workflows/ci.yml` configures Python 3.9–3.13 on Linux, macOS
and Windows for core tests. A separate Linux job checks selected optional extras,
documentation and distribution contents. Workflow configuration is not evidence
that hosted jobs have run: the release commit must link to actual green runs.
The optional job also installs scikit-learn/seaborn for PartEvo objective-feature
fixtures; it does not download a language model or exercise BERT feature extraction.

Coverage includes all `llm4ad` implementation modules, including unexecuted
optional features. Only the literal scientific observation tables listed in
`pyproject.toml` are excluded. Generate both line and branch reports; CI uploads the raw JSON and HTML
as workflow artifacts.

## Installation profiles

| Profile | Components / limits |
| --- | --- |
| core | NumPy, Matplotlib, pytz, tqdm; common TSP/OBP evaluation and standard-library HTTP backends |
| `gui` | ttkbootstrap and PyYAML; desktop/source-checkout launcher |
| `openai`, `local-ollama` | Optional SDK integrations |
| `local-vllm` | Separate CUDA/model deployment stack; not promised on all core CI platforms |
| `tensorboard`, `wandb` | Optional logging stacks |
| `funsearch`, `multi-objective`, `meoh`, `partevo`, `llamea` | Method-specific requirements; LLaMEA integration pins upstream 1.2.0 instead of following `main`; full integration validation remains pending |
| `machine-learning`, `science-discovery`, `co-bench`, `pymoo-task`, `tsp-gls` | Task-specific libraries/data requirements |
| `test`, `docs` | Development tools, not core runtime requirements |

Python 3.13 core support requires NumPy 2.x; NumPy 1.x remains usable on supported
older Python versions. Optional binary/GPU/Box2D dependencies can impose narrower
ranges, so install only needed extras. `all` combines runtime features and may
not install on a CPU-only or non-Linux platform. Python 3.14 is not declared
supported until the implementation and dependency matrix have been tested.
Main control evaluators already use Gymnasium; no algorithm migration was needed.

Matplotlib remains in core because many existing task modules import it. Further
plotting separation would require additional compatibility work; it is not
silently removed from common task installations.

The legacy `environment.yaml` is a Windows-specific snapshot, not a portable
lock file or the supported-platform definition. Use the documented virtual
environment and extras for current installations.
