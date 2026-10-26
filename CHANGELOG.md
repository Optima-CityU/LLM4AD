# Changelog

## 1.20.0 — 2026-10-26

- Add automated tests, CI and API/developer documentation.
- Add LLM recording/replay and checkpoint counter tracking; fix search termination, concurrency and resource cleanup.
- Fix incomplete-solution scoring, 2D packing overlap checks and optimization score directions.
- Support core Python 3.13, make GUI dependencies optional and clarify licensing/execution limits.

### Behavior changes

- `ttkbootstrap` is no longer a core dependency. GUI users install it with `pip install "llm4ad[gui]"` (or `pip install ".[gui]"` from a source checkout). NumPy 2 is now supported.
- Scores are not comparable with earlier logs for these tasks:
  - CO-Bench open-shop scheduling, CO-Bench capacitated p-median and Pymoo MOEA/D now return maximized scores; earlier versions returned them with the wrong sign.
  - 2D bin packing, CFLP and set cover now score incomplete or overlapping solutions as invalid (`None`). Set cover requires only elements that some subset covers.
- `max_sample_nums` now counts every LLM attempt, including initialization, failed requests, unparsable output and failed evaluations. Runs stop exactly at the budget; initialization may stop earlier than before when many outputs are invalid.
- A method object runs once; create a new object for another run.
- `SecureEvaluator` no longer changes the global multiprocessing start method. Process-pool evaluation on macOS/Windows uses spawn and requires picklable evaluators and an `if __name__ == '__main__':` guard.
- Profiler text logs go to their own log file and no longer propagate to the root logger.
- Resuming after a hard crash uses the larger of the saved attempt counter and the sample history, with a warning. Logs written before samples recorded `sample_order` cannot be resumed.

See [migration notes](docs/reproducibility.md) for details.

## Project history

Key milestones from Git history; dates reflect changes reaching `main`, not package releases.

| Period | Key changes | Source |
| --- | --- | --- |
| 2026-10 | Add HSEvo, additional LLM provider APIs and lazy imports. | [#75](https://github.com/Optima-CityU/LLM4AD/pull/75), [#76](https://github.com/Optima-CityU/LLM4AD/pull/76), [#77](https://github.com/Optima-CityU/LLM4AD/pull/77) |
| 2026-06 | Add Orienteering and Max-Cut tasks. | [#73](https://github.com/Optima-CityU/LLM4AD/pull/73), [#74](https://github.com/Optima-CityU/LLM4AD/pull/74) |
| 2026-03–04 | Add MLES, PartEvo and control-task examples; migrate to Gymnasium and fix GUI compatibility. | [#67](https://github.com/Optima-CityU/LLM4AD/pull/67), [#69](https://github.com/Optima-CityU/LLM4AD/pull/69) |
| 2025-12 | Add LLaMEA, extend MEoH beyond two objectives, and improve GUI/seed examples. | [#51](https://github.com/Optima-CityU/LLM4AD/pull/51), [0ed00c6](https://github.com/Optima-CityU/LLM4AD/commit/0ed00c6) |
| 2025-06–08 | Add ReEvo, LHNS, MCTS-AHD, CO-Bench, Pymoo MOEA/D and circle-packing examples. | [Commit range](https://github.com/Optima-CityU/LLM4AD/compare/51dfe57...a632369) |
| 2025-04–05 | Add MEoH, MOEAD and NSGA2; improve evaluation, initialization and logging. | [Commit range](https://github.com/Optima-CityU/LLM4AD/compare/6e2f3b3...51dfe57) |
| 2025-01–03 | Add YAML configurations, expand optimization/control tasks and integrate Feynman SRSD with 120 equation definitions. | [#5](https://github.com/Optima-CityU/LLM4AD/pull/5), [#11](https://github.com/Optima-CityU/LLM4AD/pull/11), [#12](https://github.com/Optima-CityU/LLM4AD/pull/12) |
| 2024-11 | Initial platform with EoH, FunSearch, HillClimb, RandSample, task interfaces, GUI and Colab examples. | [Initial commit](https://github.com/Optima-CityU/LLM4AD/commit/a827e7a), [#1](https://github.com/Optima-CityU/LLM4AD/pull/1) |

## Published packages

- [1.11](https://pypi.org/project/llm4ad/1.11/) — 2025-04-28.
- [1.1](https://pypi.org/project/llm4ad/1.1/) — 2025-03-13.
- [1.0](https://pypi.org/project/llm4ad/1.0/) — 2024-11-26.

The GitHub [v1.0.0 release](https://github.com/Optima-CityU/LLM4AD/releases/tag/v1.0.0) is a separate record from these PyPI versions.
