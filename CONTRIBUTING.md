# Contributing

Use a source checkout and a dedicated virtual environment:

```shell
python -m venv .venv
# Activate the environment using your platform's standard command.
python -m pip install -e ".[test,docs]"
python -m unittest discover -s tests -v
```

Optional feature tests should skip only when their optional dependency is
unavailable. Do not mock task scoring just to make its tests pass. Replace
network/download boundaries with deterministic fixtures, and test both valid
and invalid candidates. Normal CI must not call paid APIs or download models.

For parser/evaluator/selection changes, include a failure reproducer and a
regression test. Keep search operators, prompts and fitness changes explicit;
record any intentional behavior change in `CHANGELOG.md`. Avoid broad algorithm
refactors in engineering-only pull requests.

```shell
python -m coverage run -m unittest discover -s tests -v
python -m coverage combine
python -m coverage json
python -m coverage html
python scripts/build_task_catalogue.py
python scripts/build_api_reference.py
python -m sphinx -W -b html docs docs/_build/html
python -m build
python -m twine check dist/*
python scripts/check_distribution.py dist
```

Coverage reports must include unexecuted implementation modules. The explicit
pure observation-table exclusions are listed in `pyproject.toml`; evaluator,
method and backend implementations are not excluded to increase a percentage.
The target is coverage close to 100%, not a claim that the current suite has
already reached it. Include line/branch gaps and test skips when reporting.

See [extension guides](docs/development.md), [execution limits](SECURITY.md) and
[release procedure](docs/releasing.md). Retain source copyright/attribution;
new original project files should identify BSD-2-Clause. Third-party code keeps
its original license. Citation requests are not extra license conditions.

For changes to sampling/termination, exercise valid, malformed and failing
candidates with a tiny budget, both with and without a profiler. Include
concurrent workers, replay mismatch/exhaustion and cleanup checks. For checkpoint
changes, verify scores, order, generation, consumed attempts and unchanged sample
history; describe any state not restored. Update the unreleased CHANGELOG entry
without inventing a release date, tag or numerical fidelity result.
