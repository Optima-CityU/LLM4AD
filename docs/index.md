# LLM4AD development documentation

These documents describe the checked-out development version. They do not
claim that the older public PyPI package or external documentation site has
already been updated. Build this browsable, searchable reference with:

```shell
python -m pip install ".[docs]"
python -m sphinx -W -b html docs docs/_build/html
```

```{toctree}
:maxdepth: 2

quickstart
development
api
engineering
security
reproducibility
task_catalogue
releasing
changelog
```
