# Quick start

Install the development checkout with `python -m pip install .`. Install only
the extras used by a specific method/task/backend. For a GUI checkout use
`python -m pip install ".[gui]"`; launch from `GUI` with `python run_gui.py`.
This is a desktop feature and is not exercised by headless CI interaction.

Run an end-to-end offline fixture, then replay its LLM responses:

```shell
python -m example.reproducibility.run_record_replay --backend fixture --output runs/fixture --samples 4
python -m example.reproducibility.run_record_replay --replay runs/fixture/responses.jsonl --output runs/replayed --samples 4
```

Both commands run real code parsing and Online Bin Packing evaluation. The
fixture's fixed responses test integration; they are not evidence of model
performance. Each output directory must be new. Results include generated
programs, task scores, a dataset snapshot/checksum and an environment manifest.

For an actual local model, follow the pinned-digest example in
[reproducibility](reproducibility.md). Review [execution limits](security.md)
before evaluating generated code. `None` means failed/invalid evaluation;
scalar scores follow the maximization convention. A numeric zero is valid.
