# Record/replay example

From an installed source checkout, run:

```shell
python -m example.reproducibility.run_record_replay --output runs/record --samples 4
python -m example.reproducibility.run_record_replay --replay runs/record/responses.jsonl --output runs/replay --samples 4
```

Use fresh directories. The default backend is a deterministic test fixture,
not a learned model. See `docs/reproducibility.md` for the local open-weight
backend, mandatory model digest and limits of deterministic regeneration.
