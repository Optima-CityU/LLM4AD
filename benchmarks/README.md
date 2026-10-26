# Runtime/memory measurement tools

Install `.[benchmark]` and run from the checkout:

```shell
python -m benchmarks.run benchmarks/fixture.json --output runs/benchmark/report.json
```

The fixture executes EoH/OBP with fake responses for ten seeds. It validates the
measurement tool, not algorithm quality or competitor performance. Real framework
experiments require explicit revision-pinned adapters and filled comparison
configurations.
