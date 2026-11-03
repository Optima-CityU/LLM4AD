# Reproducibility and open-weight example

Three claims must be distinguished:

1. **Artifact replay:** reuse the exact recorded LLM responses and deterministic
   evaluation to inspect a previous run.
2. **Regeneration:** query a model again with recorded inputs. Seeds, temperature,
   weight revision, tokenizer/inference versions and hardware are necessary
   provenance, but do not guarantee identical generations.
3. **Method fidelity:** reproduce an original method's behavior/results on its
   benchmark. Neither a shared interface nor response replay establishes this.

## Record and replay

```python
from llm4ad.tools.llm.replay import RecordingLLM, ReplayLLM

# backend is any existing LLM instance; never put its API key in metadata.
# llm = RecordingLLM(backend, "new-responses.jsonl",
#                    model_metadata={"model": "ID", "revision": "immutable-ID"})
# response = llm.draw_sample(prompt)
# llm.close()
# replay = ReplayLLM("new-responses.jsonl")
# assert replay.draw_sample(prompt) == response
```

JSONL records preserve start-order IDs, request hashes, complete requests/raw
responses, failures by exception type and elapsed time. Existing files are never
overwritten. Hashes detect changed requests, not a maliciously rewritten journal.
New format-2 recordings include a completion footer and sample count; incomplete
journals are rejected. Closing waits for active requests before writing the footer.
Legacy format-1 recordings remain readable, but cannot prove tail completeness or
distinguish changes to redacted argument values. Format 2 hashes the original
request before redacting credential fields in the stored copy.
Metadata credential fields are redacted; prompt/response text remains private
unless reviewed for publication. Keep recordings with the matching manifest,
dataset checksums, code and results.

Exact search-trajectory replay is currently tested for deterministic serial EoH.
Recording is thread-safe, but replay does not reproduce asynchronous scheduling.
Native sampler-based methods propagate terminal replay errors from worker threads
to `run()` and close their resources. HSEvo also propagates these errors through
its prompt-format fallback. Legacy third-party LLaMEA integration still needs
separate end-to-end validation; its upstream control flow is not covered by the
native runtime tests. Ordinary transient backend errors retain method-specific
retry behavior and consume reserved candidate slots in the native runtime.
Active backend requests still need finite timeouts: a stop signal cannot forcibly
interrupt an arbitrary blocking Python thread.
Some legacy HTTP adapters retry internally without a retry limit. Their per-request
timeout does not bound the whole `draw_sample` call; this runtime does not add a
global deadline or API-call quota to those adapters. The local Ollama example
propagates request failures instead of retrying internally.
Timing objectives, GPU nondeterminism and external evaluator state can change
scores. Older proprietary models may no longer be callable; response replay can
preserve existing artifacts but cannot recover unrecorded past responses.

## End-to-end local model

The bounded example uses real Online Bin Packing, parsing and process evaluation.
It is an integration example, not a reproduction of a paper's method benchmark.
Install Ollama separately, select an openly available model whose license fits
your use, and install its weights using that tool. This repository does not
download weights or run paid APIs automatically.

Get the exact installed name/digest from `http://127.0.0.1:11434/api/tags` and the
server version from `/api/version`. Supply both the model name and digest:

```shell
python -m example.reproducibility.run_record_replay --backend ollama --model MODEL_NAME --model-digest EXACT_DIGEST --output runs/open-model --samples 4 --seed 17
python -m example.reproducibility.run_record_replay --replay runs/open-model/responses.jsonl --output runs/open-model-replay --samples 4 --seed 17
```

The script verifies the digest, records server/generation metadata, snapshots the
generated dataset and writes `results.json`, `manifest.json`, `responses.jsonl`.
Replay requires the same seed, task configuration, dataset hash and sample budget
as the recorded run. Journals without this run metadata cannot use this example's
strict preflight check; the generic replay adapter still supports legacy journals.
Use new output directories. The manifest deliberately records the task generator's
seed separately from the search seed. Run from an installed source checkout.

The default `--backend fixture` exercises the same path offline with fixed code.
Only actual `--backend ollama` results count as an open-model run. Until such an
artifact is collected, the open-model experimental requirement remains pending.
Record model source/revision/license and hardware with the final submission.

## Candidate budgets and termination

For native sampler-based methods, `max_sample_nums` bounds reserved candidate
attempts, including malformed output and failed evaluation. Concurrent workers
reserve slots before sampling; the last FunSearch batch is truncated to the
remaining budget. A failed batch consumes its reserved slots because a backend
may have issued some requests before failing. The counter is independent of
profiler use. Template evaluation is separate; PartEvo also counts its local
seed evaluations. Reflection/refinement requests can add LLM calls, so this
parameter is not a token or API-call quota. HSEvo retains its upstream evaluation
accounting and LLaMEA its upstream budget; their budgets are not equated to this
candidate-attempt convention.

The owner of a run closes the shared evaluator executor after sampling workers
finish. Early initialization returns and terminal replay errors follow the same
cleanup path. Candidate code still has the execution limits in SECURITY.md.

## Checkpoint warm starts

Checkpoint loading and response replay provide different capabilities:

| Method | Restored state / limits |
| --- | --- |
| RandSample | Sample history and candidate-attempt count |
| HillClimb | History, attempt count and best function with its score |
| EoH / ReEvo | Saved population order, scores, algorithm descriptions, generation and attempt count |
| MEoH / MOEAD / NSGA2 | Saved population and objective vectors; full archive/RNG restoration remains incomplete |
| MCTS-AHD | Saved population and counters; a new tree is built from that population |
| LHNS | Unsupported: saved artifacts do not contain the required current/best/elite/annealing state; the loader raises a clear error |
| MLES / PartEvo / HSEvo / LLaMEA | Full checkpoint restoration is not validated by this native resume API test suite |

New profiled runs write `search_state.json` atomically before and after sampling.
It records method identity, claimed/completed candidate slots and the associated
history order. Loaders reject counters that do not match the method/history.
Claims interrupted before completion conservatively consume budget on restart.
Old checkpoints without this file use logged sample order, which cannot recover
unlogged failures and can include template records.

The supported loaders preserve history files and restore population snapshots
without invoking selection or survival again. A final available population/DB
snapshot is requested before profiler shutdown. Checkpoints still omit such
state as pending offspring, full trees/archives and RNG state; these are warm
starts, not a guarantee of identical continuation. Do not combine files from
different runs or claim response replay repairs missing checkpoint state.
