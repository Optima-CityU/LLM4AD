# Extension and architecture guide

The core flow is `LLM -> SampleTrimmer -> Function/Program -> SecureEvaluator ->
task score -> method population/database -> profiler`. Existing methods retain
their operators and selection/survival algorithms. Record/replay wraps the LLM
boundary and does not introduce a new search strategy.

## Add a task

Subclass `Evaluation`, provide a single-function template and implement
`evaluate_program`. Keep dataset generation separate and seed it explicitly.
Return a scalar maximization score (or the method's documented objective vector)
and `None` for an infeasible or failed solution. Check completeness, constraints,
index ranges and score direction independently of the generated heuristic.

```python
from llm4ad.base import Evaluation

class ToyEvaluation(Evaluation):
    def __init__(self):
        super().__init__("def f(x):\n    return x\n", timeout_seconds=5)

    def evaluate_program(self, program_str, callable_func, **kwargs):
        error = abs(callable_func(3) - 6)
        return -error
```

Add an `__init__.py`, source attribution, a human-readable description and YAML
configuration if the GUI exposes the task. Regenerate the catalogue. Add tiny
fixtures with independently known valid/invalid answers, not only a template
that executes without raising. External data must have provenance, license,
revision and checksums; do not download it at module import time.

## Add an LLM backend

Subclass `LLM`, implement `draw_sample(prompt, *args, **kwargs) -> str`, and release
resources in `close`. Declare any non-core dependency in its own extra. Keep API
credentials separate from generation parameters and define which prompt,
message and multimodal formats are accepted. Bound request deadlines and expose
errors rather than silently returning successful empty responses.

```python
from llm4ad.base import LLM

class ExampleLLM(LLM):
    def draw_sample(self, prompt, *args, **kwargs):
        return "def f(x):\n    return x * 2\n"
```

Test with a fake transport: inspect the sent payload, exercise provider errors
and timeouts, and verify that no credential reaches logs. Wrap the backend in
`RecordingLLM` to capture raw output before trimming. `ReplayLLM` requires matching
requests and never contacts a real backend.

## Add a method

Place the method in `llm4ad/method/<name>`, re-export public classes and provide
an example and optional GUI YAML. Reuse `Evaluation`, `LLM`, `Function`, `Program`
and the score convention instead of creating task-specific interfaces. Record
the original paper/source, immutable upstream revision and all adaptations.
Do not equate a standardized interface with fidelity to the original method.

Test initialization, each enabled operator, population/database updates,
termination with and without a profiler, malformed/failed candidates and
resource closure. Use one sampler/evaluator for a deterministic integration
fixture, then test concurrent behavior separately. Checkpoint loading and exact
LLM response replay are separate capabilities; document the saved state and gaps.

## API documentation

`scripts/build_api_reference.py` indexes public classes, functions, methods,
signatures and available docstrings directly from AST, without importing GPU,
network or task dependencies. Missing docstrings are explicitly marked; the
index's completeness does not imply complete parameter semantics. Improve the
source docstrings and regenerate the index in the same change.
