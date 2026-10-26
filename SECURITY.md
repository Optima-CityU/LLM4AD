# Execution limits and threat model

`SecureEvaluator` is a historical API name. It provides **process separation and
optional wall-clock interruption**, not an adversarial-code sandbox. Generated
code runs with the evaluating user's permissions. Do not interpret successful
evaluation as proof that code is safe.

| Capability | Current behavior |
| --- | --- |
| Main evaluation worker | Separate process when `safe_evaluate=True`; direct execution otherwise |
| Wall-clock deadline | Enforced for the direct worker when `timeout_seconds` is finite and process evaluation is enabled |
| Worker crash / `SystemExit` | Returns failed evaluation without waiting indefinitely for an absent result |
| File-system access | Unrestricted by this package; same permissions as the user |
| Network access | Unrestricted by this package |
| Memory limit | No package-enforced limit |
| CPU-time limit | No package-enforced CPU quota; wall-clock deadline is a different mechanism |
| Descendant processes | Daemon mode disallows Python multiprocessing children, but is not a general process-spawning restriction; termination does not guarantee cleanup of arbitrary descendants |
| Environment variables / credentials | Accessible to generated code unless the execution environment removes them |
| Malicious-code protection | Not provided by process separation, AST parsing, Numba or protected division |

The intended built-in mode is convenient evaluation of trusted or manually
reviewed experimental code. For untrusted candidates, use a separately managed
container/VM or external runner that applies filesystem, network, user, memory,
CPU and process-tree policies. Those policies are deployment responsibilities,
not guarantees made by the current evaluator. No hardened container backend is
claimed or enabled by this change.

Use `fork_proc=False` for explicit spawn; evaluator objects then need to be
pickleable. The evaluator uses a local multiprocessing context and does not
change the host application's global start method. `fork_proc='auto'` retains
the previous fork preference on Linux/macOS; that may be unsuitable for some
threaded native libraries. CI exercises explicit spawn as well.

Recordings contain complete prompts and responses. Credential-named metadata
fields are redacted, but arbitrary text is not scanned for secrets. Inspect
artifacts before sharing. API credentials should come from the environment and
must not be inserted into prompts, dataset files, commands or model metadata.
