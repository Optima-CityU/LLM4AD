# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM4AD is a platform for automatic algorithm design using Large Language Models. It provides unified interfaces for methods (search algorithms), tasks (optimization problems), and LLM backends.

## Installation

```bash
pip install .                    # Install locally
pip install llm4ad              # Install from PyPI
```

Python version: >= 3.9, < 3.13

## Running

### GUI

```bash
cd GUI && python run_gui.py
```

### Example Script

```python
from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH, EoHProfiler

llm = HttpsApi(
    host='https://api.openai.com/v1',  # Base URL (OpenAI-compatible API)
    key='sk-xxx',
    model='gpt-4o-mini',
    timeout=60
)

task = OBPEvaluation()
method = EoH(llm=llm, profiler=EoHProfiler(log_dir='logs'), evaluation=task)
method.run()
```

## Architecture

```text
llm4ad/
├── base/                  # Core abstractions
│   ├── code.py           # Function, Program classes for code representation
│   ├── evaluate.py       # Evaluation base class (secure subprocess evaluation)
│   └── sample.py         # LLM base class, SampleTrimmer for parsing LLM output
├── method/               # Search methods
│   ├── eoh/              # Evolution of Heuristics
│   ├── funsearch/        # FunSearch (island-based evolution)
│   ├── hillclimb/        # (1+1)-EPS hill climbing
│   ├── reevo/            # Reflective Evolution
│   ├── mles/             # Multimodal LLM Evolutionary Search
│   └── ...               # Other methods
├── task/                 # Problem domains
│   ├── optimization/     # TSP, bin packing, CVRP, scheduling...
│   ├── machine_learning/ # RL control tasks (Acrobot, CartPole...)
│   └── science_discovery/# Symbolic regression, equation discovery
└── tools/
    ├── llm/              # LLM interfaces
    │   ├── llm_api_https.py    # OpenAI-compatible API (uses openai SDK)
    │   └── local_ollama.py     # Local Ollama deployment
    └── profiler/         # Logging (local, TensorBoard, WandB)
```

## Key Classes

- **LLM** (`base/sample.py`): Abstract base for LLM backends. Implement `draw_sample(prompt)`.
- **Evaluation** (`base/evaluate.py`): Abstract base for problem evaluation. Implement `evaluate_program(callable_func, program_str)`. Supports secure subprocess evaluation with timeout.
- **SampleTrimmer** (`base/sample.py`): Parses LLM output to extract function body using AST.
- **Program/Function** (`base/code.py`): Code representation with template-based generation.

## Adding New Components

### New LLM Backend

1. Create file in `llm4ad/tools/llm/`
2. Inherit from `LLM` in `llm4ad/base/sample.py`
3. Implement `draw_sample(prompt) -> str`

### New Task

1. Create folder in `llm4ad/task/<domain>/`
2. Inherit from `Evaluation`
3. Implement `evaluate_program()` returning a float score
4. Add `paras.yaml` for parameter configuration

### New Method

1. Create folder in `llm4ad/method/<name>/`
2. Add profiler class inheriting from `ProfilerBase`
3. Method class coordinates LLM sampling and evaluation

## LLM API Configuration

The `HttpsApi` class uses the OpenAI SDK and accepts a base URL:

```python
host='https://api.openai.com/v1'           # OpenAI
host='https://api.deepseek.com/v1'         # DeepSeek
host='https://api.lkeap.cloud.tencent.com/coding/v3'  # Tencent Cloud (OpenAI-compatible)
```

## Method Parameters

Each method has a `paras.yaml` defining configurable parameters. Common parameters:

- `max_sample_nums`: Total samples to evaluate
- `num_samplers`: Parallel sampling threads
- `num_evaluators`: Parallel evaluation processes
- `pop_size`: Population size (for evolutionary methods)
