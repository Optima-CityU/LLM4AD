# SPDX-License-Identifier: BSD-2-Clause
"""Read current batched sample logs and legacy individual-sample logs."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import warnings

from llm4ad.base import TextFunctionProgramConverter as Converter


def latest_snapshot(log_dir, directory='population', prefix='pop'):
    """Select only numbered snapshots; unrelated files cannot change ordering."""
    candidates = []
    for path in (Path(log_dir) / directory).glob(f'{prefix}_*.json'):
        match = re.fullmatch(re.escape(prefix) + r'_(\d+)\.json', path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise ValueError(f'No {prefix} snapshot found in {Path(log_dir) / directory}')
    order, path = max(candidates)
    return str(path), order


def restore_population(population_class, log_dir, pop_size, *, init_pop_size=None, template_program=None):
    """Restore saved individuals directly, without re-running survival/selection."""
    path, generation = latest_snapshot(log_dir)
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('Population snapshot must contain a nonempty list')
    functions = []
    objective_count = next((len(row['score']) for row in rows if isinstance(row, dict)
                            and isinstance(row.get('score'), list)), None)
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('function'), str) or 'score' not in row:
            raise ValueError('Invalid population record')
        function = Converter.text_to_function(row['function'])
        if function is None:
            raise ValueError('Invalid function in population snapshot')
        score = row['score']
        values = score if isinstance(score, list) else [score]
        if any(type(value) not in (int, float) for value in values if value is not None):
            raise ValueError('Invalid population score')
        if any(isinstance(value, float) and math.isnan(value) for value in values):
            raise ValueError('NaN is not a valid population score')
        if isinstance(score, list) and (not score or any(value is None for value in score)):
            raise ValueError('Invalid objective vector')
        if objective_count is not None and score is not None and (not isinstance(score, list) or len(score) != objective_count):
            raise ValueError('Inconsistent population objective dimensions')
        if score is None:
            score = [float('-inf')] * objective_count if objective_count else float('-inf')
        function.score = score
        function.algorithm = row.get('algorithm', '')
        program = row.get('program')
        if program is not None and not isinstance(program, str):
            raise ValueError('Invalid program in population snapshot')
        if not program:
            program = Converter.function_to_program(function, template_program) if template_program is not None else str(function)
        if program is None:
            raise ValueError('Population function is incompatible with the template')
        function.entire_code = str(program)
        functions.append(function)
    kwargs = {'pop_size': pop_size, 'generation': generation, 'pop': functions}
    if init_pop_size is not None:
        kwargs['init_pop_size'] = init_pop_size
    return population_class(**kwargs)


def restore_attempt_count(log_dir, legacy_count, method_name):
    """Use persisted attempt counts; old checkpoints fall back to history order."""
    path = Path(log_dir) / 'search_state.json'
    if not path.exists():
        return legacy_count
    state = json.loads(path.read_text(encoding='utf-8'))
    if (not isinstance(state, dict) or type(state.get('schema_version')) is not int
            or state['schema_version'] != 1
            or state.get('method') != method_name
            or type(state.get('history_samples')) is not int):
        raise ValueError('Search counters do not match this method/sample history')
    completed, claimed = state.get('completed_attempts'), state.get('claimed_attempts')
    if type(completed) is not int or type(claimed) is not int or not 0 <= completed <= claimed:
        raise ValueError('Invalid search attempt counters')
    if state['history_samples'] != legacy_count:
        # A hard kill can land between a counter save and a sample write.
        warnings.warn('Search counters lag the sample history (e.g. after a hard crash); '
                      'using the larger attempt count.', RuntimeWarning, stacklevel=2)
        return max(claimed, legacy_count)
    # Claims were saved before backend calls; an interrupted in-flight request
    # conservatively consumes its slot rather than gaining budget after restart.
    return claimed


def load_samples(log_dir):
    """Return sample dictionaries ordered by recorded sample order.

    Best-only logs are excluded. Malformed files raise instead of silently
    constructing a partial experiment history. Zero scores are preserved.
    """
    paths = Path(log_dir, "samples").glob("samples_*.json")
    paths = sorted(p for p in paths if p.name != "samples_best.json")
    samples = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        rows = payload if isinstance(payload, list) else [payload]
        for row in rows:
            if not isinstance(row, dict) or not {"function", "score", "sample_order"} <= row.keys():
                raise ValueError(f"Invalid sample record in {path}")
            if type(row['sample_order']) is not int or row['sample_order'] < 1 or not isinstance(row['function'], str):
                raise ValueError(f"Invalid sample order or function in {path}")
            score = row['score']
            values = score if isinstance(score, list) else [score]
            if (isinstance(score, list) and not score) or any(value is not None and type(value) not in (int, float) for value in values):
                raise ValueError(f"Invalid score in {path}")
            if isinstance(score, list) and any(value is None for value in score):
                raise ValueError(f"Invalid objective score in {path}")
            if any(isinstance(value, float) and math.isnan(value) for value in values):
                raise ValueError(f"NaN score in {path}")
            samples.append(row)
    samples.sort(key=lambda row: row["sample_order"])
    orders = [row["sample_order"] for row in samples]
    if len(orders) != len(set(orders)):
        raise ValueError("Duplicate sample orders in checkpoint history")
    return samples
