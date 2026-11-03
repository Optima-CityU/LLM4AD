# SPDX-License-Identifier: BSD-2-Clause
from __future__ import annotations

import copy

from tqdm.auto import tqdm

from ...tools.checkpoint import load_samples, restore_attempt_count

from .hillclimb import HillClimb
from .profiler import HillClimbProfiler
from ...base import TextFunctionProgramConverter as tfpc, Function


def _get_all_samples_and_scores(path, get_algorithm=False):
    samples = load_samples(path)
    funcs = [row['function'] for row in samples]
    scores = [row['score'] for row in samples]
    maximum = max((row['sample_order'] for row in samples), default=0)
    if get_algorithm:
        return funcs, scores, maximum, [row.get('algorithm', '') for row in samples]
    return funcs, scores, maximum


def _resume_text2func(f, s, template_func: Function):
    temp = copy.deepcopy(template_func)
    f = tfpc.text_to_function(f)
    if f is None:
        temp.body = '    pass'
        temp.score = None
        return temp
    else:
        f.score = s
        return f


def _resume_pf(log_path: str, pf: HillClimbProfiler, template_func: Function):
    funcs, scores, sample_max_order = _get_all_samples_and_scores(log_path)
    print(f'RESUME HillClimb: Sample order: {sample_max_order}.', flush=True)
    # pf.__class__._num_samples = sample_max_order
    for i in tqdm(range(len(funcs)), desc='Resume HillClimb Profiler'):  # noqa
        f, s = funcs[i], scores[i]  # noqa
        f = _resume_text2func(f, s, template_func)
        pf.register_function(f, resume_mode=True)


def resume_hillclimb(hc: HillClimb):
    hc._resume_mode = True
    pf = hc._profiler
    assert pf is not None
    log_path = pf._log_dir
    template_func = hc._function_to_evolve
    # resume profiler
    _resume_pf(log_path, pf, template_func)

    # resume hillclimb
    funcs, scores, sample_max_order = _get_all_samples_and_scores(log_path)
    hc._tot_sample_nums = restore_attempt_count(log_path, sample_max_order, type(hc).__name__)

    def get_best_func():
        best_score = float('-inf')
        best_func = ...
        for f, s in zip(funcs, scores):
            if s is not None and s > best_score:
                best_score = s
                best_func = f
        if best_func is Ellipsis:
            raise ValueError('Checkpoint contains no finite best function.')
        function = tfpc.text_to_function(best_func)
        function.score = best_score
        return function

    hc._best_function_found = get_best_func()
