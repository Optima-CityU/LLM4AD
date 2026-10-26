# SPDX-License-Identifier: BSD-2-Clause
from __future__ import annotations

import copy

from tqdm.auto import tqdm

from ...tools.checkpoint import load_samples, latest_snapshot, restore_population, restore_attempt_count

from .meoh import MEoH
from .profiler import MEoHProfiler
from .population import Population
from ...base import TextFunctionProgramConverter as tfpc, Function


def _get_latest_pop_json(log_path: str):
    return latest_snapshot(log_path)


def _get_all_samples_and_scores(path):
    samples = load_samples(path)
    funcs = [row['function'] for row in samples]
    scores = [row['score'] for row in samples]
    maximum = max((row['sample_order'] for row in samples), default=0)
    return funcs, scores, maximum


def _resume_pop(log_path: str, pop_size, template_program=None) -> Population:
    return restore_population(Population, log_path, pop_size, template_program=template_program)


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


def _resume_pf(log_path: str, pf: MEoHProfiler, template_func):
    _, db_max_order = _get_latest_pop_json(log_path)
    funcs, scores, sample_max_order = _get_all_samples_and_scores(log_path)
    print(f'RESUME MEoH: Sample order: {sample_max_order}.', flush=True)
    pf._prog_db_order = db_max_order
    # pf.__class__._num_samples = sample_max_order
    for i in tqdm(range(len(funcs)), desc='Resume MEoH Profiler'):  # noqa
        f, s = funcs[i], scores[i]  # noqa
        f = _resume_text2func(f, s, template_func)
        pf.register_function(f, resume_mode=True)


def resume_meoh(meoh: MEoH):
    meoh._resume_mode = True
    pf = meoh._profiler
    log_path = pf._log_dir
    # resume program database
    pop = _resume_pop(log_path, meoh._pop_size, template_program=meoh._template_program)
    meoh._population = pop
    # resume profiler
    template_func = meoh._function_to_evolve
    _resume_pf(log_path, pf, template_func)
    # resume meoh
    _, _, sample_max_order = _get_all_samples_and_scores(log_path)
    meoh._tot_sample_nums = restore_attempt_count(log_path, sample_max_order, type(meoh).__name__)
