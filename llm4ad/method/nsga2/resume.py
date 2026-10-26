# SPDX-License-Identifier: BSD-2-Clause
from __future__ import annotations

import copy

from tqdm.auto import tqdm

from ...tools.checkpoint import load_samples, latest_snapshot, restore_population, restore_attempt_count

from .nsga2 import NSGA2
from .profiler import NSGA2Profiler
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


def _resume_pf(log_path: str, pf: NSGA2Profiler, template_func):
    _, db_max_order = _get_latest_pop_json(log_path)
    funcs, scores, sample_max_order = _get_all_samples_and_scores(log_path)
    print(f'RESUME NSGA2: Sample order: {sample_max_order}.', flush=True)
    pf._prog_db_order = db_max_order
    # pf.__class__._num_samples = sample_max_order
    for i in tqdm(range(len(funcs)), desc='Resume NSGA2 Profiler'):  # noqa
        f, s = funcs[i], scores[i]  # noqa
        f = _resume_text2func(f, s, template_func)
        pf.register_function(f, resume_mode=True)


def resume_nsga2(nsga2: NSGA2):
    nsga2._resume_mode = True
    pf = nsga2._profiler
    log_path = pf._log_dir
    # resume program database
    pop = _resume_pop(log_path, nsga2._pop_size, template_program=nsga2._template_program)
    nsga2._population = pop
    # resume profiler
    template_func = nsga2._function_to_evolve
    _resume_pf(log_path, pf, template_func)
    # resume nsga2
    _, _, sample_max_order = _get_all_samples_and_scores(log_path)
    nsga2._tot_sample_nums = restore_attempt_count(log_path, sample_max_order, type(nsga2).__name__)
