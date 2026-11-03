# SPDX-License-Identifier: BSD-2-Clause
from __future__ import annotations

import copy

from tqdm.auto import tqdm

from ...tools.checkpoint import load_samples, latest_snapshot, restore_population, restore_attempt_count

from .reevo import ReEvo
from .profiler import ReEvoProfiler
from .population import Population
from ...base import TextFunctionProgramConverter as tfpc, Function


def _get_latest_pop_json(log_path: str):
    return latest_snapshot(log_path)


def _get_all_samples_and_scores(path, get_algorithm=True):
    samples = load_samples(path)
    funcs = [row['function'] for row in samples]
    scores = [row['score'] for row in samples]
    maximum = max((row['sample_order'] for row in samples), default=0)
    if get_algorithm:
        return funcs, scores, maximum, [row.get('algorithm', '') for row in samples]
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


def _resume_pf(log_path: str, pf: ReEvoProfiler, template_func):
    _, db_max_order = _get_latest_pop_json(log_path)
    funcs, scores, sample_max_order, algorithms = _get_all_samples_and_scores(log_path)
    print(f'RESUME ReEvo: Sample order: {sample_max_order}.', flush=True)
    pf._prog_db_order = db_max_order
    # pf.__class__._num_samples = sample_max_order
    for i in tqdm(range(len(funcs)), desc='Resume ReEvo Profiler'):  # noqa
        f, s, algo = funcs[i], scores[i], algorithms[i]
        f = _resume_text2func(f, s, template_func)
        f.algorithm = algo
        pf.register_function(f, resume_mode=True)


def resume_reevo(reevo: ReEvo, path):
    reevo._resume_mode = True
    pf = reevo._profiler
    log_path = path
    # resume program database
    pop = _resume_pop(log_path, reevo._pop_size, template_program=reevo._template_program)
    reevo._population = pop
    # resume profiler
    template_func = reevo._function_to_evolve
    _resume_pf(log_path, pf, template_func)
    # resume reevo
    _, _, sample_max_order, _ = _get_all_samples_and_scores(log_path)
    reevo._tot_sample_nums = restore_attempt_count(log_path, sample_max_order, type(reevo).__name__)
