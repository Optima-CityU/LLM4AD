# SPDX-License-Identifier: BSD-2-Clause
from __future__ import annotations

import copy
import json

from tqdm.auto import tqdm

from ...tools.checkpoint import load_samples, latest_snapshot, restore_attempt_count

from .funsearch import FunSearch
from .profiler import FunSearchProfiler
from .programs_database import ProgramsDatabase
from ...base import TextFunctionProgramConverter as tfpc, Function


def _get_latest_db_json(log_path: str):
    return latest_snapshot(log_path, 'prog_db', 'db')


def _get_all_samples_and_scores(path, get_algorithm=False):
    samples = load_samples(path)
    funcs = [row['function'] for row in samples]
    scores = [row['score'] for row in samples]
    maximum = max((row['sample_order'] for row in samples), default=0)
    if get_algorithm:
        return funcs, scores, maximum, [row.get('algorithm', '') for row in samples]
    return funcs, scores, maximum


def _resume_db(log_path: str, db_config, template, func_to_evol) -> ProgramsDatabase:
    # ======================================================================================================================
    # [
    #     [{'score': -300, 'functions': [xxx, xxx, xxx, ...]}, {'score': -200, 'functions': [xxx, xxx, xxx, ...]}, {...}],
    #     [{...}, {...}],
    # ]
    # ======================================================================================================================
    path, db_max_o = _get_latest_db_json(log_path)
    print(f'RESUME FunSearch: ProgramDB order: {db_max_o}.', flush=True)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    db = ProgramsDatabase(db_config, template, func_to_evol)
    for isld_id, island in enumerate(data):
        for cluster in island:
            score = cluster['score']
            funcs = cluster['functions']
            funcs = [tfpc.text_to_function(f) for f in funcs]
            for f in funcs:
                db.register_function(f, isld_id, score)
    return db


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


def _resume_pf(log_path: str, pf: FunSearchProfiler, template_func: Function):
    _, db_max_order = _get_latest_db_json(log_path)
    funcs, scores, sample_max_order = _get_all_samples_and_scores(log_path)
    print(f'RESUME FunSearch: Sample order: {sample_max_order}.', flush=True)
    pf._prog_db_order = db_max_order
    # pf.__class__._num_samples = sample_max_order
    for i in tqdm(range(len(funcs)), desc='Resume FunSearch Profiler'):  # noqa
        f, s = funcs[i], scores[i]  # noqa
        f = _resume_text2func(f, s, template_func)
        pf.register_function(f, resume_mode=True)


def resume_funsearch(fs: FunSearch):
    fs._resume_mode = True
    pf = fs._profiler  # noqa
    log_path = pf._log_dir  # noqa
    # resume program database
    template = tfpc.text_to_program(fs._template_program_str)  # noqa
    template_func = fs._function_to_evolve
    config = fs._config.programs_database  # noqa
    func_to_evol = fs._function_to_evolve_name  # noqa
    db = _resume_db(log_path, config, template, func_to_evol)
    fs._database = db
    # resume profiler
    _resume_pf(log_path, pf, template_func)
    # resume funsearch
    _, _, sample_max_order = _get_all_samples_and_scores(log_path)
    fs._tot_sample_nums = restore_attempt_count(log_path, sample_max_order, type(fs).__name__)
