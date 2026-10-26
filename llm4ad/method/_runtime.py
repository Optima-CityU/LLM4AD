# SPDX-License-Identifier: BSD-2-Clause
"""Shared search bookkeeping; operators and population policies stay in methods."""
from __future__ import annotations

from functools import wraps
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from threading import Event, Lock, Thread
from weakref import WeakSet

import numpy as np

from llm4ad.base.errors import FatalSearchError

_creation_lock = Lock()


class _SearchStopped(Exception):
    """Internal normal termination when a candidate budget has been claimed."""


class _SearchState:
    def __init__(self):
        self.lock = Lock()
        self.stopped = Event()
        self.claimed = 0
        self.error = None
        self.threads = WeakSet()

    def fail(self, error):
        with self.lock:
            if self.error is None:
                self.error = error
            self.stopped.set()


def _state(owner):
    with _creation_lock:
        if not hasattr(owner, '_search_state'):
            owner._search_state = _SearchState()
        return owner._search_state


def _is_stopped(owner):
    return _state(owner).stopped.is_set()


def _write_search_state(owner, status='running'):
    """Caller holds the state lock. Save counters separately from sample history."""
    profiler = getattr(owner, '_profiler', None)
    directory = getattr(profiler, '_log_dir', None)
    if not directory:
        return
    state = _state(owner)
    payload = {'schema_version': 1, 'method': type(owner).__name__, 'status': status,
               'completed_attempts': owner._tot_sample_nums, 'claimed_attempts': state.claimed,
               'history_samples': profiler._num_samples}
    target = Path(directory) / 'search_state.json'
    temporary = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix='.search-state-', dir=target.parent)
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(payload, stream, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, target)
    except (OSError, ValueError) as error:
        raise FatalSearchError('Unable to save search counters') from error
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def _claim_sample(owner, count=1):
    state = _state(owner)
    with state.lock:
        if state.stopped.is_set():
            raise _SearchStopped()
        state.claimed = max(state.claimed, owner._tot_sample_nums)
        limit = owner._max_sample_nums
        remaining = count if limit is None else min(count, max(0, limit - state.claimed))
        if remaining == 0:
            raise _SearchStopped()
        state.claimed += remaining
        _write_search_state(owner)
        return remaining


def _complete_sample(owner, count=1):
    state = _state(owner)
    with state.lock:
        owner._tot_sample_nums += count
        _write_search_state(owner)


@contextmanager
def _sample_attempt(owner):
    """Reserve and complete one candidate slot, including local seed evaluation."""
    _claim_sample(owner)
    try:
        yield
    finally:
        _complete_sample(owner)


def _count_sample_attempt(function):
    """Bound concurrent candidate attempts, including parse/evaluation failures."""
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        with _sample_attempt(self):
            return function(self, *args, **kwargs)
    return wrapped


def _draw_sample_batch(owner, prompts):
    """Reserve a bounded batch while retaining the backend's batch interface."""
    count = _claim_sample(owner, len(prompts))
    try:
        return owner._sampler.draw_samples(prompts[:count])
    finally:
        _complete_sample(owner, count)


class _SearchThread(Thread):
    """Return worker failures to run(), and signal other workers to stop."""
    def __init__(self, *, owner, **kwargs):
        super().__init__(**kwargs)
        self._owner = owner
        _state(owner).threads.add(self)

    def run(self):
        try:
            super().run()
        except _SearchStopped:
            pass
        except BaseException as error:
            _state(self._owner).fail(error)


def _raise_worker_error(owner):
    error = _state(owner).error
    if error is not None:
        raise error


def _managed_run(function):
    """Clean up on success, early return and failure; surface worker exceptions."""
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        state = _state(self)
        limit = getattr(self, '_max_sample_nums', None)
        error = None
        result = None
        try:
            if limit is not None and (type(limit) is not int or limit < 0):
                raise ValueError('max_sample_nums must be a nonnegative integer or None')
            with state.lock:
                state.claimed = max(state.claimed, getattr(self, '_tot_sample_nums', 0))
            if getattr(self, '_resume_mode', False):
                population = getattr(self, '_population', None)
                if population is None:
                    population = getattr(self, '_pool', None)
                if population is not None:
                    saved = list(population.population)
                    if not any(f.score is not None and np.all(np.isfinite(f.score)) for f in saved):
                        raise ValueError('Resume mode requires a restored population with a finite score')
                best = getattr(self, '_best_function_found', None)
                if best is not None and best.score is None:
                    raise ValueError('Resume mode requires a restored best function and score')
            result = function(self, *args, **kwargs)
        except _SearchStopped:
            pass
        except BaseException as caught:
            error = caught
        finally:
            state.stopped.set()
            for thread in list(state.threads):
                if thread.ident is not None:
                    thread.join()
            error = state.error or error
            sampler = getattr(self, '_sampler', None)
            llm = getattr(sampler, 'llm', None) or getattr(self, '_llm', None)
            profiler = getattr(self, '_profiler', None)
            executor = getattr(self, '_evaluation_executor', None)
            cleanups = []
            if profiler is not None and getattr(profiler, '_log_dir', None):
                population = getattr(self, '_population', None)
                if population is None:
                    population = getattr(self, '_pool', None)
                if population is not None and hasattr(profiler, 'register_population'):
                    cleanups.append(lambda: profiler.register_population(population))
                database = getattr(self, '_database', None)
                if database is not None and hasattr(profiler, 'register_program_db'):
                    cleanups.append(lambda: profiler.register_program_db(database))
            if executor is not None:
                cleanups.append(lambda: executor.shutdown(wait=True, cancel_futures=True))
            if profiler is not None:
                cleanups.append(profiler.finish)
            if llm is not None:
                cleanups.append(llm.close)
            for cleanup in cleanups:
                try:
                    cleanup()
                except BaseException as caught:
                    if error is None:
                        error = caught
            if hasattr(self, '_tot_sample_nums'):
                try:
                    with state.lock:
                        _write_search_state(self, 'failed' if error is not None else 'completed')
                except BaseException as caught:
                    if error is None:
                        error = caught
        if error is not None:
            raise error
        return result
    return wrapped
