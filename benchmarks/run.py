# SPDX-License-Identifier: BSD-2-Clause
"""Measure explicit framework commands without assuming their interfaces match."""
from __future__ import annotations

import argparse
import json
import math
import re
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


def measure(command, result_file, timeout):
    """Measure one command; unavailable memory or metrics remain explicitly null."""
    import psutil
    result_file = Path(result_file)
    started = time.monotonic()
    peak = None
    memory_status = 'unavailable: no process sample'
    timed_out = False
    process = None
    family = []
    launch_error = None
    stopped = False

    def stop():
        nonlocal stopped
        if stopped:
            return
        stopped = True
        # The monitor may have become inaccessible or exited between samples.
        for member in reversed(family):
            try:
                member.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, PermissionError):
                pass
        if process is not None and process.poll() is None:
            process.kill()

    with result_file.with_suffix('.stdout.txt').open('w', encoding='utf-8') as stdout, result_file.with_suffix('.stderr.txt').open('w', encoding='utf-8') as stderr:
        try:
            try:
                process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            except OSError as error:
                launch_error = type(error).__name__
            if process is not None:
                while process.poll() is None:
                    try:
                        monitor = psutil.Process(process.pid)
                        family = [monitor] + monitor.children(recursive=True)
                        rss = 0
                        sampled = False
                        for member in family:
                            try:
                                rss += member.memory_info().rss
                                sampled = True
                            except psutil.NoSuchProcess:
                                pass
                        if sampled:
                            peak = max(peak or 0, rss)
                            if 'denied' not in memory_status:
                                memory_status = 'sampled'
                    except psutil.NoSuchProcess:
                        pass
                    except (PermissionError, psutil.AccessDenied):
                        memory_status = 'unavailable: process inspection denied'
                    if time.monotonic() - started > timeout:
                        timed_out = True
                        stop()
                        break
                    time.sleep(0.02)
        finally:
            if process is not None:
                stop()
                process.wait()
    elapsed = time.monotonic() - started
    metrics = None
    metrics_error = None
    count = None
    if result_file.exists():
        try:
            def reject_constant(value):
                raise ValueError('Non-finite JSON value: ' + value)
            metrics = json.loads(result_file.read_text(encoding='utf-8'), parse_constant=reject_constant)
            if not isinstance(metrics, dict):
                raise ValueError('Candidate metrics must be a JSON object')
            count = metrics.get('evaluated_candidates')
            if count is not None and (type(count) is not int or count < 0):
                raise ValueError('evaluated_candidates must be a nonnegative integer')
        except (OSError, ValueError) as error:
            metrics, count = None, None
            metrics_error = str(error)
    return {'exit_code': process.returncode if process is not None else None,
            'launch_error': launch_error, 'timed_out': timed_out,
            'wall_seconds': elapsed, 'peak_process_tree_rss_bytes': peak if memory_status == 'sampled' else None,
            'memory_status': memory_status, 'metrics_error': metrics_error,
            'rss_sampling_interval_seconds': 0.02, 'candidate_metrics': metrics,
            'throughput_candidates_per_second': count / elapsed if count is not None and elapsed > 0 else None}


def validate_config(config):
    """Reject incomplete or ambiguous benchmark configurations before any run."""
    if not isinstance(config, dict) or not isinstance(config.get('kind'), str):
        raise ValueError('Configuration requires a kind string')
    frameworks = config.get('frameworks')
    if not isinstance(frameworks, list) or not frameworks:
        raise ValueError('Configuration requires at least one framework')
    ids = set()
    for framework in frameworks:
        if not isinstance(framework, dict):
            raise ValueError('Each framework must be an object')
        name = framework.get('id')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) or name in ids:
            raise ValueError('Framework IDs must be unique, simple directory names')
        ids.add(name)
        command, revision = framework.get('command'), framework.get('revision')
        if (not isinstance(command, list) or not command or not all(isinstance(arg, str) for arg in command)
                or not isinstance(revision, str) or not revision.strip() or revision == 'UNPINNED'):
            raise ValueError('Every framework requires an explicit argv command and pinned revision')
    seeds = config.get('seeds')
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError('Seeds must be unique integers in [0, 2**32)')
    if type(config.get('budget')) is not int or config['budget'] < 1:
        raise ValueError('Budget must be a positive integer')
    timeout = config.get('timeout_seconds')
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Timeout must be finite and positive')


def expand_command(command, values):
    """Expand known placeholders while preserving literal JSON/code braces."""
    return [re.sub(r'\{(python|seed|budget|run_dir|result)\}', lambda match: str(values[match.group(1)]), arg)
            for arg in command]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    from llm4ad.tools.reproducibility import source_fingerprint
    source_sha256 = source_fingerprint()
    try:
        validate_config(config)
    except ValueError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error('Output report already exists')
    for framework in config['frameworks']:
        for seed in config['seeds']:
            if (args.output.parent / f"{framework['id']}-{seed}").exists():
                parser.error('A run directory already exists')
    records = []
    for framework in config['frameworks']:
        for seed in config['seeds']:
            run_dir = args.output.parent / f"{framework['id']}-{seed}"
            run_dir.mkdir(exist_ok=False)
            result = run_dir / 'metrics.json'
            values = {'python': sys.executable, 'seed': seed, 'budget': config['budget'],
                      'run_dir': str(run_dir.resolve()), 'result': str(result.resolve())}
            command = expand_command(framework['command'], values)
            records.append({'framework': framework['id'], 'revision': framework['revision'],
                            'seed': seed, 'command': command,
                            **measure(command, result, config['timeout_seconds'])})
            args.output.write_text(json.dumps({'schema_version': 1, 'kind': config['kind'],
                'config': config, 'platform': platform.platform(), 'python': platform.python_version(),
                'llm4ad_source_sha256': source_sha256,
                'hardware': {'machine': platform.machine(), 'processor': platform.processor(), 'logical_cpus': os.cpu_count()},
                'records': records}, indent=2, allow_nan=False), encoding='utf-8')
    if any(r['exit_code'] != 0 or r['timed_out'] or r['candidate_metrics'] is None for r in records):
        raise SystemExit('A command failed or omitted candidate metrics; see report')


if __name__ == '__main__':
    main()
