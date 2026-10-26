# SPDX-License-Identifier: BSD-2-Clause
"""Bounded candidate search: local model -> code -> task score -> offline replay.

The fixture backend checks plumbing only; it is not an open-model benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from llm4ad.base import LLM, SampleTrimmer, SecureEvaluator
from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
from llm4ad.tools.llm.ollama_http import OllamaHTTP
from llm4ad.tools.llm.replay import RecordingLLM, ReplayLLM
from llm4ad.tools.reproducibility import seed_all, write_run_manifest


class FixtureLLM(LLM):
    """Deterministic integration fixture; no learned model is involved."""
    def draw_sample(self, prompt, *args, **kwargs):
        return 'def priority(item, bins):\n    return -bins\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['fixture', 'ollama'], default='fixture')
    parser.add_argument('--model')
    parser.add_argument('--model-digest', help='Required immutable digest for an Ollama run')
    parser.add_argument('--endpoint', default='http://127.0.0.1:11434')
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('--samples must be positive')
    if not args.replay and args.backend == 'ollama' and (not args.model or not args.model_digest):
        parser.error('Ollama runs require --model and --model-digest')
    seed_all(args.seed)
    task = OBPEvaluation(n_instances=2, n_items=100, capacity=100)
    dataset_text = json.dumps(task._datasets, sort_keys=True, allow_nan=False,
                              default=lambda value: value.tolist())
    run_config = {
        'samples': args.samples, 'workers': 1, 'task': 'online_bin_packing',
        'n_instances': 2, 'n_items': 100, 'capacity': 100,
        'dataset_generator_seed': 2024,
    }
    run_metadata = {'config': run_config, 'seed': args.seed,
                    'dataset_sha256': hashlib.sha256(dataset_text.encode('utf-8')).hexdigest()}
    backend = None
    llm = None
    results = []
    try:
        if args.replay:
            llm = ReplayLLM(args.replay)
            if llm.run_metadata != run_metadata:
                parser.error('Replay task, dataset, seed or sample budget differs from the recording (or run metadata is missing)')
            model = llm.model_metadata
        else:
            if args.backend == 'ollama':
                backend = OllamaHTTP(args.model, args.endpoint, seed=args.seed)
                model = backend.model_info()
                if model['digest'] != args.model_digest:
                    parser.error('Installed model digest does not match --model-digest')
            else:
                backend = FixtureLLM()
                model = {'model': 'deterministic-fixture', 'revision': '1', 'scientific_model': False}
        # Complete preflight validation before creating experiment output.
        args.output.mkdir(parents=True, exist_ok=False)
        if llm is None:
            llm = RecordingLLM(backend, args.output / 'responses.jsonl',
                               model_metadata=model, run_metadata=run_metadata)
        datasets = args.output / 'datasets.json'
        datasets.write_text(dataset_text, encoding='utf-8')
        evaluator = SecureEvaluator(task)
        prompt = ('Return only the Python priority function. Minimize used bins.\n'
                  + task.task_description + '\n' + task.template_program)
        for order in range(args.samples):
            response = llm.draw_sample(prompt)
            program = SampleTrimmer.sample_to_program(SampleTrimmer.auto_trim(response), task.template_program)
            score = None if program is None else evaluator.evaluate_program(program)
            results.append({'sample_order': order + 1, 'program': str(program) if program else None,
                            'score': None if score is None else float(score)})
        if args.replay and llm.remaining:
            raise ValueError('Replay did not consume the complete recording; check --samples')
        if args.backend == 'ollama' and backend is not None and backend.model_info() != model:
            raise ValueError('Model metadata changed during the run')
        (args.output / 'results.json').write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf-8')
        write_run_manifest(args.output / 'manifest.json', config={
            **run_config, 'replay': str(args.replay) if args.replay else None,
        }, model=model, seed=args.seed, datasets=[datasets],
            repository=Path(__file__).resolve().parents[2] if (Path(__file__).resolve().parents[2] / '.git').exists() else None)
    finally:
        if llm is not None:
            llm.close()
        elif backend is not None:
            backend.close()
    print(json.dumps({'mode': 'replay' if args.replay else args.backend,
                      'samples': len(results), 'valid': sum(r['score'] is not None for r in results),
                      'best_score': max((r['score'] for r in results if r['score'] is not None), default=None)}))


if __name__ == '__main__':
    main()
