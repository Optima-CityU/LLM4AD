# SPDX-License-Identifier: BSD-2-Clause
"""Real EoH/OBP with a deterministic fake LLM, not an algorithm benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm4ad.base import LLM
from llm4ad.method.eoh import EoH
from llm4ad.task.optimization.online_bin_packing import OBPEvaluation
from llm4ad.tools.llm.replay import RecordingLLM
from llm4ad.tools.reproducibility import seed_all


class FixtureLLM(LLM):
    def draw_sample(self, prompt, *args, **kwargs):
        return '{Best fit fixture}\ndef priority(item, bins):\n    return -bins\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--budget', type=int, required=True)
    parser.add_argument('--result', type=Path, required=True)
    args = parser.parse_args()
    task = OBPEvaluation(n_instances=2, n_items=100, capacity=100)
    seed_all(args.seed)
    llm = RecordingLLM(FixtureLLM(), args.result.with_suffix('.responses.jsonl'),
                       model_metadata={'model': 'deterministic-fixture', 'revision': '1'})
    method = EoH(llm, task, profiler=None, pop_size=2, max_sample_nums=args.budget,
                 max_generations=None, num_samplers=1, num_evaluators=1)
    method.run()
    args.result.write_text(json.dumps({'quality': max(f.score for f in method._population.population),
        'evaluated_candidates': method._tot_sample_nums, 'llm_calls': method._tot_sample_nums,
        'input_tokens': None, 'output_tokens': None, 'failures': 0,
        'model': 'deterministic-fixture', 'task': 'OBP 2 instances x 100 items',
        'scientific_comparison': False}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
