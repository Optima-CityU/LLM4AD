# SPDX-License-Identifier: BSD-2-Clause
"""Build an import-free task-interface inventory; never download datasets."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'llm4ad/task/catalogue.json'
DOCUMENT = ROOT / 'docs/task_catalogue.md'

VERIFIED = {
    'optimization/online_bin_packing_2O': (
        'Two-objective online bin packing',
        'Minimize mean bin count and mean evaluation wall-clock time; both returned with negative signs. '
        '2O means two objectives, not two-orthant constraints. Runtime is hardware-dependent.'),
    'optimization/admissible_set': (
        'Priority search for symmetric constant-weight admissible sets',
        'Score candidate vectors to construct a pre-admissible set, then expand the symmetric groups. '
        'Maximize the expanded set cardinality minus the configured reference cardinality; see TRIPLES and surviving-child checks in the evaluator.'),
    'optimization/bp_2d_construct': ('Constructive rectangular 2D bin packing',
        'Place every item without overlap inside finite bins; maximize negative mean used-bin count. Incomplete packing returns None.'),
    'optimization/cflp_construct': ('Constructive capacitated facility assignment',
        'Assign every customer; maximize negative mean assignment cost. Incomplete assignment returns None. '
        'The current objective sums assignment costs, not a separate facility-opening cost.'),
    'optimization/set_cover_construct': ('Constructive set cover',
        'Cover every universal-set element using supplied subsets; maximize negative mean number of selected subsets.'),
    'optimization/pymoo_moead': ('Pymoo MOEA/D decomposition design',
        'Design a decomposition function for MOEA/D on generated DTLZ4 instances; maximize hypervolume.'),
    'optimization/co_bench/open_shop_scheduling_co_bench': ('CO-Bench open shop scheduling',
        'Respect job/machine non-overlap; maximize mean lower-bound/makespan ratio.'),
    'optimization/co_bench/p_median_capacitated_co_bench': ('CO-Bench capacitated p-median',
        'Select p customer medians and feasible capacity-limited assignments; maximize mean best-known/computed-cost ratio.'),
}


def catalogue():
    entries = []
    for path in sorted((ROOT / 'llm4ad/task').rglob('evaluation.py')):
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        identifier = path.parent.relative_to(ROOT / 'llm4ad/task').as_posix()
        suite = 'CO-Bench' if '/co_bench/' in '/' + identifier else (
            'Feynman SRSD' if identifier.endswith('/feynman_srsd') else None)
        for cls in tree.body:
            if not isinstance(cls, ast.ClassDef) or not any(isinstance(b, ast.Name) and b.id == 'Evaluation' for b in cls.bases):
                continue
            init = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__'), None)
            template_path = path.parent / 'template.py'
            description = None
            if template_path.exists():
                for node in ast.parse(template_path.read_text(encoding='utf-8-sig')).body:
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'task_description' for t in node.targets):
                        try:
                            description = ast.literal_eval(node.value)
                        except (ValueError, TypeError):
                            pass
            title, semantics = VERIFIED.get(identifier, (identifier.rsplit('/', 1)[-1].replace('_', ' '), None))
            entries.append({
                'id': identifier, 'class': cls.name,
                'module': '.'.join(path.relative_to(ROOT).with_suffix('').parts),
                'source_file': path.relative_to(ROOT).as_posix(), 'title': title,
                'suite': suite, 'counting_unit': 'task interface',
                'signature': ast.unparse(init.args) if init else '',
                'description_from_source': description,
                'verified_semantics': semantics,
                'semantic_review': 'checked against evaluator' if semantics else 'pending manual review',
            })
    equations = ast.parse((ROOT / 'llm4ad/task/science_discovery/feynman_srsd/feynman_equations.py').read_text())
    equation_count = sum(isinstance(n, ast.ClassDef) and any(
        isinstance(d, ast.Name) and d.id == 'register_feynman_eq_class' for d in n.decorator_list) for n in equations.body)
    return {'schema_version': 1, 'counting_rule': 'One Evaluation subclass = one interface; equations and instances are separate.',
            'interface_count': len(entries), 'suite_interfaces': dict(Counter(e['suite'] or 'Other' for e in entries)),
            'feynman_equation_definitions': equation_count, 'tasks': entries}


def outputs():
    data = catalogue()
    text = '# Task catalogue\n\nGenerated from source by `scripts/build_task_catalogue.py`.\n\n'
    text += (f"There are **{data['interface_count']} task interfaces**, including "
             f"**{data['suite_interfaces'].get('CO-Bench', 0)} CO-Bench interfaces** and "
             f"**{data['feynman_equation_definitions']} registered Feynman equation definitions** "
             'behind the Feynman interface. These are different counting units.\n\n'
             'The earlier “100+ tasks” description includes wrapped external suites; it must '
             'not be read as 100+ independently implemented native task interfaces. Dataset '
             'instances, equation IDs and parameter variations are not added to the interface count.\n\n'
             'This inventory does not certify semantic correctness or dataset licensing. '
             'Rows pending manual review must not be represented as validated benchmark definitions. '
             'AILS2 is not identified by that name in the current source tree; any external reference '
             'to it must be mapped to a concrete implementation.\n\n'
             '| Interface | Class | Suite | Semantic review |\n| --- | --- | --- | --- |\n')
    for entry in data['tasks']:
        text += f"| `{entry['id']}` | `{entry['class']}` | {entry['suite'] or 'Other'} | {entry['semantic_review']} |\n"
    text += '\n## Checked definitions\n\n'
    for entry in data['tasks']:
        if entry['verified_semantics']:
            text += f"- **{entry['title']}**: {entry['verified_semantics']}\n"
    return {OUTPUT: json.dumps(data, indent=2, ensure_ascii=False) + '\n', DOCUMENT: text}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for path, content in outputs().items():
        if args.check:
            if not path.exists() or path.read_text(encoding='utf-8') != content:
                raise SystemExit(f'Outdated generated file: {path}')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')


if __name__ == '__main__':
    main()
