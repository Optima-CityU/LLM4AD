# SPDX-License-Identifier: BSD-2-Clause
"""Source inventories and public metadata must remain internally consistent."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import runpy
import unittest

import llm4ad

ROOT = Path(__file__).resolve().parents[1]


class MetadataTests(unittest.TestCase):
    def test_version_is_single_sourced(self):
        self.assertEqual(llm4ad.__version__, runpy.run_path(str(ROOT / 'llm4ad/_version.py'))['__version__'])
        self.assertIn(llm4ad.__version__, (ROOT / 'README.md').read_text())

    def test_task_inventory_maps_to_real_source_classes(self):
        data = json.loads((ROOT / 'llm4ad/task/catalogue.json').read_text())
        self.assertEqual(data['interface_count'], len(data['tasks']))
        self.assertEqual(len({row['id'] for row in data['tasks']}), data['interface_count'])
        for row in data['tasks']:
            with self.subTest(task=row['id']):
                path = ROOT / row['source_file']
                names = [n.name for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef)]
                self.assertIn(row['class'], names)
        self.assertEqual(sum(row['suite'] == 'CO-Bench' for row in data['tasks']), data['suite_interfaces']['CO-Bench'])

    def test_highlighted_task_descriptions_are_explicit(self):
        data = json.loads((ROOT / 'llm4ad/task/catalogue.json').read_text())
        row = next(row for row in data['tasks'] if row['id'].endswith('/online_bin_packing_2O'))
        self.assertIn('not two-orthant', row['verified_semantics'])


if __name__ == '__main__':
    unittest.main()
