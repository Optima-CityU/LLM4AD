# SPDX-License-Identifier: BSD-2-Clause
"""Offline regression checks for constructive feasibility and score direction.

Run with: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from llm4ad.task.optimization.bp_2d_construct.evaluation import BP2DEvaluation
from llm4ad.task.optimization.cflp_construct.evaluation import CFLPEvaluation
from llm4ad.task.optimization.set_cover_construct.evaluation import SCPEvaluation


class ConstructiveCompletionTests(unittest.TestCase):
    def test_set_cover_rejects_early_stop(self):
        evaluator = SCPEvaluation.__new__(SCPEvaluation)
        evaluator.n_instance = 1
        evaluator._datasets = [([1, 2], [[1], [2]])]
        for completed_steps in (0, 1):
            with self.subTest(completed_steps=completed_steps):
                decisions = iter([[1]][:completed_steps] + [None])
                self.assertIsNone(evaluator.evaluate(lambda *args: next(decisions)))

    def test_set_cover_preserves_complete_solution_score(self):
        evaluator = SCPEvaluation.__new__(SCPEvaluation)
        evaluator.n_instance = 1
        evaluator._datasets = [([1, 2], [[1], [2]])]
        self.assertEqual(evaluator.evaluate(lambda selected, remaining, uncovered: remaining[0]), -2)

    def test_set_cover_accepts_empty_universe(self):
        evaluator = SCPEvaluation.__new__(SCPEvaluation)
        evaluator.n_instance = 1
        evaluator._datasets = [([], [])]
        heuristic = Mock(side_effect=AssertionError("No selection is needed"))
        self.assertEqual(evaluator.evaluate(heuristic), 0)
        heuristic.assert_not_called()

    def test_set_cover_ignores_uncoverable_elements(self):
        evaluator = SCPEvaluation.__new__(SCPEvaluation)
        evaluator.n_instance = 1
        evaluator._datasets = [([1, 2, 3], [[1], [2]])]
        greedy = lambda selected, remaining, uncovered: next(
            (s for s in remaining if set(s) & set(uncovered)), None)
        self.assertEqual(evaluator.evaluate(greedy), -2)

    def test_set_cover_default_template_is_feasible(self):
        from llm4ad.task.optimization.set_cover_construct.template import template_program
        namespace = {}
        exec("from typing import List\n" + template_program, namespace)
        self.assertEqual(SCPEvaluation().evaluate(namespace["select_next_subset"]), -13.8125)

    def test_cflp_rejects_early_stop(self):
        evaluator = CFLPEvaluation.__new__(CFLPEvaluation)
        evaluator.n_instance = 1
        evaluator._datasets = [{
            "facility_capacities": [2],
            "customer_demands": [1, 1],
            "assignment_costs": [[3, 7]],
        }]
        for completed_steps in (0, 1):
            with self.subTest(completed_steps=completed_steps):
                decisions = iter([(0, 0)][:completed_steps] + [(None, None)])
                self.assertIsNone(evaluator.evaluate_cflp(lambda *args: next(decisions)))

    def test_cflp_preserves_complete_solution_score(self):
        evaluator = CFLPEvaluation.__new__(CFLPEvaluation)
        evaluator.n_instance = 1
        evaluator._datasets = [{
            "facility_capacities": [2],
            "customer_demands": [1, 1],
            "assignment_costs": [[3, 7]],
        }]
        heuristic = lambda assignments, remaining, capacities, demands, costs: (remaining[0], 0)
        self.assertEqual(evaluator.evaluate_cflp(heuristic), -10)

    def test_invalid_instance_invalidates_entire_evaluation(self):
        evaluator = SCPEvaluation.__new__(SCPEvaluation)
        evaluator.n_instance = 2
        evaluator._datasets = [([1], [[1]]), ([2], [[2]])]
        decisions = iter([[1], None])
        self.assertIsNone(evaluator.evaluate(lambda *args: next(decisions)))


class BinPackingFeasibilityTests(unittest.TestCase):
    def setUp(self):
        self.evaluator = BP2DEvaluation.__new__(BP2DEvaluation)

    @staticmethod
    def first_item_first_bin(remaining, matrices):
        return remaining[0], 0

    def assert_valid_packing(self, result, items, dimensions):
        self.assertIsNotNone(result)
        used_bins, bins = result
        self.assertEqual(used_bins, sum(bool(contents) for contents in bins))
        packed_items = []
        for contents in bins:
            occupied = set()
            for (x, y), (width, height) in contents:
                self.assertGreaterEqual(x, 0)
                self.assertGreaterEqual(y, 0)
                self.assertLessEqual(x + width, dimensions[0])
                self.assertLessEqual(y + height, dimensions[1])
                cells = {(i, j) for i in range(x, x + width) for j in range(y, y + height)}
                self.assertTrue(occupied.isdisjoint(cells), "Packed rectangles must not overlap")
                occupied.update(cells)
                packed_items.append((width, height))
        self.assertCountEqual(packed_items, items)

    def test_rejects_early_stop(self):
        self.evaluator.n_instance = 1
        self.evaluator.n_bins = 2
        self.evaluator._datasets = [([(1, 1), (1, 1)], (2, 2))]
        for completed_steps in (0, 1):
            with self.subTest(completed_steps=completed_steps):
                decisions = iter([((1, 1), 0)][:completed_steps] + [((1, 1), None)])
                self.assertIsNone(self.evaluator.evaluate_2d(lambda *args: next(decisions)))

    def test_does_not_discard_unplaceable_item(self):
        for items, dimensions in [([(2, 2), (1, 1)], (2, 2)), ([(3, 1)], (2, 2))]:
            with self.subTest(items=items):
                result = self.evaluator.pack_items_2d(items, dimensions, self.first_item_first_bin, 1)
                self.assertIsNone(result)

    def test_failed_placement_propagates_to_score(self):
        self.evaluator.n_instance = 1
        self.evaluator.n_bins = 1
        self.evaluator._datasets = [([(2, 2), (1, 1)], (2, 2))]
        self.assertIsNone(self.evaluator.evaluate_2d(self.first_item_first_bin))

    def test_fallback_uses_another_bin_without_overlapping(self):
        # The last rectangle's corners are free in bin 0, but its interior is occupied.
        items = [(3, 4), (2, 2), (1, 3), (5, 1)]
        result = self.evaluator.pack_items_2d(items, (5, 5), self.first_item_first_bin, 2)
        self.assert_valid_packing(result, items, (5, 5))
        self.assertEqual(result[0], 2)

    def test_rejects_packing_that_requires_overlap(self):
        items = [(3, 4), (2, 2), (1, 3), (5, 1)]
        result = self.evaluator.pack_items_2d(items, (5, 5), self.first_item_first_bin, 1)
        self.assertIsNone(result)

    def test_fallback_preserves_all_duplicate_items(self):
        items = [(2, 2), (2, 2)]
        result = self.evaluator.pack_items_2d(items, (2, 2), self.first_item_first_bin, 2)
        self.assert_valid_packing(result, items, (2, 2))
        self.assertEqual(result[0], 2)

    def test_complete_solution_score_and_touching_edges(self):
        self.evaluator.n_instance = 1
        self.evaluator.n_bins = 1
        items = [(1, 2), (1, 2)]
        self.evaluator._datasets = [(items, (2, 2))]
        result = self.evaluator.pack_items_2d(items, (2, 2), self.first_item_first_bin, 1)
        self.assert_valid_packing(result, items, (2, 2))
        self.assertEqual(self.evaluator.evaluate_2d(self.first_item_first_bin), -1)


def load_offline_co_bench_evaluator(task_name, class_name):
    """Load the actual evaluator with only its dataset-download dependency replaced."""
    utils_name = "llm4ad.task.optimization.co_bench.utils"
    utils = ModuleType(utils_name)
    utils.load_subdir_as_text = Mock(side_effect=AssertionError("Dataset downloads are disabled"))
    path = (Path(__file__).resolve().parents[1] / "llm4ad/task/optimization/co_bench"
            / task_name / "evaluation.py")
    spec = importlib.util.spec_from_file_location(f"_regression_{task_name}", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {utils_name: utils}):
        spec.loader.exec_module(module)
    return getattr(module, class_name).__new__(getattr(module, class_name))


class ScoreDirectionTests(unittest.TestCase):
    def test_open_shop_prefers_shorter_makespan(self):
        evaluator = load_offline_co_bench_evaluator("open_shop_scheduling_co_bench", "OSSEvaluationCB")
        case = {
            "n_jobs": 1, "n_machines": 1, "times": [[10]], "machines": [[0]],
            "lower_bound": 10, "upper_bound": 10,
        }
        evaluator._datasets = {"case": "unused"}
        evaluator.load_data = lambda text: [case]
        faster = evaluator.evaluate_program("", lambda *args: {"start_times": [[0]]})
        slower = evaluator.evaluate_program("", lambda *args: {"start_times": [[10]]})
        self.assertAlmostEqual(faster, 1.0)
        self.assertAlmostEqual(slower, 0.5)
        self.assertGreater(faster, slower)

    def test_capacitated_p_median_prefers_lower_cost(self):
        evaluator = load_offline_co_bench_evaluator("p_median_capacitated_co_bench", "PMCEvaluationCB")
        case = {
            "best_known": 11, "n": 3, "p": 1, "Q": 10,
            "customers": [(1, 0, 0, 1), (2, 1, 0, 1), (3, 10, 0, 1)],
        }
        evaluator._datasets = {"case": "unused"}
        evaluator.load_data = lambda text: [case]
        cheaper = evaluator.evaluate_program("", lambda *args: {
            "objective": 11, "medians": [1], "assignments": [1, 1, 1],
        })
        costlier = evaluator.evaluate_program("", lambda *args: {
            "objective": 19, "medians": [3], "assignments": [3, 3, 3],
        })
        self.assertAlmostEqual(cheaper, 1.0)
        self.assertAlmostEqual(costlier, 11 / 19)
        self.assertGreater(cheaper, costlier)

    @unittest.skipUnless(importlib.util.find_spec("pymoo"), "Requires the optional pymoo dependency")
    def test_moead_prefers_larger_hypervolume(self):
        module = importlib.import_module("llm4ad.task.optimization.pymoo_moead.evaluation")
        evaluator = module.MOEAD_PYMOO_Evaluation(n_partitions=5, n_gen=1, seed=1)
        scores = []
        for objective_value in (0.8, 0.2):
            front = np.full((1, 3), objective_value)
            result = SimpleNamespace(opt=SimpleNamespace(get=lambda key: front))
            with patch.object(module, "minimize", return_value=result):
                score = evaluator.evaluate_program("", None)
            self.assertAlmostEqual(score, (1.1 - objective_value) ** 3)
            self.assertEqual(score, evaluator.last_result["hv"])
            scores.append(score)
        self.assertGreater(scores[1], scores[0])


if __name__ == "__main__":
    unittest.main()
