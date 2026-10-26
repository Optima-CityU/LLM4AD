# SPDX-License-Identifier: BSD-2-Clause
"""Benchmark transport failure handling without launching external frameworks."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


@unittest.skipUnless(importlib.util.find_spec('psutil'), 'Optional benchmark dependency psutil is not installed')
class BenchmarkRunnerTests(unittest.TestCase):
    def test_denied_memory_inspection_is_reported_not_fabricated(self):
        from benchmarks.run import measure
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / 'metrics.json'
            result.write_text(json.dumps({'evaluated_candidates': 3}), encoding='utf-8')
            process = Mock(pid=123, returncode=0)
            process.poll.side_effect = [None, 0, 0]
            monitor = Mock()
            monitor.children.side_effect = PermissionError('denied')
            with patch('benchmarks.run.subprocess.Popen', return_value=process), patch('psutil.Process', return_value=monitor), patch('benchmarks.run.time.sleep'):
                metrics = measure(['fixture'], result, 5)
            self.assertIsNone(metrics['peak_process_tree_rss_bytes'])
            self.assertIn('denied', metrics['memory_status'])
            self.assertEqual(metrics['exit_code'], 0)

    def test_timeout_kills_and_waits_for_workload(self):
        from benchmarks.run import measure
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / 'metrics.json'
            process = Mock(pid=123, returncode=-9)
            process.poll.return_value = None
            monitor = Mock()
            monitor.children.return_value = []
            monitor.memory_info.return_value.rss = 100
            with patch('benchmarks.run.subprocess.Popen', return_value=process), patch('psutil.Process', return_value=monitor), patch('benchmarks.run.time.monotonic', side_effect=[0, 2, 3]):
                metrics = measure(['fixture'], result, 1)
            self.assertTrue(metrics['timed_out'])
            process.kill.assert_called_once()
            process.wait.assert_called_once()
            self.assertIsNone(metrics['candidate_metrics'])

    def test_missing_count_does_not_fabricate_zero_throughput(self):
        from benchmarks.run import measure
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / 'metrics.json'
            result.write_text(json.dumps({'best_score': 1}), encoding='utf-8')
            process = Mock(pid=123, returncode=0)
            process.poll.return_value = 0
            with patch('benchmarks.run.subprocess.Popen', return_value=process):
                metrics = measure(['fixture'], result, 5)
            self.assertIsNone(metrics['throughput_candidates_per_second'])
            self.assertIsNone(metrics['peak_process_tree_rss_bytes'])

    def test_corrupt_metrics_are_reported(self):
        from benchmarks.run import measure
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / 'metrics.json'
            process = Mock(pid=123, returncode=0)
            process.poll.return_value = 0
            for payload in ['broken', '[]', '{"evaluated_candidates": -1}', '{"best_score": NaN}']:
                result.write_text(payload, encoding='utf-8')
                with patch('benchmarks.run.subprocess.Popen', return_value=process):
                    metrics = measure(['fixture'], result, 5)
                self.assertIsNone(metrics['candidate_metrics'])
                self.assertTrue(metrics['metrics_error'])

    def test_fast_exit_and_launch_failure_are_reported(self):
        import psutil
        from benchmarks.run import measure
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / 'metrics.json'
            process = Mock(pid=123, returncode=0)
            process.poll.side_effect = [None, 0, 0]
            with patch('benchmarks.run.subprocess.Popen', return_value=process), patch('psutil.Process', side_effect=psutil.NoSuchProcess(123)), patch('benchmarks.run.time.sleep'):
                metrics = measure(['fixture'], result, 5)
            self.assertEqual(metrics['exit_code'], 0)
            self.assertIsNone(metrics['peak_process_tree_rss_bytes'])
            with patch('benchmarks.run.subprocess.Popen', side_effect=FileNotFoundError()):
                metrics = measure(['missing'], result, 5)
            self.assertEqual(metrics['launch_error'], 'FileNotFoundError')

    def test_interruption_cleans_up_the_workload(self):
        from benchmarks.run import measure
        with tempfile.TemporaryDirectory() as directory:
            process = Mock(pid=123)
            process.poll.return_value = None
            monitor = Mock()
            monitor.children.return_value = []
            monitor.memory_info.return_value.rss = 100
            with patch('benchmarks.run.subprocess.Popen', return_value=process), patch('psutil.Process', return_value=monitor), patch('benchmarks.run.time.sleep', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    measure(['fixture'], Path(directory) / 'metrics.json', 5)
            process.kill.assert_called_once()
            process.wait.assert_called_once()


class BenchmarkConfigTests(unittest.TestCase):
    def test_rejects_ambiguous_configurations_and_preserves_literal_braces(self):
        from benchmarks.run import validate_config, expand_command
        config = {'kind': 'fixture', 'frameworks': [{'id': 'fixture', 'command': ['{python}'], 'revision': '1'}],
                  'seeds': [1, 2], 'budget': 3, 'timeout_seconds': 5}
        validate_config(config)
        for update in [{'seeds': [1, 1]}, {'budget': 0}, {'timeout_seconds': float('nan')},
                       {'frameworks': [{**config['frameworks'][0], 'id': '../outside'}]},
                       {'frameworks': config['frameworks'] * 2}]:
            with self.assertRaises(ValueError):
                validate_config({**config, **update})
        self.assertEqual(expand_command(['{python}', '{"literal": 1}', '{seed}'], {'python': 'python', 'seed': 2}),
                         ['python', '{"literal": 1}', '2'])


if __name__ == '__main__':
    unittest.main()
