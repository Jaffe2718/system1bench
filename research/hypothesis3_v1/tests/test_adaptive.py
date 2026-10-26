"""Scheduler tests use injected engines/meters, with no model download or GPU."""
import collections
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import adaptive
import test_bank


class Meter:
    def sample(self):
        return dict(ram_ratio=.2, gpu_free=12 * 2**30, gpu_total=16 * 2**30,
                    gpu_used=4 * 2**30, gpu_allocated=2 * 2**30, gpu_reserved=3 * 2**30)


class Engine:
    def __init__(self, oom_above=100, bad=None, prepare_bad=None):
        self.oom_above, self.bad, self.prepare_bad = oom_above, bad, prepare_bad
        self.calls, self.recoveries = [], 0

    def prepare(self, row, worker):
        if row['id'] == self.prepare_bad:
            raise ValueError('invalid input fixture')
        return dict(row=row, rows=len(row.get('questions', {'q': None})), length=16,
                    audit={q: {'complete': True} for q in row.get('questions', {'q': None})},
                    worker=worker, prepare_seconds=0.)

    def forward(self, batch):
        self.calls.append([p['row']['id'] for p in batch])
        if len(batch) > self.oom_above:
            raise RuntimeError('CUDA out of memory fixture')
        if any(p['row']['id'] == self.bad for p in batch):
            raise ValueError('bad model input fixture')
        outputs = []
        for p in batch:
            row = p['row']
            if 'questions' in row:
                output = test_bank.FakeAdapter('fixture').predict(
                    [row['state']], row['questions'], 'en', {}, 1)[0]
            else:
                output = {'case': row['id']}
            outputs.append(output)
        return outputs, dict(batch_seconds=.1, gpu_peak_reserved=3 * 2**30)

    def recover_oom(self):
        self.recoveries += 1


def rows(count=30):
    return [dict(id=str(i), panel='paired', k=2, language='en') for i in range(count)]


def collect(inputs, engine, events=None, **options):
    return list(adaptive.collect(inputs, None, {}, options,
                lambda kind, **data: events.append((kind, data)) if events is not None else None,
                engine=engine, meter=Meter()))


class AdaptiveTests(unittest.TestCase):
    def test_sixty_four_consecutive_active_samples_and_reset(self):
        c = adaptive.Controller(dict(adaptive.DEFAULTS))
        low = dict(ram_ratio=.2, gpu_ratio=.3, gpu_free=10 * 2**30,
                   completed_batches=1, backlog=10)
        for _ in range(63):
            c.observe(low)
        self.assertEqual(c.limit(), 8)
        c.observe(dict(low, gpu_ratio=.71))
        for _ in range(63):
            c.observe(low)
        self.assertEqual(c.limit(), 8)
        c.observe(low)
        self.assertEqual(c.limit(), 12)
        c.observe(dict(low, ram_ratio=.86))
        self.assertEqual(c.limit(), 6)
        for _ in range(127):
            c.observe(low)
        self.assertEqual(c.limit(), 6)
        c.observe(low)
        self.assertEqual(c.limit(), 9)

    def test_no_growth_for_idle_samples_and_capacity_cap(self):
        c = adaptive.Controller(dict(adaptive.DEFAULTS, max_batch_rows=9))
        low = dict(ram_ratio=.2, gpu_ratio=.3, gpu_free=10 * 2**30,
                   completed_batches=0, backlog=10)
        for _ in range(100):
            c.observe(low)
        self.assertEqual(c.limit(), 8)
        for _ in range(128):
            c.observe(dict(low, completed_batches=1))
        self.assertEqual(c.limit(), 9)

    def test_monitor_keeps_between_sample_peak(self):
        c = adaptive.Controller(dict(adaptive.DEFAULTS))
        m = adaptive.Monitor(Meter(), c, adaptive.DEFAULTS, lambda: 20, lambda *a, **k: None)
        m.completed(14 * 2**30, 4)
        sample = m.poll()
        self.assertEqual(sample['completed_requests'], 4)
        self.assertGreater(sample['gpu_ratio'], .85)
        self.assertEqual(c.limit(), 4)

    def test_bounded_pipeline_keeps_all_cases_exactly_once(self):
        engine = Engine()
        output = collect(rows(73), engine, prefetch_cases=2, lookahead_cases=16)
        self.assertEqual({r['prepared']['row']['id'] for r in output}, {str(i) for i in range(73)})
        self.assertEqual(len(output), 73)
        self.assertTrue(any(len(b) > 1 for b in engine.calls))
        self.assertTrue(all(r['output']['case'] == r['prepared']['row']['id'] for r in output))

    def test_oom_splits_without_losing_or_failing_requests(self):
        engine, events = Engine(oom_above=2), []
        output = collect(rows(), engine, events)
        self.assertEqual(len(output), 30)
        self.assertTrue(all(r['error'] is None for r in output))
        self.assertGreater(engine.recoveries, 0)
        self.assertTrue(any(k == 'shrink' for k, _ in events))

    def test_one_bad_forward_and_one_bad_preparation_continue(self):
        output = collect(rows(), Engine(bad='3', prepare_bad='8'))
        failed = {r['prepared']['row']['id'] for r in output if r['error'] is not None}
        self.assertEqual(failed, {'3', '8'})
        self.assertEqual(len(output), 30)

    def test_length_and_row_limits_keep_multiselect_atomic(self):
        engine = Engine()
        prepared = [engine.prepare(r, 0) for r in rows(5)]
        prepared[1]['row']['k'] = 3
        prepared[2]['rows'] = 5
        buffer = collections.deque(prepared)
        selected = adaptive.take_batch(buffer, 3, 48)
        self.assertLessEqual(sum(p['rows'] for p in selected), 3)
        self.assertTrue(all(p['row']['k'] == 2 for p in selected))
        self.assertEqual(len(selected) + len(buffer), 5)

    def test_close_stops_producers_even_with_full_queue(self):
        generator = adaptive.collect(rows(500), None, {}, dict(prefetch_cases=1),
                                     lambda *a, **k: None, engine=Engine(), meter=Meter())
        next(generator)
        generator.close()

    def test_invalid_options_rejected(self):
        for options in ({'producer_workers': 0}, {'sample_interval': 0},
                        {'grow_below': .9}, {'initial_batch_rows': 129}):
            with self.assertRaises(ValueError):
                adaptive.validate_options(options)

    def test_adaptive_runner_raw_outputs_resume_and_signature(self):
        runner, build = test_bank.runner, test_bank.build
        fixture = test_bank.fixture_items()[1:]
        cases, _ = build.single_records(fixture, 'native', {b['base_id']: 'confirmation' for b in fixture})
        original = adaptive.collect
        engine = Engine(bad=cases[0]['id'])

        def injected(inputs, adapter, budget, options, event):
            return original(inputs, adapter, budget, options, event, engine=engine, meter=Meter())

        with tempfile.TemporaryDirectory(dir=build.ROOT) as tmp:
            root = Path(tmp)
            bank = root / 'bank'
            bank.mkdir()
            build.dump(bank / 'manifest.json', {'fixture': True})
            with patch.object(runner, 'load_bank', return_value=cases), \
                 patch.object(adaptive, 'collect', side_effect=injected), contextlib.redirect_stdout(io.StringIO()):
                options = dict(scheduler='adaptive')
                result = runner.run_model(bank, root/'output', 'english',
                                          adapter_factory=test_bank.FakeAdapter, options=options)
                count = len(engine.calls)
                runner.run_model(bank, root/'output', 'english',
                                 adapter_factory=test_bank.FakeAdapter, options=options)
                self.assertEqual(len(engine.calls), count)
                with self.assertRaises(ValueError):
                    runner.run_model(bank, root/'output', 'english',
                                     adapter_factory=test_bank.FakeAdapter, options=dict(scheduler='serial'))
            self.assertEqual(result['failed_requests'], 1)
            self.assertEqual(result['completed_requests'], len(cases))
            saved = [json.loads(s) for s in (root/'output/predictions.jsonl').read_text().splitlines()]
            self.assertTrue(all('execution' in r and 'accuracy' not in r for r in saved))
            self.assertFalse(result['signature']['evaluation_performed'])


if __name__ == '__main__':
    unittest.main()
