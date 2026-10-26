"""Model admission/dispatch tests: fake backends, no installs or weight downloads."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import adaptive
import models
import test_bank


class ModelTests(unittest.TestCase):
    def test_complete_cohort_excludes_only_jev_api_and_small_models_first(self):
        names = models.select_models(['all'])
        self.assertEqual(len(names), 22)
        self.assertIn('nanojev', names)
        self.assertNotIn('jev-1.13.0', names)
        sizes = [models.PARAMETERS_B.get(n, .5) for n in names]
        self.assertEqual(sizes, sorted(sizes))
        with self.assertRaises(ValueError):
            models.select_models(['not-a-model'])

    def test_t4_memory_precision_and_explicit_large_native_block(self):
        self.assertEqual(models.memory_policy('kev_08b', 14.), 'float16')
        self.assertEqual(models.memory_policy('qwen3_17b', 14.), 'float16')
        self.assertEqual(models.memory_policy('qwen3_8b', 14.), 'nf4')
        self.assertEqual(models.memory_policy('nanojev', 14.), 'float32')
        self.assertEqual(models.memory_policy('kev_27b', 80.), 'float16')
        for model in ('kev_27b', 'startlux_27b', 'qwen38_27b'):
            with self.assertRaises(models.ModelBlocked) as cm:
                models.memory_policy(model, 14.)
            self.assertEqual(cm.exception.status, 'BLOCKED_RESOURCE')
        with self.assertRaises(models.ModelBlocked):
            models.memory_policy('kev_4b', 14., 'nf4')
        with self.assertRaises(models.ModelBlocked):
            models.memory_policy('intern_4b', 14., 'bfloat16', False)

    def test_dispatch_each_supported_non_laya_model_without_loading(self):
        cuda = types.SimpleNamespace(mem_get_info=lambda: (80*2**30, 80*2**30),
                                     is_bf16_supported=lambda: True)
        fake_torch = types.SimpleNamespace(cuda=cuda)
        for name in models.model_names():
            if name in models.LAYA:
                continue
            with patch.dict(sys.modules, {'torch': fake_torch}), \
                 patch.object(models, 'prepare_checkpoint', return_value={'path':'fixture'}) as prepare, \
                 patch.object(models, 'NativeBridge', return_value='native') as native, \
                 patch.object(models, 'CodedLM', return_value='lm') as lm:
                adapter = models.create_adapter(name, {name:{'path':'fixture'}})
                self.assertEqual(adapter, 'native' if name in models.NATIVE else 'lm')
                prepare.assert_called_once()
                self.assertEqual(native.call_count + lm.call_count, 1)

    def test_jev_skip_never_imports_torch_or_calls_download(self):
        with patch.object(models, 'prepare_checkpoint') as prepare:
            with self.assertRaises(models.ModelBlocked) as cm:
                models.create_adapter('jev-1.13.0')
            self.assertEqual(cm.exception.status, 'SKIPPED_API')
            prepare.assert_not_called()

    def test_resource_block_happens_before_download(self):
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(
            mem_get_info=lambda: (14*2**30,15*2**30), is_bf16_supported=lambda: False))
        with patch.dict(sys.modules, {'torch': fake_torch}), patch.object(models, 'prepare_checkpoint') as prepare:
            with self.assertRaises(models.ModelBlocked):
                models.create_adapter('kev_27b')
            prepare.assert_not_called()

    def test_blocked_model_does_not_stop_next_model(self):
        runner, build = test_bank.runner, test_bank.build
        with tempfile.TemporaryDirectory(dir=build.ROOT) as tmp:
            root=Path(tmp);bank=root/'bank';bank.mkdir()
            build.dump(bank/'manifest.json', {'fixture':True})
            build.dump(bank/'audit.json', {'fixture':True})
            item=test_bank.fixture_items()[1]
            rows,_=build.single_records([item],'native',{item['base_id']:'confirmation'})

            def factory(name):
                if name=='kev_27b':
                    raise models.ModelBlocked('BLOCKED_RESOURCE','fixture: larger GPU required')
                return test_bank.FakeAdapter(name)

            def execute(args, log):
                name=args[args.index('--model')+1]
                runner.run_model(bank,Path(args[args.index('--output')+1]),name,adapter_factory=factory)
                return 0

            with patch.object(runner,'load_bank',return_value=rows), contextlib.redirect_stdout(io.StringIO()):
                result=runner.run_all(bank,root/'results',['kev_27b','english'],execute=execute)
            self.assertEqual(result['models']['kev_27b']['status'],'BLOCKED_RESOURCE')
            self.assertEqual(result['models']['english']['status'],'OK')

    def test_native_boolean_and_choice_envelope(self):
        q=dict(type='choice',criteria={'second':'x','first':'y'})
        a=models.wrap_distribution(q, {'second':.8,'first':.2})
        self.assertEqual(a['choice'],'second')
        a=models.wrap_distribution({'type':'noul'}, {'false':.1,'true':.9})
        self.assertEqual(a['noul'], .9)

    def test_monitor_default_64_samples_at_four_hertz_and_long_batch_activity(self):
        from test_adaptive import Meter
        self.assertEqual(adaptive.DEFAULTS['low_samples'], 64)
        self.assertEqual(adaptive.DEFAULTS['sample_interval'], .25)
        c=adaptive.Controller(dict(adaptive.DEFAULTS))
        monitor=adaptive.Monitor(Meter(),c,adaptive.DEFAULTS,lambda:10,lambda *a,**k:None)
        monitor.begin_forward()
        for _ in range(63):
            sample=monitor.poll()
            self.assertTrue(sample['forward_inflight'])
            self.assertEqual(sample['completed_batches'],0)
        self.assertEqual(c.limit(),8)
        monitor.poll()
        self.assertEqual(c.limit(),12)
        monitor.end_forward()
        monitor.poll()
        self.assertEqual(c.low_count,0)

    def test_model_overrides_reject_unknown_parameters(self):
        with tempfile.TemporaryDirectory(dir=test_bank.build.ROOT) as tmp:
            path=Path(tmp)/'overrides.json'
            path.write_text(json.dumps({'kev_08b':{'repo':'unrelated/model'}}))
            with self.assertRaises(ValueError):
                models.load_overrides(path)

    def test_vendor_fetch_can_resume_an_initialized_repo_without_head(self):
        with tempfile.TemporaryDirectory(dir=test_bank.build.ROOT) as tmp:
            root=Path(tmp)
            target=root/'.aris/vendor/kev_source'
            (target/'.git').mkdir(parents=True)
            revision=models.VENDORS['kev'][1]
            with patch.object(models,'ROOT',root), patch.object(sys,'path',list(sys.path)), \
                 patch.object(models.subprocess,'run',return_value=types.SimpleNamespace(returncode=1)) as run, \
                 patch.object(models.subprocess,'check_output',side_effect=[revision+'\n','']):
                self.assertEqual(models.ensure_vendor('kev'),target)
            calls=[args.args[0] for args in run.call_args_list]
            self.assertTrue(any('fetch' in args and revision in args for args in calls))
            self.assertTrue(any('checkout' in args and revision in args for args in calls))


if __name__ == '__main__':
    unittest.main()
