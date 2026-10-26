"""Cache and dependency decisions with synthetic files; no installs/downloads."""
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import environment
import models
import test_bank


def standard_snapshot(path):
    path.mkdir(parents=True,exist_ok=True)
    for name in ('config.json','tokenizer_config.json','tokenizer.json','model.safetensors'):
        (path/name).write_text('{}',encoding='utf-8')
    return str(path)


class CacheEnvironmentTests(unittest.TestCase):
    def test_explicit_online_setting_can_undo_previous_notebook_offline_setting(self):
        with patch.object(models,'STORAGE',dict(cache_dir=None,offline=False)), \
             patch.dict(os.environ,{'HF_HUB_OFFLINE':'1'}):
            models.configure_storage()
            self.assertTrue(models.STORAGE['offline'])
            models.configure_storage(offline=False)
            self.assertFalse(models.STORAGE['offline'])
            self.assertNotIn('HF_HUB_OFFLINE',os.environ)

    def test_complete_cache_hit_never_calls_online_download(self):
        with tempfile.TemporaryDirectory(dir=test_bank.build.ROOT) as tmp:
            path=standard_snapshot(Path(tmp)/'snapshot')
            snapshot=Mock(return_value=path)
            with patch.dict(sys.modules,{'huggingface_hub':types.SimpleNamespace(snapshot_download=snapshot)}), \
                 patch.object(models,'STORAGE',dict(cache_dir='/persistent/hub',offline=False)):
                result=models.cached_snapshot('org/model','a'*40,['*.json'],'qwen3_17b')
            self.assertEqual(result,path)
            snapshot.assert_called_once()
            self.assertTrue(snapshot.call_args.kwargs['local_files_only'])
            self.assertEqual(snapshot.call_args.kwargs['cache_dir'],'/persistent/hub')

    def test_partial_cache_online_fills_missing_without_force_redownload(self):
        with tempfile.TemporaryDirectory(dir=test_bank.build.ROOT) as tmp:
            root=Path(tmp);partial=root/'partial';partial.mkdir()
            (partial/'config.json').write_text('{}')
            full=standard_snapshot(root/'full')
            snapshot=Mock(side_effect=[str(partial),full])
            with patch.dict(sys.modules,{'huggingface_hub':types.SimpleNamespace(snapshot_download=snapshot)}), \
                 patch.object(models,'STORAGE',dict(cache_dir=None,offline=False)):
                self.assertEqual(models.cached_snapshot('org/model','a'*40,['*'],'qwen3_17b'),full)
            self.assertEqual([c.kwargs['local_files_only'] for c in snapshot.call_args_list],[True,False])
            self.assertTrue(all('force_download' not in c.kwargs for c in snapshot.call_args_list))

    def test_offline_partial_cache_blocks_without_network(self):
        with tempfile.TemporaryDirectory(dir=test_bank.build.ROOT) as tmp:
            partial=Path(tmp);(partial/'config.json').write_text('{}')
            snapshot=Mock(return_value=str(partial))
            with patch.dict(sys.modules,{'huggingface_hub':types.SimpleNamespace(snapshot_download=snapshot)}), \
                 patch.object(models,'STORAGE',dict(cache_dir=None,offline=True)):
                with self.assertRaises(models.ModelBlocked) as caught:
                    models.cached_snapshot('org/model','a'*40,['*'],'qwen3_17b')
            self.assertEqual(caught.exception.status,'BLOCKED_CACHE')
            snapshot.assert_called_once()
            self.assertTrue(snapshot.call_args.kwargs['local_files_only'])

    def test_missing_weight_shard_is_not_complete(self):
        with tempfile.TemporaryDirectory(dir=test_bank.build.ROOT) as tmp:
            path=Path(standard_snapshot(Path(tmp)))
            (path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'shard-1.safetensors','b':'shard-2.safetensors'}}))
            (path/'shard-1.safetensors').write_text('fixture')
            self.assertFalse(models.snapshot_complete(path,'qwen3_17b'))
            (path/'shard-2.safetensors').write_text('fixture')
            self.assertTrue(models.snapshot_complete(path,'qwen3_17b'))

    def test_explicit_local_model_path_does_not_call_hub(self):
        with tempfile.TemporaryDirectory(dir=test_bank.build.ROOT) as tmp:
            path=standard_snapshot(Path(tmp))
            api=Mock();snapshot=Mock()
            with patch.dict(sys.modules,{'huggingface_hub':types.SimpleNamespace(HfApi=api,snapshot_download=snapshot)}):
                pin=models.prepare_checkpoint('qwen3_17b',{'path':path})
            self.assertEqual(pin['path'],str(Path(path).resolve()))
            api.assert_not_called();snapshot.assert_not_called()

    def test_offline_missing_native_source_does_not_run_git(self):
        with tempfile.TemporaryDirectory(dir=test_bank.build.ROOT) as tmp:
            with patch.object(models,'ROOT',Path(tmp)), \
                 patch.object(models,'STORAGE',dict(cache_dir=None,offline=True)), \
                 patch.object(models.subprocess,'run') as run:
                with self.assertRaises(models.ModelBlocked) as caught:
                    models.ensure_vendor('kev')
            self.assertEqual(caught.exception.status,'BLOCKED_CACHE')
            run.assert_not_called()

    def test_nano_absent_private_api_avoids_failing_constructor_branch(self):
        missing=ModuleNotFoundError("No module named 'torch._native'",name='torch._native')
        predictor=Mock(return_value='fixture-predictor')
        with patch.object(models.importlib,'import_module',side_effect=missing):
            model,disable,reason=models.make_nano_predictor(predictor,Path('fixture'))
        self.assertEqual(model,'fixture-predictor')
        self.assertFalse(disable)
        self.assertIn('absent',reason)
        self.assertIs(predictor.call_args.kwargs['disable_native_triton'],False)
        self.assertEqual(predictor.call_args.kwargs['precision'],'fp32')

    def test_nano_available_registry_uses_disable_hook(self):
        native=types.SimpleNamespace(triton_utils=types.SimpleNamespace(deregister_op_overrides=lambda:None))
        with patch.object(models.importlib,'import_module',return_value=native):
            disable,reason=models.nano_triton_policy()
        self.assertTrue(disable)

    def test_unrelated_import_failure_is_not_hidden(self):
        missing=ModuleNotFoundError("No module named 'other_dependency'",name='other_dependency')
        with patch.object(models.importlib,'import_module',side_effect=missing):
            with self.assertRaises(ModuleNotFoundError):
                models.nano_triton_policy()

    def test_dependency_inspection_reuses_compatible_newer_versions(self):
        with tempfile.TemporaryDirectory(dir=test_bank.build.ROOT) as tmp:
            path=Path(tmp)/'requirements.txt'
            path.write_text('transformers>=5.16.1,<6\nlaya==0.3.20\nnumpy>=1.26\n')
            installed={'transformers':'5.17.0','laya':'0.3.20','numpy':'2.2.0'}
            missing,reused=environment.missing_requirements(path,installed.__getitem__)
            self.assertEqual(missing,[])
            self.assertEqual(len(reused),3)

    def test_dependency_inspection_installs_only_missing_or_incompatible(self):
        with tempfile.TemporaryDirectory(dir=test_bank.build.ROOT) as tmp:
            path=Path(tmp)/'requirements.txt'
            path.write_text('transformers>=5.16.1,<6\nlaya==0.3.20\npsutil>=5.9\n')
            def installed(name):
                if name=='psutil':
                    raise metadata.PackageNotFoundError(name)
                return {'transformers':'5.17.0','laya':'0.3.19'}[name]
            missing,reused=environment.missing_requirements(path,installed)
            self.assertEqual(missing,['laya==0.3.20','psutil>=5.9'])
            self.assertEqual(reused,[{'name':'transformers','version':'5.17.0'}])


if __name__=='__main__':
    unittest.main()
