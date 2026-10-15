"""真实 CLI 在单机 Torch/sklearn 训练结束后导出可校验成果包。"""
import json
from zipfile import ZipFile

import pytest
import torch

from dl_helper.training.artifacts import write_yaml
from dl_helper.training.bundles import default_bundle_path
from dl_helper.training.cli import main
from dl_helper.training.config import default_schema


@pytest.mark.parametrize('backend', ['torch', 'sklearn'])
def test_train_without_alist_generates_local_bundle(tmp_path, backend, monkeypatch):
    from dataclasses import replace
    from dl_helper.training.platform import Platform

    resolve = Platform.resolve_torch_resources
    monkeypatch.setattr(Platform, 'resolve_torch_resources',
                        lambda self, config, batch: replace(resolve(self, config, batch),
                                                            num_workers=0, persistent_workers=False,
                                                            prefetch_factor=None))
    schema = default_schema()
    schema['run']['id'] = f'local-{backend}'
    schema['run']['output_root'] = str(tmp_path)
    schema['training']['max_epochs'] = 1
    schema['selection'] = {'metric': 'val/accuracy', 'mode': 'max', 'patience': 5, 'min_delta': 0.0}
    schema['distributed'] = {'num_processes': 1}
    schema['remote'] = {'type': 'none'}
    schema['notifications'] = {'type': 'none'}
    if backend == 'sklearn':
        schema['backend'] = {
            'type': 'sklearn', 'torch': None,
            'sklearn': {'fit_mode': 'batch', 'evaluation_batch_size': 4096,
                        'n_jobs': None, 'random_state': 'run_seed', 'sample_weight_parameter': None},
        }
        experiment = 'experiments.sklearn_batch:build_experiment'
    else:
        schema['backend']['torch']['mixed_precision'] = 'no'
        experiment = 'experiments.toy_multiclass:build_experiment'
    config = tmp_path / 'config.yaml'
    write_yaml(str(config), schema)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        assert main(['train', '--config', str(config), '--experiment', experiment,
                     '--no-use-alist']) == 0
    finally:
        torch.set_num_threads(threads)
    run_dir = tmp_path / 'runs' / schema['run']['id']
    output = default_bundle_path(run_dir)
    with ZipFile(output) as archive:
        manifest = json.loads(archive.read('run-manifest.json'))
        inventory = json.loads(archive.read('bundle-manifest.json'))
        assert manifest['status'] == 'succeeded'
        assert manifest['run_id'] == schema['run']['id']
        assert 'services/service-manifest.json' in inventory['files']
        assert 'report/index.html' in inventory['files']
        assert not any(name.startswith(('checkpoints/', 'checkpoint-archives/'))
                       for name in archive.namelist())
