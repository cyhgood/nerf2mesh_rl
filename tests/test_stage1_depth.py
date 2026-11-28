import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from nerf.stage1_depth import Stage1DepthSource, stage1_depth_loss
from test_stage1_silhouette import load_methods


class Stage1DepthTest(unittest.TestCase):
    def manifest(self, directory, **overrides):
        data = {'split': 'train', 'definition': 'camera_z', 'source_unit': 'cm',
                'transformed_unit': 'scene_units', 'scale': .01, 'offset': 1.,
                'invalid_value': -1, 'frames': {'0': 'depth.npy'}}
        data.update(overrides)
        path = os.path.join(directory, 'depth.json')
        with open(path, 'w') as f:
            json.dump(data, f)
        return path

    def test_manifest_values_and_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            np.save(os.path.join(d, 'depth.npy'), np.array([[100., -1.], [300., np.nan]], np.float32))
            source = Stage1DepthSource(self.manifest(d))
            target, valid = source.load(0, (2, 2), 'cpu')
            torch.testing.assert_close(target, torch.tensor([[2., 0.], [4., 0.]]))
            self.assertEqual(valid.tolist(), [[True, False], [True, False]])
            self.assertEqual(source.describe()['definition'], 'camera_z')
            self.assertEqual(len(source.describe()['manifest_sha256']), 64)
            with self.assertRaisesRegex(ValueError, 'no .npy'):
                source.load(1, (2, 2), 'cpu')
            with self.assertRaisesRegex(ValueError, 'shape'):
                source.load(0, (4, 1), 'cpu')
            for overrides in ({'split': 'test'}, {'definition': 'unknown'}, {'scale': 0},
                              {'source_unit': ''}, {'transformed_unit': 'mystery'}):
                with self.assertRaises(ValueError):
                    Stage1DepthSource(self.manifest(d, **overrides))
            source = Stage1DepthSource(self.manifest(d, frames={'0': 'unknown.png'}))
            with self.assertRaisesRegex(ValueError, 'no .npy'):
                source.load(0, (2, 2), 'cpu')

    def test_masked_loss_and_empty(self):
        pred = torch.tensor([1., 100., 2., 5.], requires_grad=True)
        target = torch.tensor([2., 2., 4., 6.])
        loss, n = stage1_depth_loss(pred, torch.tensor([1, 0, 1, 1]), target, torch.tensor([1, 1, 1, 0]))
        self.assertEqual(n, 2)
        self.assertAlmostEqual(loss.item(), 1.5)
        loss.backward()
        torch.testing.assert_close(pred.grad, torch.tensor([-.5, 0., -.5, 0.]))
        empty, n = stage1_depth_loss(pred, torch.zeros(4), target, torch.ones(4))
        self.assertEqual(n, 0)
        self.assertEqual(empty.item(), 0.)
        empty.backward()

    def test_stage1_trainer_and_branch_interface(self):
        from nerf.silhouette_loss import stage1_silhouette_loss
        train_step, = load_methods('nerf/utils.py', 'Trainer', ['train_step'],
                                   {'torch': torch, 'stage1_silhouette_loss': stage1_silhouette_loss,
                                    'Stage1DepthSource': Stage1DepthSource, 'stage1_depth_loss': stage1_depth_loss})
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vertices_offsets = torch.nn.Parameter(torch.tensor([1.]))
            def render_stage1(self, *args, **kwargs):
                result = {'image': torch.zeros(4, 3)}
                if kwargs.get('depth_definition'):
                    result.update(surface_depth=self.vertices_offsets.expand(4), surface_depth_hit=torch.ones(4, dtype=torch.bool))
                return result
        with tempfile.TemporaryDirectory() as d:
            np.save(os.path.join(d, 'depth.npy'), np.full((2, 2), 100., np.float32))
            opt = SimpleNamespace(stage=1, color_space='srgb', background='white', refine=False,
                                  lambda_lpips=0, lambda_lap=0, lambda_normal=0, lambda_edgelen=0,
                                  lambda_offsets=0, lambda_sil=0, lambda_tv=0, lambda_stage1_depth=0,
                                  stage1_depth_manifest=self.manifest(d), data_format='nerf', train_split='train')
            model = Model()
            trainer = SimpleNamespace(opt=opt, model=model, device='cpu', global_step=0,
                                      criterion=torch.nn.MSELoss(reduction='none'), log=lambda message: None)
            data = {'rays_o': torch.zeros(4, 3), 'rays_d': torch.ones(4, 3), 'index': [0],
                    'mvp': torch.eye(4).unsqueeze(0), 'H': 2, 'W': 2, 'images': torch.zeros(4, 3),
                    'camera_pose': torch.eye(4).unsqueeze(0)}
            self.assertEqual(train_step(trainer, data)[2].item(), 0.)
            self.assertFalse(hasattr(trainer, 'stage1_depth_source'))
            opt.lambda_stage1_depth = .5
            loss = train_step(trainer, data)[2]
            self.assertAlmostEqual(loss.item(), .5)
            self.assertEqual(trainer.stage1_depth_valid_count, 4)
            loss.backward()
            self.assertAlmostEqual(model.vertices_offsets.grad.item(), -.5)
            opt.train_split = 'test'
            with self.assertRaisesRegex(ValueError, 'only nerf train'):
                train_step(trainer, data)


if __name__ == '__main__':
    unittest.main()
