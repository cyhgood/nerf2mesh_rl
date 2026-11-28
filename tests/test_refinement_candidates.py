import unittest
from types import SimpleNamespace

import numpy as np
import torch

from meshutils import subdivide_longest_edges
from nerf.refinement_candidates import (candidate_features, normal_angles, remap_vertex_ema,
                                        update_vertex_ema, visibility_counts)
from nerf.refinement_experiment import _train_branch


class CandidateTest(unittest.TestCase):
    def test_area_fraction_visibility_and_stable_ties(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.], [2., 0., 0.]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [0, 1, 4]])
        observations = [(np.array([0, 1, 1, 2]), np.array([1., 2., 4., 5.])),
                        (np.array([0, -1]), np.array([2., 0.]))]
        sums, pixels, views = visibility_counts(observations, 3)
        ids, features, info = candidate_features(vertices, faces, sums, pixels, views, 2,
                                                np.zeros(5), np.zeros(5, dtype=bool), 1., .5)
        np.testing.assert_array_equal(ids, [0])
        self.assertEqual(info['valid_faces'], 2)
        self.assertEqual(info['candidate_faces'], 1)
        self.assertAlmostEqual(info['candidate_rgb_error_coverage'], 3. / 9.)
        self.assertEqual(features.shape, (1, 9))
        self.assertEqual(features[0, 4], 1.)
        self.assertEqual(features[0, 7], 1.)
        empty, features, info = candidate_features(vertices, faces, sums, np.zeros(3), views, 2,
                                                  np.zeros(5), np.zeros(5, dtype=bool), 1.)
        self.assertEqual(len(empty), 0)
        self.assertEqual(features.shape, (0, 9))
        self.assertIsNone(info['candidate_rgb_error_coverage'])

    def test_angles_boundaries_and_nonmanifold(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., -1., 0.]])
        faces = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]])
        _, angles, valid = normal_angles(vertices, faces[:2], 1e-12)
        np.testing.assert_allclose(angles, [np.pi / 2, np.pi / 2])
        self.assertTrue(valid.all())
        _, angles, valid = normal_angles(vertices, faces, 1e-12)
        self.assertFalse(valid.any())
        np.testing.assert_array_equal(angles, [0., 0., 0.])

    def test_midpoint_mapping_and_skips(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        new_vertices, _, _, event = subdivide_longest_edges(vertices, faces, [True, False], True)
        ema, seen, mapping = remap_vertex_ema(np.array([2., 4., 6., 8.]), np.ones(4, dtype=bool),
                                               event, len(new_vertices))
        np.testing.assert_array_equal(mapping[-1], [0, 2])
        self.assertEqual(ema[-1], 4.)
        self.assertTrue(seen[-1])
        state = {'ema': ema.copy(), 'seen': seen.copy()}
        update_vertex_ema(state, np.ones(5) * np.nan, .5, True)
        update_vertex_ema(state, np.ones(5), .5, False)
        np.testing.assert_array_equal(state['ema'], ema)

    def test_unscale_before_ema_and_skip_overflow(self):
        class Scaler:
            def __init__(self, skip=False):
                self.factor = 8.
                self.skip = skip
            def get_scale(self):
                return self.factor
            def is_enabled(self):
                return True
            def scale(self, loss):
                return loss * self.factor
            def unscale_(self, optimizer):
                for group in optimizer.param_groups:
                    for param in group['params']:
                        if param.grad is not None:
                            param.grad.div_(self.factor)
            def step(self, optimizer):
                if not self.skip:
                    optimizer.step()
            def update(self):
                if self.skip:
                    self.factor /= 2

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vertices_offsets = torch.nn.Parameter(torch.zeros(1, 3))

        for skip in (False, True):
            model = Model()
            trainer = SimpleNamespace(model=model, scaler=Scaler(skip), global_step=0)
            trainer.optimizer_fn = lambda m: torch.optim.SGD(m.parameters(), lr=.1)
            trainer.lr_scheduler_fn = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
            trainer.post_train_step = lambda: None
            trainer.train_step = lambda data: (None, None, (model.vertices_offsets * torch.tensor([[3., 4., 0.]])).sum())
            state = {'ema': np.zeros(1), 'seen': np.zeros(1, dtype=bool)}
            _train_branch(trainer, SimpleNamespace(collate=lambda index: None), [0], 0, state, 0.)
            self.assertEqual(state['ema'][0], 0. if skip else 5.)
            self.assertEqual(bool(state['seen'][0]), not skip)


if __name__ == '__main__':
    unittest.main()
