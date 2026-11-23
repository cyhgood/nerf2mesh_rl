import copy
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from meshutils import subdivide_longest_edges
from nerf.refinement_multiround import edge_costs, lineage_returns, local_rewards, _region_stats, _silhouette_owners, joint_policy_loss
from nerf import refinement_multiround
from nerf.refinement_rl import FacePolicy


class MultiRoundTest(unittest.TestCase):
    def test_uncovered_target_pixels_assigned_to_nearby_face(self):
        faces = np.array([0, -1, -1, 1, -1])
        target = np.array([1., 1., 1., 1., 0.])
        owners = _silhouette_owners(faces, target, (1, 5), 1.0)
        np.testing.assert_array_equal(owners, [0, 0, 1, 1, -1])
        before = [(faces, np.zeros(5), np.array([0., 1., 1., 0., 0.]), target, (1, 5))]
        after = [(faces, np.zeros(5), np.array([0., 0., 1., 0., 0.]), target, (1, 5))]
        counts, first, last = _region_stats(before, after, 2, 2, 1.0)
        np.testing.assert_array_equal(counts, [2, 2])
        np.testing.assert_allclose(first, [0.5, 0.5])
        np.testing.assert_allclose(last, [0.0, 0.5])
        np.testing.assert_array_equal(_silhouette_owners(np.full(5, -1), target, (1, 5), 2), np.full(5, -1))

    def test_joint_score_includes_hold_and_failed_requests(self):
        class Logits(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.values = torch.nn.Parameter(torch.zeros(3))
            def forward(self, x):
                return self.values
        policy = Logits()
        transition = {'features': np.zeros((3, 3), dtype=np.float32),
                      'actions': np.array([False, True, True]),
                      'event': {'request_status': {1: 'conflict', 2: 'budget'}}}
        objective, losses = joint_policy_loss(policy, [transition], [np.array([1.0])])
        self.assertEqual(objective, 1.0)
        losses[0].backward()
        torch.testing.assert_close(policy.values.grad, torch.tensor([0.5, -0.5, -0.5]))

    def test_local_reward_and_invalid_observation(self):
        counts = np.array([2, 0])
        reward = local_rewards(np.array([0.8, 0.]), np.array([0.5, 0.]),
                               np.array([0.4, 0.]), np.array([0.1, 0.]),
                               counts, np.array([2, 0]), [2., 3., 0.1])
        self.assertAlmostEqual(reward[0], 1.3)
        self.assertTrue(np.isnan(reward[1]))

    def test_shared_edge_cost_and_budget(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        _, result, parent, event = subdivide_longest_edges(vertices, faces, [True, True], True, [0, 1], 2)
        self.assertEqual(len(result) - len(faces), 2)
        np.testing.assert_array_equal(parent, [0, 0, 1, 1])
        np.testing.assert_array_equal(edge_costs(event, 2), [2, 0])
        self.assertEqual(event['request_status'][1], 'budget')
        _, result, _, event = subdivide_longest_edges(vertices, faces, [True, False], True, [0], 1)
        self.assertEqual(len(result), 2)
        np.testing.assert_array_equal(edge_costs(event, 2), [0, 0])

    def test_three_round_area_weighted_lineage(self):
        transitions = [
            {'reward': np.array([1., 2.]), 'parent_face_id': np.array([0, 0, 1]), 'child_areas': np.array([1., 3., 2.])},
            {'reward': np.array([3., 4., 5.]), 'parent_face_id': np.array([0, 1, 1, 2]), 'child_areas': np.array([1., 1., 1., 2.])},
            {'reward': np.array([2., 4., 6., 8.])},
        ]
        result = lineage_returns(transitions, 0.5)
        np.testing.assert_allclose(result[2], [2., 4., 6., 8.])
        np.testing.assert_allclose(result[1], [4., 6.5, 9.])
        np.testing.assert_allclose(result[0], [1 + 0.5 * (4. + 3 * 6.5) / 4, 6.5])
        transitions[2]['reward'][1] = np.nan
        result = lineage_returns(transitions, 0.5)
        self.assertTrue(np.isnan(result[1][1]))
        self.assertTrue(np.isnan(result[0][0]))
        self.assertTrue(np.isfinite(result[0][1]))

    def test_interrupted_resume_matches_uninterrupted(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
                self.triangles = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)
                self.vertices_offsets = torch.nn.Parameter(torch.zeros_like(self.vertices))
                self.color = torch.nn.Parameter(torch.tensor(0.4))
                self.v_cumsum = np.array([0, 4])
                self.f_cumsum = np.array([0, 2])

            def refine_and_decimate(self, export_mesh, candidate_order, max_added_faces):
                mask = np.isin(np.arange(len(self.triangles)), candidate_order)
                verts, faces, parent, event = subdivide_longest_edges(
                    (self.vertices + self.vertices_offsets).detach().numpy(), self.triangles.numpy(),
                    mask, True, candidate_order, max_added_faces)
                self.vertices = torch.tensor(verts)
                self.triangles = torch.tensor(faces)
                self.vertices_offsets = torch.nn.Parameter(torch.zeros_like(self.vertices))
                self.v_cumsum = np.array([0, len(verts)])
                self.f_cumsum = np.array([0, len(faces)])
                self.parent_face_id = torch.tensor(parent)
                self.refinement_event = event

        class Dataset:
            poses = [0, 1]

            def collate(self, index):
                return {'index': index[0], 'images': torch.ones(1, 3, 4), 'H': 1, 'W': 3}

        def make_trainer(output, ckpt, mesh, resume=''):
            opt = SimpleNamespace(refine_rl=True, refine_compare_random=False, refine_rl_eval=False,
                                  refine_face_budget=2, refine_multiround_rounds=3, refine_rl_lr=0.01,
                                  refine_multiround_gamma=0.9, refine_multiround_sil_weight=1.,
                                  refine_multiround_rgb_weight=1., refine_multiround_face_cost=0.001,
                                  refine_rl_reward_views=[0], ckpt=ckpt, mesh=mesh, seed=42,
                                  refine_experiment_steps=1, refine_experiment_output=output,
                                  refine_multiround_resume=resume, refine=True)
            model = Model()
            trainer = SimpleNamespace(opt=opt, model=model, device=torch.device('cpu'), global_step=0,
                                      criterion=torch.nn.SmoothL1Loss(reduction='none'),
                                      scaler=torch.cuda.amp.GradScaler(enabled=False))
            trainer.optimizer_fn = lambda m: torch.optim.Adam(m.parameters(), lr=0.01)
            trainer.lr_scheduler_fn = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
            trainer.optimizer = trainer.optimizer_fn(model)
            trainer.lr_scheduler = trainer.lr_scheduler_fn(trainer.optimizer)
            trainer.post_train_step = lambda: None
            trainer.log = lambda message: None

            def evaluate(data):
                model.triangles_errors_id = torch.tensor([[0, min(1, len(model.triangles) - 1), -1]])
                model.refinement_alpha = torch.tensor([[0.6, 0.8, 0.]])
                return torch.full((1, 3, 3), float(model.color.detach())), None, torch.ones(1, 3, 3), None

            trainer.eval_step = evaluate
            trainer.train_step = lambda data: (None, None, (model.color - 0.8).square() + model.vertices_offsets.square().sum())
            return trainer

        with tempfile.TemporaryDirectory() as directory:
            ckpt, mesh = os.path.join(directory, 'original.pt'), os.path.join(directory, 'mesh.ply')
            for path in (ckpt, mesh):
                with open(path, 'w') as output:
                    output.write(path)
            full_dir, interrupted_dir, resumed_dir = [os.path.join(directory, name) for name in ('full', 'interrupted', 'resumed')]
            for path in (full_dir, interrupted_dir, resumed_dir):
                os.mkdir(path)
            full = make_trainer(full_dir, ckpt, mesh)
            refinement_multiround.run_multiround(full, Dataset(), Dataset(), [0], [1])

            interrupted = make_trainer(interrupted_dir, ckpt, mesh)
            real_save = refinement_multiround._save

            def save_then_interrupt(path, state):
                real_save(path, state)
                raise RuntimeError('simulated interruption')

            with mock.patch.object(refinement_multiround, '_save', side_effect=save_then_interrupt):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    refinement_multiround.run_multiround(interrupted, Dataset(), Dataset(), [0], [1])
            resumed = make_trainer(resumed_dir, ckpt, mesh, os.path.join(interrupted_dir, 'trajectory.pt'))
            refinement_multiround.run_multiround(resumed, Dataset(), Dataset(), [0], [1])
            full_state = torch.load(os.path.join(full_dir, 'trajectory.pt'), map_location='cpu')
            resumed_state = torch.load(os.path.join(resumed_dir, 'trajectory.pt'), map_location='cpu')
            torch.manual_seed(42)
            initial = FacePolicy().state_dict()
            self.assertTrue(any(not torch.equal(initial[key], full_state['policy'][key]) for key in initial))
            self.assertEqual(full_state['global_step'], 3)
            self.assertEqual(resumed_state['global_step'], 3)
            torch.testing.assert_close(full_state['vertices'], resumed_state['vertices'])
            torch.testing.assert_close(full_state['triangles'], resumed_state['triangles'])
            for key in full_state['policy']:
                torch.testing.assert_close(full_state['policy'][key], resumed_state['policy'][key])
            self.assertEqual(len(resumed_state['transitions']), 3)
            continued_dir = os.path.join(directory, 'continued')
            os.mkdir(continued_dir)
            continued = make_trainer(continued_dir, ckpt, mesh, os.path.join(full_dir, 'trajectory.pt'))
            continued.opt.refine_multiround_rounds = 1
            refinement_multiround.run_multiround(continued, Dataset(), Dataset(), [0], [1])
            continued_state = torch.load(os.path.join(continued_dir, 'trajectory.pt'), map_location='cpu')
            self.assertEqual(continued_state['global_step'], 4)
            self.assertEqual(len(continued_state['history']), 3)
            self.assertEqual(continued_state['transitions'][0]['round'], 4)
            legacy = copy.deepcopy(full_state)
            legacy['config'].pop('horizon')
            legacy['config'].pop('silhouette_radius')
            legacy['config']['pixel_regions'] = 'pre-update face index for each reward-view pixel'
            legacy_path = os.path.join(directory, 'legacy.pt')
            torch.save(legacy, legacy_path)
            legacy_dir = os.path.join(directory, 'legacy_continued')
            os.mkdir(legacy_dir)
            legacy_trainer = make_trainer(legacy_dir, ckpt, mesh, legacy_path)
            legacy_trainer.opt.refine_multiround_rounds = 1
            refinement_multiround.run_multiround(legacy_trainer, Dataset(), Dataset(), [0], [1])
            restored = torch.load(os.path.join(legacy_dir, 'trajectory.pt'), map_location='cpu')
            self.assertEqual(len(restored['history']), 3)
            self.assertEqual(restored['transitions'][0]['round'], 4)
            # A six-round segment resumed after round three matches an uninterrupted segment.
            paths = [os.path.join(directory, name) for name in ('full_six', 'first_three', 'last_three')]
            for path in paths:
                os.mkdir(path)
            full_six = make_trainer(paths[0], ckpt, mesh)
            full_six.opt.refine_multiround_horizon = 6
            full_six.opt.refine_multiround_rounds = 6
            refinement_multiround.run_multiround(full_six, Dataset(), Dataset(), [0], [1])
            first_three = make_trainer(paths[1], ckpt, mesh)
            first_three.opt.refine_multiround_horizon = 6
            refinement_multiround.run_multiround(first_three, Dataset(), Dataset(), [0], [1])
            pending = torch.load(os.path.join(paths[1], 'trajectory.pt'), map_location='cpu')
            self.assertFalse(pending['completed'])
            self.assertEqual(len(pending['transitions']), 3)
            last_three = make_trainer(paths[2], ckpt, mesh, os.path.join(paths[1], 'trajectory.pt'))
            last_three.opt.refine_multiround_horizon = 6
            refinement_multiround.run_multiround(last_three, Dataset(), Dataset(), [0], [1])
            expected = torch.load(os.path.join(paths[0], 'trajectory.pt'), map_location='cpu')
            actual = torch.load(os.path.join(paths[2], 'trajectory.pt'), map_location='cpu')
            self.assertTrue(actual['completed'])
            self.assertEqual(len(actual['transitions']), 6)
            torch.testing.assert_close(expected['vertices'], actual['vertices'])
            for key in expected['policy']:
                torch.testing.assert_close(expected['policy'][key], actual['policy'][key])
            for expected_round, actual_round in zip(expected['transitions'], actual['transitions']):
                np.testing.assert_allclose(expected_round['lineage_return'], actual_round['lineage_return'], equal_nan=True)


if __name__ == '__main__':
    unittest.main()
