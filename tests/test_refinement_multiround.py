import copy
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from meshutils import subdivide_longest_edges
from nerf.refinement_multiround import (edge_costs, lineage_returns, local_rewards, _region_stats,
                                        _silhouette_owners, face_policy_losses, reward_distributions, action_counts,
                                        grouped_rewards)
from nerf import refinement_multiround
from nerf.refinement_candidates import CandidatePolicy


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

    def test_face_returns_change_the_corresponding_ranking_gradient(self):
        class Logits(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.values = torch.nn.Parameter(torch.zeros(3))
            def forward(self, x):
                return self.values
        policy = Logits()
        transition = {'features': np.zeros((3, 3), dtype=np.float32), 'visible': np.arange(3),
                      'selection_order': np.array([0, 1, 2]), 'considered_count': 2,
                      'actions': np.array([True, True, False]),
                      'event': {'request_status': {0: 'executed', 1: 'conflict', 2: 'budget_not_considered'},
                                'considered_count': 2, 'executed_edges': [[0, 1]]}}
        losses = face_policy_losses(policy, [transition], [np.array([2., -1., np.nan])])
        losses[0].backward()
        torch.testing.assert_close(policy.values.grad, torch.tensor([-1., .5, 0.]))
        counts = action_counts(transition)
        self.assertEqual(counts['executed_request_count'], 1)
        self.assertEqual(counts['conflict_count'], 1)
        self.assertEqual(counts['budget_not_considered_count'], 1)
        transition['event']['request_status'][1] = 'budget'
        self.assertEqual(action_counts(transition)['budget_rejected_count'], 1)
        policy.zero_grad()
        losses = face_policy_losses(policy, [transition], [np.array([2., 1., 2.])])
        losses[0].backward()
        torch.testing.assert_close(policy.values.grad, torch.tensor([-1., -.5, 1.]))

    def test_reward_components_distribution(self):
        transition = {'reward_pixels': np.array([2, 1, 0]), 'sil_before': np.array([.5, .8, 0]),
                      'sil_after': np.array([.3, .5, 0]), 'rgb_before': np.array([.7, .4, 0]),
                      'rgb_after': np.array([.4, .3, 0]), 'costs': np.array([2, 0, 0]),
                      'reward': np.array([.7, .9, np.nan])}
        result = reward_distributions(transition, [2., 3., .2], np.array([1., -1., np.nan]))
        self.assertAlmostEqual(result['silhouette']['sum'], 1.)
        self.assertAlmostEqual(result['rgb']['sum'], 1.2)
        self.assertAlmostEqual(result['face_cost']['sum'], -.4)
        self.assertEqual(result['lineage_return']['count'], 2)

    def test_advantages_after_returns_and_clipping(self):
        policy = CandidatePolicy()
        transition = {'features': np.ones((3, 9), dtype=np.float32), 'visible': np.arange(3),
                      'selection_order': np.arange(3), 'considered_count': 3}
        transition['actions'] = np.array([True, False, True])
        raw = np.array([1., 2., 100.])
        _, stats = face_policy_losses(policy, [transition], [raw], 'mean', 'std', .5, True)
        self.assertAlmostEqual(stats[0]['baseline'], raw.mean())
        self.assertEqual(stats[0]['raw']['max'], 100.)
        self.assertLessEqual(stats[0]['processed']['max'], .5)
        self.assertGreaterEqual(stats[0]['processed']['min'], -.5)
        np.testing.assert_array_equal(raw, [1., 2., 100.])

    def test_grouped_components_share_valid_faces(self):
        transition = {'reward_pixels': np.array([1, 1, 0]), 'reward': np.array([.2, .1, np.nan]),
                      'sil_before': np.array([1., 2., 3.]), 'sil_after': np.array([.9, 1., 2.]),
                      'rgb_before': np.array([1., 1., 1.]), 'rgb_after': np.array([.9, .5, .2]),
                      'costs': np.array([2, 0, 0]),
                      'event': {'edge_events': [{'requester_face': 0}], 'passive_faces': [1],
                                'request_status': {0: 'executed', 1: 'conflict', 2: 'budget_not_considered'}}}
        groups = grouped_rewards(transition, [1, 1, .1], np.array([.2, np.nan, 5.]))
        self.assertEqual(groups['executed']['same_valid_set']['face_cost']['count'], 1)
        self.assertEqual(groups['passive']['same_valid_set']['lineage_return']['count'], 0)
        self.assertEqual(groups['unexecuted']['same_valid_set']['rgb']['count'], 1)
        self.assertEqual(groups['not_considered']['same_valid_set']['rgb']['count'], 0)

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
        result, stats = lineage_returns(transitions, 0.5, return_missing=True)
        self.assertAlmostEqual(result[1][1], 7.)
        self.assertAlmostEqual(result[0][0], 1 + .5 * (4 + 3 * 7) / 4)
        self.assertEqual(stats[1]['missing_child_counts'][1], 1)
        self.assertTrue(np.isfinite(result[0][1]))
        transitions[2]['reward'][2] = np.nan
        result = lineage_returns(transitions, 0.5)
        self.assertEqual(result[1][1], 4.)

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

            def refine_and_decimate(self, export_mesh, candidate_order, max_added_faces, decision_fn=None):
                mask = np.isin(np.arange(len(self.triangles)), candidate_order)
                verts, faces, parent, event = subdivide_longest_edges(
                    (self.vertices + self.vertices_offsets).detach().numpy(), self.triangles.numpy(),
                    mask, True, candidate_order, max_added_faces, decision_fn)
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
                                  refine_advantage_baseline='none', refine_advantage_scale='none', refine_advantage_clip=0.,
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
            initial = CandidatePolicy().state_dict()
            self.assertTrue(any(not torch.equal(initial[key], full_state['policy'][key]) for key in initial))
            self.assertEqual(full_state['global_step'], 3)
            self.assertEqual(resumed_state['global_step'], 3)
            torch.testing.assert_close(full_state['vertices'], resumed_state['vertices'])
            torch.testing.assert_close(full_state['triangles'], resumed_state['triangles'])
            np.testing.assert_allclose(full_state['vertex_gradient_ema'], resumed_state['vertex_gradient_ema'])
            np.testing.assert_array_equal(full_state['vertex_gradient_seen'], resumed_state['vertex_gradient_seen'])
            self.assertEqual(len(resumed_state['vertex_mapping']), len(resumed_state['vertices']))
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
            legacy['config'].pop('policy_objective')
            legacy['config'].pop('feature_version')
            legacy['config'].pop('advantage_config')
            legacy['config'].pop('lineage_missing_rule')
            for key in ('refine_advantage_baseline', 'refine_advantage_scale', 'refine_advantage_clip'):
                legacy['config']['options'].pop(key, None)
            legacy['config']['pixel_regions'] = 'pre-update face index for each reward-view pixel'
            legacy_path = os.path.join(directory, 'legacy.pt')
            torch.save(legacy, legacy_path)
            legacy_dir = os.path.join(directory, 'legacy_continued')
            os.mkdir(legacy_dir)
            legacy_trainer = make_trainer(legacy_dir, ckpt, mesh, legacy_path)
            legacy_trainer.opt.refine_multiround_rounds = 1
            with self.assertRaisesRegex(ValueError, 'incompatible input dimensions'):
                refinement_multiround.run_multiround(legacy_trainer, Dataset(), Dataset(), [0], [1])
            legacy_trainer.opt.refine_reinitialize_legacy_policy = True
            refinement_multiround.run_multiround(legacy_trainer, Dataset(), Dataset(), [0], [1])
            restored = torch.load(os.path.join(legacy_dir, 'trajectory.pt'), map_location='cpu')
            self.assertEqual(len(restored['history']), 3)
            self.assertEqual(restored['transitions'][0]['round'], 4)
            self.assertEqual(restored['ema_initialization'], 'legacy_mesh_zero_ema_and_reinitialized_policy')
            self.assertEqual(len(restored['policy_optimizer']['state']), 0)
            static_budget = copy.deepcopy(full_state)
            static_budget['config']['policy_objective'] = 'candidate_bernoulli_ema_v3'
            static_budget['config'].pop('budget_decision_mode')
            static_path = os.path.join(directory, 'static_budget.pt')
            torch.save(static_budget, static_path)
            static_dir = os.path.join(directory, 'static_budget_resume')
            os.mkdir(static_dir)
            static_trainer = make_trainer(static_dir, ckpt, mesh, static_path)
            static_trainer.opt.refine_multiround_rounds = 1
            refinement_multiround.run_multiround(static_trainer, Dataset(), Dataset(), [0], [1])
            static_state = torch.load(os.path.join(static_dir, 'trajectory.pt'), map_location='cpu')
            self.assertEqual(len(static_state['policy_optimizer']['state']), 0)
            self.assertEqual(static_state['ema_initialization'], static_budget['ema_initialization'])
            for key in static_budget['policy']:
                torch.testing.assert_close(static_budget['policy'][key], static_state['policy'][key])
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
            old_open = copy.deepcopy(pending)
            old_open['config']['policy_objective'] = 'candidate_sequential_budget_v4'
            old_open_path = os.path.join(directory, 'old_dynamic_open.pt')
            torch.save(old_open, old_open_path)
            old_open_trainer = make_trainer(paths[2], ckpt, mesh, old_open_path)
            old_open_trainer.opt.refine_multiround_horizon = 6
            with self.assertRaisesRegex(ValueError, 'open segment from a different policy objective'):
                refinement_multiround.run_multiround(old_open_trainer, Dataset(), Dataset(), [0], [1])
            old_completed = copy.deepcopy(full_state)
            old_completed['config']['policy_objective'] = 'candidate_sequential_budget_v4'
            old_completed_path = os.path.join(directory, 'old_dynamic_completed.pt')
            torch.save(old_completed, old_completed_path)
            old_completed_dir = os.path.join(directory, 'old_dynamic_continued')
            os.mkdir(old_completed_dir)
            old_trainer = make_trainer(old_completed_dir, ckpt, mesh, old_completed_path)
            old_trainer.opt.refine_multiround_rounds = 1
            refinement_multiround.run_multiround(old_trainer, Dataset(), Dataset(), [0], [1])
            old_resumed = torch.load(os.path.join(old_completed_dir, 'trajectory.pt'), map_location='cpu')
            self.assertEqual(len(old_resumed['policy_optimizer']['state']), 0)
            for key in old_completed['policy']:
                torch.testing.assert_close(old_completed['policy'][key], old_resumed['policy'][key])
            corrupt = copy.deepcopy(pending)
            corrupt['vertex_gradient_ema'] = corrupt['vertex_gradient_ema'][:-1]
            corrupt_path = os.path.join(directory, 'corrupt_ema.pt')
            torch.save(corrupt, corrupt_path)
            corrupt_dir = os.path.join(directory, 'corrupt_ema')
            os.mkdir(corrupt_dir)
            corrupt_trainer = make_trainer(corrupt_dir, ckpt, mesh, corrupt_path)
            corrupt_trainer.opt.refine_multiround_horizon = 6
            with self.assertRaisesRegex(ValueError, 'vertex EMA or mapping'):
                refinement_multiround.run_multiround(corrupt_trainer, Dataset(), Dataset(), [0], [1])
            old_pending = copy.deepcopy(pending)
            old_pending['config'].pop('policy_objective')
            old_pending_path = os.path.join(directory, 'old_open.pt')
            torch.save(old_pending, old_pending_path)
            old_open_dir = os.path.join(directory, 'old_open')
            os.mkdir(old_open_dir)
            old_open = make_trainer(old_open_dir, ckpt, mesh, old_pending_path)
            old_open.opt.refine_multiround_horizon = 6
            with self.assertRaisesRegex(ValueError, 'open segment'):
                refinement_multiround.run_multiround(old_open, Dataset(), Dataset(), [0], [1])
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
            warm_dir = os.path.join(directory, 'warm')
            warm_resume_dir = os.path.join(directory, 'warm_resume')
            os.mkdir(warm_dir)
            os.mkdir(warm_resume_dir)
            warm = make_trainer(warm_dir, ckpt, mesh)
            warm.opt.refine_multiround_rounds = 1
            warm.opt.refine_gradient_ema_warmup_steps = 2
            refinement_multiround.run_multiround(warm, Dataset(), Dataset(), [0], [1])
            warm_state = torch.load(os.path.join(warm_dir, 'trajectory.pt'), map_location='cpu')
            self.assertEqual(warm_state['global_step'], 3)
            self.assertEqual(warm_state['ema_warmup']['steps'], 2)
            self.assertIn('edge_outcomes', warm_state['transitions'][0])
            warm_resume = make_trainer(warm_resume_dir, ckpt, mesh, os.path.join(warm_dir, 'trajectory.pt'))
            warm_resume.opt.refine_multiround_rounds = 1
            warm_resume.opt.refine_gradient_ema_warmup_steps = 2
            refinement_multiround.run_multiround(warm_resume, Dataset(), Dataset(), [0], [1])
            resumed_warm_state = torch.load(os.path.join(warm_resume_dir, 'trajectory.pt'), map_location='cpu')
            self.assertEqual(resumed_warm_state['global_step'], 4)
            self.assertEqual(resumed_warm_state['ema_warmup']['steps'], 2)


if __name__ == '__main__':
    unittest.main()
