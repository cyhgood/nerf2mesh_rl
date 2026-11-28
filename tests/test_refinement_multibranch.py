import copy
import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from meshutils import subdivide_longest_edges
from nerf.refinement_multibranch import (full_frame_metrics, load_trajectory_policy, run_multibranch,
                                         selection_face_observations)
from nerf.refinement_rl import FacePolicy, file_sha256
from nerf.refinement_candidates import CandidatePolicy, FEATURE_VERSION
from nerf.refinement_gnn import SparseFacePolicy, TextureFacePolicy, RankedTextureFacePolicy, RANK_POLICY_VERSION, POLICY_VERSION, TEXTURE_POLICY_VERSION
from nerf.refinement_texture import TEXTURE_FEATURE_VERSION, sobel_observation, target_sobel_mask


class MultiBranchTest(unittest.TestCase):
    def test_face_observations_raw_features_and_contour_distance(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        observations = [(np.array([0, 0, 1, -1]), np.array([1., 3., 4., 0.]),
                         np.zeros(4), np.array([0., 1., 1., 0.]), (1, 4))]
        _, counts, means, areas, near, distance, views = selection_face_observations(
            observations, 2, vertices, faces, 0.)
        np.testing.assert_array_equal(counts, [2, 1])
        np.testing.assert_allclose(means, [2., 4.])
        np.testing.assert_allclose(areas, [.5, .5])
        np.testing.assert_array_equal(near, [1, 1])
        np.testing.assert_allclose(distance, [0., 0.])
        self.assertEqual(views, 1)

    def test_full_frame_metrics_use_same_pixels(self):
        first = [(np.array([0, -1, -1]), np.array([1., 4., 7.]), np.array([0., 1., 0.5]), None, (1, 3))]
        second = [(np.array([-1, 2, -1]), np.array([3., 2., 1.]), np.array([1., 0., 0.]), None, (1, 3))]
        a = full_frame_metrics(first, [4], [(1, 3)])
        b = full_frame_metrics(second, [4], [(1, 3)])
        self.assertEqual(a['overall']['pixels'], b['overall']['pixels'])
        self.assertEqual(a['overall']['rgb_error'], 4.)
        self.assertEqual(b['overall']['rgb_error'], 2.)
        self.assertEqual(a['overall']['silhouette_error'], 0.5)
        with self.assertRaises(ValueError):
            full_frame_metrics(second, [4], [(3, 1)])

    def test_gradient_fixed_target_region_and_invalid_render(self):
        target = torch.zeros(5, 5, 3)
        target[:, 3:] = 1.
        alpha = torch.ones(5, 5)
        alpha[0, 0] = 0.
        target_mask = target_sobel_mask(target, alpha)
        self.assertEqual(target_mask.sum(), 8)

        def observation(pred, faces, alpha_value=alpha):
            rgb = torch.nn.functional.smooth_l1_loss(pred, target, reduction='none').mean(-1)
            return (faces.numpy().reshape(-1), rgb.numpy().reshape(-1), np.zeros(25),
                    alpha_value.numpy().reshape(-1), (5, 5),
                    *sobel_observation(pred, target, alpha_value, faces), target_sobel_mask(target, alpha_value))

        a = observation(torch.zeros_like(target), torch.zeros(5, 5, dtype=torch.long))
        b = observation(target, torch.full((5, 5), -1, dtype=torch.long))
        ma = full_frame_metrics([a, a], [0, 1], [(5, 5), (5, 5)], [target_mask, target_mask])
        mb = full_frame_metrics([b, b], [0, 1], [(5, 5), (5, 5)], [target_mask, target_mask])
        self.assertEqual(ma['overall']['gradient_valid_pixels'], 16)
        self.assertEqual(mb['overall']['gradient_valid_pixels'], 16)
        self.assertGreater(ma['overall']['gradient_error'], 0.)
        self.assertEqual(mb['overall']['gradient_error'], 0.)
        self.assertEqual(ma['per_view'][0]['gradient_valid_pixels'], 8)
        bad = target.clone(); bad[2, 2, 0] = float('nan')
        invalid = observation(bad, torch.zeros(5, 5, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, 'Nonfinite rendered RGB'):
            full_frame_metrics([invalid], [0], [(5, 5)], [target_mask])
        altered_alpha = alpha.clone(); altered_alpha[2, 2] = 0.
        changed = observation(target, torch.zeros(5, 5, dtype=torch.long), altered_alpha)
        with self.assertRaisesRegex(ValueError, 'foreground mask changed'):
            full_frame_metrics([changed], [0], [(5, 5)], [target_mask])
        empty_alpha = torch.zeros(5, 5)
        empty = observation(target, torch.zeros(5, 5, dtype=torch.long), empty_alpha)
        empty_result = full_frame_metrics([empty], [0], [(5, 5)], [target_sobel_mask(target, empty_alpha)])
        self.assertIsNone(empty_result['overall']['gradient_error'])
        self.assertEqual(empty_result['overall']['gradient_valid_pixels'], 0)

    def test_fixed_policy_budget_and_identical_training_sequences(self):
        starts = []

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
                self.triangles = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)
                self.vertices_offsets = torch.nn.Parameter(torch.full((4, 3), 0.1))
                self.color = torch.nn.Parameter(torch.tensor(0.4))
                self.v_cumsum = np.array([0, 4])
                self.f_cumsum = np.array([0, 2])

            def refine_and_decimate(self, export_mesh, candidate_order, max_added_faces, decision_fn=None):
                starts.append((len(self.triangles), float(self.color.detach()), self.vertices.clone(),
                               self.vertices_offsets.detach().clone()))
                candidates = np.isin(np.arange(len(self.triangles)), candidate_order)
                vertices, faces, parent, event = subdivide_longest_edges(
                    (self.vertices + self.vertices_offsets).detach().numpy(), self.triangles.numpy(),
                    candidates, True, candidate_order, max_added_faces, decision_fn)
                self.vertices = torch.tensor(vertices)
                self.triangles = torch.tensor(faces)
                self.vertices_offsets = torch.nn.Parameter(torch.zeros_like(self.vertices))
                self.parent_face_id = torch.tensor(parent)
                self.refinement_event = event

        class Dataset:
            poses = [0, 1, 2]

            def collate(self, index):
                return {'index': index[0], 'images': torch.ones(1, 3, 4), 'H': 1, 'W': 3}

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, 'stage1.pt')
            mesh = os.path.join(directory, 'mesh.ply')
            for path in (checkpoint, mesh):
                with open(path, 'w') as output:
                    output.write(path)
            torch.manual_seed(4)
            trained_policy = FacePolicy()
            policy_path = os.path.join(directory, 'trajectory.pt')
            torch.save({'policy': trained_policy.state_dict(),
                        'config': {'checkpoint_sha256': file_sha256(checkpoint),
                                   'mesh_sha256': file_sha256(mesh)}}, policy_path)
            loaded = load_trajectory_policy(policy_path, file_sha256(checkpoint), file_sha256(mesh))
            for key, value in trained_policy.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[key])
            self.assertFalse(any(parameter.requires_grad for parameter in loaded.parameters()))
            with self.assertRaises(ValueError):
                load_trajectory_policy(policy_path, 'different', file_sha256(mesh))

            model = Model()
            opt = SimpleNamespace(refine_multiround=False, refine_rl=False, refine_compare_random=False,
                                  refine_rl_eval=False, refine_multibranch_rounds=3, refine_face_budget=2,
                                  refine_multibranch_total_budget=3, refine_random_seeds=[0, 1],
                                  refine_multibranch_policy=policy_path, ckpt=checkpoint, mesh=mesh,
                                  seed=7, refine_experiment_steps=2, refine_experiment_output=directory,
                                  refine=True)
            trainer = SimpleNamespace(opt=opt, model=model, device=torch.device('cpu'), global_step=10,
                                      criterion=torch.nn.SmoothL1Loss(reduction='none'),
                                      scaler=torch.cuda.amp.GradScaler(enabled=False))
            trainer.optimizer_fn = lambda m: torch.optim.Adam(m.parameters(), lr=0.01)
            trainer.lr_scheduler_fn = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
            trainer.post_train_step = lambda: None
            trainer.log = lambda message: None
            used_samples = []

            def eval_step(data):
                count = len(model.triangles)
                model.triangles_errors_id = torch.tensor([[0, min(1, count - 1), -1]])
                model.refinement_alpha = torch.tensor([[0.6, 0.8, 0.]])
                return torch.full((1, 3, 3), float(model.color.detach())), None, torch.ones(1, 3, 3), None

            def train_step(data):
                used_samples.append(data['index'])
                return None, None, (model.color - 0.8).square() + model.vertices_offsets.square().sum()

            trainer.eval_step = eval_step
            trainer.train_step = train_step
            initial_state = copy.deepcopy(model.state_dict())
            run_multibranch(trainer, Dataset(), Dataset(), [0], [1], initial_state,
                            model.vertices.clone(), model.triangles.clone(), model.v_cumsum.copy(),
                            model.f_cumsum.copy(), copy.deepcopy(trainer.scaler.state_dict()), 10)
            with open(os.path.join(directory, 'multibranch_comparison.json')) as source:
                result = json.load(source)
            self.assertEqual([branch['method'] for branch in result['branches']],
                             ['rl', 'high_error', 'random', 'random'])
            self.assertEqual(len(used_samples), 24)
            for offset in range(6, len(used_samples), 6):
                self.assertEqual(used_samples[:6], used_samples[offset:offset + 6])
            for offset in (0, 3, 6, 9):
                self.assertEqual(starts[offset][0], 2)
                self.assertAlmostEqual(starts[offset][1], 0.4, places=6)
                torch.testing.assert_close(starts[0][2], starts[offset][2])
                torch.testing.assert_close(starts[0][3], starts[offset][3])
            for branch in result['branches']:
                self.assertLessEqual(branch['actual_added_faces'], 3)
                self.assertEqual(branch['initial_metrics'], result['initial_metrics'])
                self.assertEqual(branch['final_metrics']['overall']['pixels'], 3)
                for step in branch['rounds']:
                    self.assertLessEqual(step['actual_added_faces'], step['round_budget'])
                    self.assertEqual(step['event']['added_faces'], step['actual_added_faces'])
                    with np.load(os.path.join(directory, step['face_observations'])) as faces:
                        self.assertEqual(len(faces['area']), step['faces_before'])
                        self.assertEqual(len(faces['visible_pixels']), step['faces_before'])
                        self.assertEqual(int(faces['executed_requester'].sum()), len(step['event']['edge_events']))

            candidate_policy = CandidatePolicy()
            with torch.no_grad():
                candidate_policy.network[-1].bias.fill_(10.)
            torch.save({'policy': candidate_policy.state_dict(),
                        'config': {'checkpoint_sha256': file_sha256(checkpoint),
                                   'mesh_sha256': file_sha256(mesh), 'feature_version': FEATURE_VERSION}}, policy_path)
            model = Model()
            trainer.model = model
            trainer.global_step = 10
            opt.refine = True
            opt.refine_candidate_fraction = .5
            opt.refine_min_face_area = 1e-12
            opt.refine_gradient_ema_beta = .9
            opt.refine_observation_graph = True
            opt.refine_experiment_output = os.path.join(directory, 'candidates')
            os.mkdir(opt.refine_experiment_output)
            run_multibranch(trainer, Dataset(), Dataset(), [0], [1], copy.deepcopy(model.state_dict()),
                            model.vertices.clone(), model.triangles.clone(), model.v_cumsum.copy(),
                            model.f_cumsum.copy(), copy.deepcopy(trainer.scaler.state_dict()), 10)
            with open(os.path.join(opt.refine_experiment_output, 'multibranch_comparison.json')) as source:
                candidate_result = json.load(source)
            rl = candidate_result['branches'][0]
            for step in rl['rounds']:
                with np.load(os.path.join(opt.refine_experiment_output, step['observation_graphs']['after'])) as graph:
                    np.testing.assert_array_equal(graph['parent_face_id'], step['event']['parent_face_id'])
                    self.assertEqual(len(graph['face_to_node']), step['faces_after'])
            self.assertEqual(rl['rounds'][0]['candidate_info']['valid_faces'], 2)
            self.assertEqual(rl['rounds'][0]['candidate_info']['candidate_faces'], 1)
            self.assertEqual(rl['rounds'][0]['candidate_info']['gradient_missing'], 1)
            self.assertGreater(rl['rounds'][0]['actual_added_faces'], 0)
            with np.load(os.path.join(opt.refine_experiment_output, rl['rounds'][0]['face_observations'])) as saved:
                candidate_ids = np.flatnonzero(saved['candidate'])
                self.assertEqual(len(candidate_ids), 1)
                self.assertEqual(saved['candidate_features'][candidate_ids[0], 7], 1.)
                self.assertGreater(saved['request_probability'][candidate_ids[0]], .5)
                self.assertIn('rgb_before', saved.files)
                self.assertIn('silhouette_after', saved.files)
            self.assertIn('edge_outcomes', rl['rounds'][0])

            model = Model(); trainer.model = model; trainer.global_step = 10; opt.refine = True
            opt.refine_face_gradient_diagnostics = True
            opt.refine_experiment_output = os.path.join(directory, 'mlp_diagnostics')
            os.mkdir(opt.refine_experiment_output)
            run_multibranch(trainer, Dataset(), Dataset(), [0], [1], copy.deepcopy(model.state_dict()),
                            model.vertices.clone(), model.triangles.clone(), model.v_cumsum.copy(),
                            model.f_cumsum.copy(), copy.deepcopy(trainer.scaler.state_dict()), 10)
            with open(os.path.join(opt.refine_experiment_output, 'multibranch_comparison.json')) as source:
                mlp_diagnostics = json.load(source)
            for original, diagnosed in zip(candidate_result['branches'], mlp_diagnostics['branches']):
                for old_round, new_round in zip(original['rounds'], diagnosed['rounds']):
                    self.assertEqual(old_round['event'], new_round['event'])
                    self.assertEqual(old_round['metrics'], new_round['metrics'])
                    with np.load(os.path.join(directory, 'candidates', old_round['face_observations'])) as old_data, \
                         np.load(os.path.join(opt.refine_experiment_output, new_round['face_observations'])) as new_data:
                        np.testing.assert_array_equal(old_data['selection_region_local_reward'], new_data['selection_region_local_reward'])
            opt.refine_face_gradient_diagnostics = False

            gnn_path = os.path.join(directory, 'gnn.pt')
            torch.save({'policy': SparseFacePolicy().state_dict(), 'config': {
                'checkpoint_sha256': file_sha256(checkpoint), 'mesh_sha256': file_sha256(mesh),
                'feature_version': FEATURE_VERSION, 'policy_type': 'gnn', 'policy_architecture': POLICY_VERSION}}, gnn_path)
            model = Model()
            trainer.model = model
            trainer.global_step = 10
            opt.refine = True
            opt.refine_multibranch_compare_policy = gnn_path
            opt.refine_experiment_output = os.path.join(directory, 'compare_gnn')
            os.mkdir(opt.refine_experiment_output)
            run_multibranch(trainer, Dataset(), Dataset(), [0], [1], copy.deepcopy(model.state_dict()),
                            model.vertices.clone(), model.triangles.clone(), model.v_cumsum.copy(),
                            model.f_cumsum.copy(), copy.deepcopy(trainer.scaler.state_dict()), 10)
            with open(os.path.join(opt.refine_experiment_output, 'multibranch_comparison.json')) as source:
                paired = json.load(source)
            self.assertEqual([branch['policy_type'] for branch in paired['branches'][:2]], ['mlp', 'gnn'])
            for first, second in zip(paired['branches'][0]['rounds'], paired['branches'][1]['rounds']):
                self.assertEqual(first['training_indices'], second['training_indices'])
                self.assertLessEqual(first['round_budget'], paired['per_round_budget'])
                self.assertLessEqual(second['round_budget'], paired['per_round_budget'])
                self.assertLessEqual(second['actual_added_faces'], second['round_budget'])
            texture_path = os.path.join(directory, 'texture.pt')
            torch.save({'policy': TextureFacePolicy().state_dict(), 'config': {
                'checkpoint_sha256': file_sha256(checkpoint), 'mesh_sha256': file_sha256(mesh),
                'feature_version': FEATURE_VERSION, 'policy_type': 'gnn',
                'policy_architecture': TEXTURE_POLICY_VERSION, 'texture_feature_version': TEXTURE_FEATURE_VERSION}}, texture_path)
            opt.refine_multibranch_policy = gnn_path
            opt.refine_multibranch_compare_policy = texture_path
            model = Model(); trainer.model = model; trainer.global_step = 10; opt.refine = True
            opt.refine_experiment_output = os.path.join(directory, 'gnn_texture_compare')
            os.mkdir(opt.refine_experiment_output)
            run_multibranch(trainer, Dataset(), Dataset(), [0], [1], copy.deepcopy(model.state_dict()),
                            model.vertices.clone(), model.triangles.clone(), model.v_cumsum.copy(),
                            model.f_cumsum.copy(), copy.deepcopy(trainer.scaler.state_dict()), 10)
            with open(os.path.join(opt.refine_experiment_output, 'multibranch_comparison.json')) as source:
                texture_pair = json.load(source)
            self.assertEqual([branch['policy_architecture'] for branch in texture_pair['branches'][:2]],
                             [POLICY_VERSION, TEXTURE_POLICY_VERSION])
            with np.load(texture_pair['branches'][1]['rounds'][0]['observation_graphs']['before']) as recorded:
                self.assertEqual(recorded['node_features'].shape[1], 13)
            categorical_path = os.path.join(directory, 'categorical.pt')
            torch.save({'policy': RankedTextureFacePolicy().state_dict(), 'config': {
                'checkpoint_sha256': file_sha256(checkpoint), 'mesh_sha256': file_sha256(mesh),
                'feature_version': FEATURE_VERSION, 'policy_type': 'gnn', 'policy_architecture': RANK_POLICY_VERSION,
                'action_mode': 'categorical', 'allow_stop': False, 'texture_feature_version': TEXTURE_FEATURE_VERSION,
                'options': {'refine_candidate_mode': 'hybrid'}}}, categorical_path)
            opt.refine_multibranch_policy = categorical_path
            opt.refine_multibranch_compare_policy = gnn_path
            model = Model(); trainer.model = model; trainer.global_step = 10; opt.refine = True
            opt.refine_experiment_output = os.path.join(directory, 'categorical_compare')
            os.mkdir(opt.refine_experiment_output)
            with self.assertRaisesRegex(ValueError, 'requires --refine_multibranch_policy_mode rank'):
                run_multibranch(trainer, Dataset(), Dataset(), [0], [1], copy.deepcopy(model.state_dict()),
                                model.vertices.clone(), model.triangles.clone(), model.v_cumsum.copy(),
                                model.f_cumsum.copy(), copy.deepcopy(trainer.scaler.state_dict()), 10)
            opt.refine_multibranch_policy_mode = 'rank'
            run_multibranch(trainer, Dataset(), Dataset(), [0], [1], copy.deepcopy(model.state_dict()),
                            model.vertices.clone(), model.triangles.clone(), model.v_cumsum.copy(),
                            model.f_cumsum.copy(), copy.deepcopy(trainer.scaler.state_dict()), 10)
            with open(os.path.join(opt.refine_experiment_output, 'multibranch_comparison.json')) as source:
                categorical_pair = json.load(source)
            new_branch = categorical_pair['branches'][0]
            self.assertEqual(new_branch['policy_evaluation_mode'], 'categorical_greedy')
            self.assertEqual(new_branch['rounds'][0]['candidate_info']['candidate_mode'], 'hybrid')
            self.assertEqual(new_branch['rounds'][0]['candidate_info']['fallback_count'], 0)
            self.assertGreater(new_branch['actual_added_faces'], 0)
            self.assertFalse(categorical_pair['face_gradient_diagnostics']['enabled'])
            self.assertIsNone(new_branch['rounds'][0]['face_gradient_diagnostics'])
            model = Model(); trainer.model = model; trainer.global_step = 10; opt.refine = True
            opt.refine_face_gradient_diagnostics = True
            opt.refine_experiment_output = os.path.join(directory, 'categorical_diagnostics')
            os.mkdir(opt.refine_experiment_output)
            run_multibranch(trainer, Dataset(), Dataset(), [0], [1], copy.deepcopy(model.state_dict()),
                            model.vertices.clone(), model.triangles.clone(), model.v_cumsum.copy(),
                            model.f_cumsum.copy(), copy.deepcopy(trainer.scaler.state_dict()), 10)
            with open(os.path.join(opt.refine_experiment_output, 'multibranch_comparison.json')) as source:
                diagnosed = json.load(source)
            self.assertTrue(diagnosed['face_gradient_diagnostics']['enabled'])
            for baseline, branch in zip(categorical_pair['branches'], diagnosed['branches']):
                self.assertEqual(baseline['actual_added_faces'], branch['actual_added_faces'])
                for original, step in zip(baseline['rounds'], branch['rounds']):
                    self.assertEqual(original['event'], step['event'])
                    self.assertEqual(original['metrics'], step['metrics'])
                    with open(os.path.join(opt.refine_experiment_output, step['face_gradient_diagnostics']['summary'])) as source:
                        report = json.load(source)
                    self.assertEqual(report['diagnostic_views'], [0])
                    self.assertEqual(report['independent_evaluation_views'], [1])
                    self.assertIn('hybrid_candidates', report['groups'])
                    self.assertIn('topology_change', report['three_phase']['groups']['visible_valid'])
                    self.assertIn('optimization_change', report['three_phase']['groups']['visible_valid'])
                    self.assertEqual(report['actual_added_faces'], step['actual_added_faces'])
                    with np.load(os.path.join(opt.refine_experiment_output, step['face_gradient_diagnostics']['faces'])) as data:
                        self.assertEqual(len(data['old_face_id']), step['faces_before'])
                        np.testing.assert_array_equal(data['parent_face_id'], step['event']['parent_face_id'])
                        self.assertEqual(len(data['three_phase_pixels']), step['faces_before'])
            opt.refine_face_gradient_diagnostics = False
            opt.refine_multibranch_policy_mode = 'threshold'
            opt.refine_multibranch_policy = policy_path
            opt.refine_multibranch_compare_policy = gnn_path
            low_mlp = CandidatePolicy()
            low_gnn = SparseFacePolicy()
            with torch.no_grad():
                for p in low_mlp.parameters():
                    p.zero_()
                low_mlp.network[-1].bias.fill_(-4.)
                for p in low_gnn.parameters():
                    p.zero_()
                low_gnn.action[-1].bias.fill_(-4.)
            common = {'checkpoint_sha256': file_sha256(checkpoint), 'mesh_sha256': file_sha256(mesh),
                      'feature_version': FEATURE_VERSION}
            torch.save({'policy': low_mlp.state_dict(), 'config': common}, policy_path)
            torch.save({'policy': low_gnn.state_dict(), 'config': dict(common, policy_type='gnn',
                                                                       policy_architecture=POLICY_VERSION)}, gnn_path)
            model = Model()
            trainer.model = model
            trainer.global_step = 10
            opt.refine = True
            opt.refine_multibranch_policy_mode = 'rank'
            opt.refine_experiment_output = os.path.join(directory, 'rank_compare')
            os.mkdir(opt.refine_experiment_output)
            run_multibranch(trainer, Dataset(), Dataset(), [0], [1], copy.deepcopy(model.state_dict()),
                            model.vertices.clone(), model.triangles.clone(), model.v_cumsum.copy(),
                            model.f_cumsum.copy(), copy.deepcopy(trainer.scaler.state_dict()), 10)
            with open(os.path.join(opt.refine_experiment_output, 'multibranch_comparison.json')) as source:
                ranked = json.load(source)
            self.assertEqual(ranked['policy_evaluation_mode'], 'rank')
            with np.load(ranked['gradient_metric']['mask_file'], allow_pickle=False) as mask_data:
                self.assertEqual(mask_data['view_1'].shape, (1, 3))
                self.assertEqual(mask_data['view_1'].sum(), 0)
            self.assertEqual(ranked['initial_metrics']['overall']['gradient_valid_pixels'], 0)
            for branch in ranked['branches'][:2]:
                first = branch['rounds'][0]
                self.assertEqual(first['policy_evaluation_mode'], 'rank')
                self.assertEqual(first['metrics']['overall']['gradient_valid_pixels'], 0)
                self.assertIsNone(first['metrics']['overall']['gradient_error'])
                self.assertLess(first['initial_candidate_probability_distribution']['max'], .5)
                self.assertGreater(first['actual_added_faces'], 0)
                self.assertGreater(first['requested_count'], 0)
                self.assertGreater(first['executed_request_count'], 0)
                with np.load(os.path.join(opt.refine_experiment_output, first['face_observations'])) as saved:
                    self.assertEqual(len(saved['decision_order']), first['requested_count'])
                    self.assertTrue(np.isnan(saved['decision_log_probability']).all())
            opt.refine_multibranch_policy_mode = 'threshold'
            opt.refine_multibranch_compare_policy = ''

            model = Model()
            trainer.model = model
            trainer.global_step = 10
            opt.refine = True
            opt.refine_gradient_ema_warmup_steps = 2
            opt.refine_experiment_output = os.path.join(directory, 'warmed')
            os.mkdir(opt.refine_experiment_output)
            run_multibranch(trainer, Dataset(), Dataset(), [0], [1], copy.deepcopy(model.state_dict()),
                            model.vertices.clone(), model.triangles.clone(), model.v_cumsum.copy(),
                            model.f_cumsum.copy(), copy.deepcopy(trainer.scaler.state_dict()), 10)
            with open(os.path.join(opt.refine_experiment_output, 'multibranch_comparison.json')) as source:
                warmed = json.load(source)
            self.assertEqual(warmed['ema_warmup']['steps'], 2)
            self.assertEqual(warmed['initial_global_step'], 12)
            self.assertEqual(warmed['ema_warmup']['observed_vertices'], 4)
            self.assertEqual(warmed['branches'][0]['rounds'][0]['candidate_info']['gradient_missing'], 0)
            self.assertTrue(all(branch['initial_global_step'] == 12 for branch in warmed['branches']))


if __name__ == '__main__':
    unittest.main()
