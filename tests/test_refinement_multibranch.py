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

            def refine_and_decimate(self, export_mesh, candidate_order, max_added_faces):
                starts.append((len(self.triangles), float(self.color.detach()), self.vertices.clone(),
                               self.vertices_offsets.detach().clone()))
                candidates = np.isin(np.arange(len(self.triangles)), candidate_order)
                vertices, faces, parent, event = subdivide_longest_edges(
                    (self.vertices + self.vertices_offsets).detach().numpy(), self.triangles.numpy(),
                    candidates, True, candidate_order, max_added_faces)
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


if __name__ == '__main__':
    unittest.main()
