import os
import tempfile
import unittest

import numpy as np
import torch

from meshutils import subdivide_longest_edges
from nerf.refinement_rl import (FacePolicy, executed_rewards, face_features, fixed_region_reward,
                                load_policy, policy_order, policy_update, save_policy)


class RefinementRLTest(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        self.faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    def test_features_only_include_visible_faces(self):
        ids, features, scales = face_features(self.vertices, self.faces, np.array([2., 0.]), np.array([4., 0.]))
        np.testing.assert_array_equal(ids, [0])
        np.testing.assert_allclose(scales, [0.5, 4., 0.5])
        np.testing.assert_allclose(features, np.full((1, 3), np.log(2)), rtol=1e-6)

    def test_budget_and_reward_mapping(self):
        _, faces, parent, event = subdivide_longest_edges(
            self.vertices, self.faces, [True, True], True, [0, 1], 2)
        self.assertEqual(len(faces) - len(self.faces), 2)
        self.assertEqual(event['request_status'], {0: 'executed', 1: 'budget'})
        self.assertEqual(event['passive_faces'], [1])
        np.testing.assert_array_equal(parent, [0, 0, 1, 1])
        before = [(np.array([0, 0, 1, -1]), np.array([3., 5., 7., 0.]))]
        after = [(np.array([-1, 1, 2, -1]), np.array([2., 4., 1., 0.]))]
        control = [(np.array([0, 0, 1, -1]), np.array([3., 6., 8., 0.]))]
        rewards, counts, *_ = executed_rewards(event, parent, before, after, control, 2)
        self.assertEqual(rewards, {0: 1.5})
        np.testing.assert_array_equal(counts, [2, 1])
        event['request_status'][0] = 'conflict'
        self.assertEqual(executed_rewards(event, parent, before, after, control, 2)[0], {})
        before_missing = [(np.array([-1, -1, 1, -1]), np.zeros(4))]
        event['request_status'][0] = 'executed'
        self.assertEqual(executed_rewards(event, parent, before_missing, after, control, 2)[0], {})

    def test_policy_update_checkpoint_and_rng(self):
        torch.manual_seed(1)
        policy = FacePolicy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
        generator = torch.Generator().manual_seed(5)
        features = torch.tensor([[0.5, 0.2, 0.8], [1., 2., 3.]])
        old = [p.detach().clone() for p in policy.parameters()]
        loss = policy_update(policy, optimizer, features, torch.tensor([0, 1]), 1, 0.1)
        self.assertIsNotNone(loss)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(old, policy.parameters())))
        provenance = {'checkpoint_sha256': 'a', 'mesh_sha256': 'b'}
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'policy.pt')
            save_policy(path, policy, optimizer, generator, 3, provenance)
            expected = torch.rand(3, generator=generator)
            restored = FacePolicy()
            restored_optimizer = torch.optim.Adam(restored.parameters())
            restored_generator = torch.Generator()
            self.assertEqual(load_policy(path, restored, restored_optimizer, restored_generator, provenance), 3)
            torch.testing.assert_close(restored(features), policy(features))
            torch.testing.assert_close(torch.rand(3, generator=restored_generator), expected)
            with self.assertRaises(ValueError):
                load_policy(path, restored, restored_optimizer, restored_generator,
                            {'checkpoint_sha256': 'a', 'mesh_sha256': 'different'})

    def test_ordered_probability_gradient_includes_unrequested_candidates(self):
        class LogitPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.tensor([0.3, -0.1, 0.2]))

            def forward(self, features):
                return self.logits

        policy = LogitPolicy()
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
        before = policy.logits.detach().clone()
        order = torch.tensor([0, 1, 2])
        expected_log_probability = (before[0] - torch.logsumexp(before, dim=0) +
                                    before[1] - torch.logsumexp(before[1:], dim=0))
        loss = policy_update(policy, optimizer, torch.zeros(3, 3), order, 2, 1.0)
        self.assertAlmostEqual(loss, float(-expected_log_probability), places=6)
        self.assertNotEqual(float(policy.logits[2]), float(before[2]))

    def test_ranking_below_half_and_budget_conflict(self):
        logits = torch.tensor([-3., -2., -1.])
        generator = torch.Generator().manual_seed(11)
        self.assertEqual(policy_order(logits, generator, 'rank').tolist(), [2, 1, 0])
        self.assertEqual(policy_order(logits, generator, 'threshold').numel(), 0)
        rank = policy_order(logits[:2], generator, 'rank').numpy()
        _, output_faces, _, event = subdivide_longest_edges(
            self.vertices, self.faces, [True, True], True, rank, 2)
        self.assertEqual(len(output_faces), 4)
        self.assertEqual(event['considered_count'], 1)
        self.assertEqual(event['added_faces'], 2)
        vertices = np.array([[0., 0., 0.], [3., 0., 0.], [1.5, 0.1, 0.], [0., -4., 0.]])
        faces = np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int32)
        _, _, _, event = subdivide_longest_edges(vertices, faces, [True, True], True, [0, 1], 3)
        self.assertEqual(event['considered_count'], 2)
        self.assertEqual(event['request_status'], {0: 'executed', 1: 'conflict'})
        _, _, _, event = subdivide_longest_edges(vertices, faces, [True, True], True, [0, 1], 2)
        self.assertEqual(event['considered_count'], 1)
        self.assertEqual(event['request_status'][1], 'budget')

    def test_global_reward_includes_held_and_passive_face_pixels(self):
        before = [(np.array([0, 1, 1, -1]), np.zeros(4))]
        after = [(np.array([0, -1, 1, -1]), np.array([1., 2., 4., 0.]))]
        control = [(np.array([0, 1, 1, -1]), np.array([2., 5., 7., 0.]))]
        self.assertAlmostEqual(fixed_region_reward(before, after, control, 2), 7. / 3)


if __name__ == '__main__':
    unittest.main()
