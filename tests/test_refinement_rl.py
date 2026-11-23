import os
import tempfile
import unittest

import numpy as np
import torch

from meshutils import subdivide_longest_edges
from nerf.refinement_rl import FacePolicy, executed_rewards, face_features, load_policy, policy_update, save_policy


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
        loss = policy_update(policy, optimizer, features, torch.tensor([True, False]), np.array([0, 1]), {0: 0.1})
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


if __name__ == '__main__':
    unittest.main()
