import unittest

import numpy as np
import torch

from meshutils import subdivide_longest_edges
from nerf.refinement_candidates import SequentialFaceDecisions
from nerf.refinement_multiround import face_policy_losses


class DynamicBudgetTest(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0., 0., 0.], [3., 0., 0.], [1.5, .1, 0.], [0., -4., 0.],
                                  [10., 0., 0.], [12., 0., 0.], [10., 1., 0.],
                                  [20., 0., 0.], [22., 0., 0.], [20., 1., 0.],
                                  [30., 0., 0.], [32., 0., 0.], [30., 1., 0.]])
        self.faces = np.array([[0, 1, 2], [1, 0, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

    def test_budget_conflict_hold_and_exhaustion(self):
        observed = []
        def decide(face, remaining):
            observed.append((face, remaining))
            return face != 2
        _, faces, parents, event = subdivide_longest_edges(
            self.vertices, self.faces, [True] * 5, True, np.arange(5), 3, decide)
        self.assertEqual(observed, [(0, 3), (1, 1), (2, 1), (3, 1), (4, 0)])
        self.assertEqual(event['request_status'], {0: 'executed', 1: 'conflict', 2: 'hold',
                                                   3: 'executed', 4: 'budget'})
        self.assertEqual(event['added_faces'], 3)
        self.assertEqual(len(faces) - len(self.faces), 3)
        self.assertEqual(np.bincount(parents, minlength=5).tolist(), [2, 2, 1, 2, 1])

    def test_insufficient_budget_does_not_charge(self):
        observed = []
        def decide(face, remaining):
            observed.append(remaining)
            return True
        _, _, _, event = subdivide_longest_edges(
            self.vertices, self.faces, [True, False, True, False, False], True, [0, 2], 1, decide)
        self.assertEqual(observed, [1, 1])
        self.assertEqual(event['request_status'], {0: 'budget', 2: 'executed'})
        self.assertEqual(event['added_faces'], 1)

    def test_training_and_evaluation_receive_same_states(self):
        class AlwaysRequest(torch.nn.Module):
            def forward(self, features):
                return torch.full((len(features),), 100.)
        features = np.zeros((5, 9), dtype=np.float32)
        train = SequentialFaceDecisions(AlwaysRequest(), np.arange(5), features, 3,
                                        generator=torch.Generator().manual_seed(3))
        evaluation = SequentialFaceDecisions(AlwaysRequest(), np.arange(5), features, 3, deterministic=True)
        result_a = subdivide_longest_edges(self.vertices, self.faces, [True] * 5, True, np.arange(5), 3, train)
        result_b = subdivide_longest_edges(self.vertices, self.faces, [True] * 5, True, np.arange(5), 3, evaluation)
        np.testing.assert_array_equal(train.remaining, evaluation.remaining)
        np.testing.assert_array_equal(train.features, evaluation.features)
        np.testing.assert_array_equal(train.actions, evaluation.actions)
        self.assertEqual(result_a[3]['request_status'], result_b[3]['request_status'])
        np.testing.assert_allclose(train.features[train.decided, 6], [1., 1/3, 1/3, 0., 0.], rtol=1e-6)
        self.assertTrue(np.isfinite(train.log_probabilities[train.decided]).all())
        selected = torch.from_numpy(train.features[train.decided])
        sampled = torch.from_numpy(train.actions[train.decided].astype(np.float32))
        recomputed = -torch.nn.functional.binary_cross_entropy_with_logits(
            AlwaysRequest()(selected), sampled, reduction='none')
        np.testing.assert_allclose(recomputed.numpy(), train.log_probabilities[train.decided], atol=1e-6)
        empty_budget = SequentialFaceDecisions(AlwaysRequest(), np.arange(5), features, 0, deterministic=True)
        _, _, _, event = subdivide_longest_edges(self.vertices, self.faces, [True] * 5, True,
                                                 np.arange(5), 0, empty_budget)
        self.assertTrue(empty_budget.decided.all())
        np.testing.assert_array_equal(empty_budget.features[:, 6], np.zeros(5))
        self.assertEqual(event['added_faces'], 0)

    def test_gradient_uses_recorded_budget_and_passive_return(self):
        class BudgetPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.))
            def forward(self, features):
                return self.weight * features[:, 6]
        policy = BudgetPolicy()
        features = np.zeros((2, 9), dtype=np.float32)
        features[:, 6] = [1., .5]
        transition = {'visible': np.array([0, 1]), 'features': features,
                      'actions': np.array([True, False]), 'decision_mask': np.array([True, True]),
                      'event': {'passive_faces': [1], 'edge_events': [
                          {'requester_face': 0, 'affected_faces': [0, 1]}]}}
        losses = face_policy_losses(policy, [transition], [np.array([1., 3.])])
        losses[0].backward()
        self.assertAlmostEqual(float(policy.weight.grad), -2.)
        policy.zero_grad()
        transition['event']['passive_faces'] = []
        losses = face_policy_losses(policy, [transition], [np.array([1., 3.])])
        losses[0].backward()
        self.assertAlmostEqual(float(policy.weight.grad), .25)


if __name__ == '__main__':
    unittest.main()
