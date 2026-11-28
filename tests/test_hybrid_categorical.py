import unittest
import numpy as np
import torch

from meshutils import subdivide_longest_edges
from nerf.refinement_candidates import candidate_features, CategoricalFaceDecisions
from nerf.refinement_multiround import categorical_policy_losses


class FakePolicy(torch.nn.Module):
    def __init__(self, stop=-20.):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        self.stop = torch.nn.Parameter(torch.tensor(stop))
    def encode(self, graph):
        return torch.as_tensor(graph['node_features'][:, :2]) * self.weight
    def score(self, hidden, budget):
        b = torch.as_tensor(budget, dtype=torch.float32)
        return hidden[:, 0] * b + hidden[:, 1] * (1 - b)
    def stop_score(self, budget):
        return self.stop + torch.as_tensor(budget) * 0


class HybridCategoricalTest(unittest.TestCase):
    def test_hybrid_selection_ties_and_fallback(self):
        scales = np.sqrt(np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, .5]) * 2)
        vertices = np.concatenate([np.array([[i * 20, 0, 0], [i * 20 + s, 0, 0], [i * 20, s, 0]]) for i, s in enumerate(scales)])
        faces = np.arange(30).reshape(-1, 3)
        tex_count = np.ones(10, dtype=int)
        error = np.array([0, 0, 0, 0, 2, 2, 1, 0, 8, 0.])
        args = (vertices, faces, np.ones(10), np.ones(10), np.ones(10), 1,
                np.zeros(30), np.zeros(30, bool), 1.)
        ids, _, info = candidate_features(*args, candidate_mode='hybrid', texture_error_sum=error, texture_pixels=tex_count)
        self.assertEqual(ids.tolist(), [0, 1, 2, 3, 8])
        self.assertEqual((info['area_count'], info['texture_count'], info['fallback_count']), (4, 1, 0))
        self.assertEqual(info['groups']['texture']['face_ids'], [8])
        error[8] = 0
        ids, _, _ = candidate_features(*args, candidate_mode='hybrid', texture_error_sum=error, texture_pixels=tex_count)
        self.assertEqual(ids[-1], 4)
        ids, _, info = candidate_features(*args, candidate_mode='hybrid', texture_error_sum=error, texture_pixels=np.zeros(10))
        self.assertEqual(ids.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(info['fallback_count'], 1)
        empty = list(args); empty[3] = np.zeros(10)
        ids, _, _ = candidate_features(*empty, candidate_mode='hybrid', texture_error_sum=error, texture_pixels=tex_count)
        self.assertEqual(len(ids), 0)

    def geometry(self):
        v = np.array([[0., 0., 0.], [3., 0., 0.], [1.5, .1, 0.], [0., -4., 0.],
                      [10., 0., 0.], [12., 0., 0.], [10., 1., 0.],
                      [20., 0., 0.], [22., 0., 0.], [20., 1., 0.]])
        f = np.array([[0, 1, 2], [1, 0, 3], [4, 5, 6], [7, 8, 9]])
        x = np.zeros((4, 9), np.float32)
        x[:, :2] = [[5, 0], [4, 0], [0, 4], [0, 1]]
        graph = {'face_to_node': np.arange(4), 'node_to_face': np.arange(4), 'node_features': x}
        return v, f, x, graph

    def test_dynamic_budget_greedy_conflict_and_one_face(self):
        v, f, x, g = self.geometry()
        policy = FakePolicy()
        dec = CategoricalFaceDecisions(policy, np.arange(4), x, 3, g, deterministic=True)
        _, _, _, event = subdivide_longest_edges(v, f, np.ones(4, bool), True, np.arange(4), 3, dec)
        self.assertEqual(dec.decision_order, [0, 2])
        self.assertEqual(event['added_faces'], 3)
        np.testing.assert_array_equal(dec.remaining, [3, -1, 1, -1])
        x[:, :2] = [[5, 5], [4, 4], [3, 3], [2, 2]]
        dec = CategoricalFaceDecisions(policy, np.arange(4), x, 3, g, deterministic=True)
        _, _, _, event = subdivide_longest_edges(v, f, np.ones(4, bool), True, np.arange(4), 3, dec)
        self.assertEqual(event['request_status'][1], 'conflict')
        self.assertEqual(dec.remaining.tolist(), [3, 1, 1, -1])
        dec = CategoricalFaceDecisions(policy, np.arange(4), x, 1, g, deterministic=True)
        _, _, _, event = subdivide_longest_edges(v, f, np.ones(4, bool), True, np.arange(4), 1, dec)
        self.assertEqual(event['request_status'][0], 'budget')
        self.assertEqual(event['request_status'][1], 'executed')
        self.assertEqual(event['added_faces'], 1)

    def test_explicit_stop(self):
        v, f, x, g = self.geometry()
        policy = FakePolicy(stop=6.)
        dec = CategoricalFaceDecisions(policy, np.arange(4), x, 3, g, deterministic=True, allow_stop=True)
        _, _, _, event = subdivide_longest_edges(v, f, np.ones(4, bool), True, np.arange(4), 3, dec)
        self.assertTrue(event['stopped'])
        self.assertEqual(event['added_faces'], 0)
        self.assertEqual(set(event['request_status'].values()), {'stopped'})
        self.assertEqual(dec.stop_event['remaining_candidates'], 4)
        transition = {'visible': np.arange(4), 'policy_graph': g, 'event': event,
                      'decision_order': np.array([], dtype=int), 'remaining_budget': dec.remaining,
                      'round_budget': 3, 'allow_stop': True, 'stop_event': dec.stop_event}
        losses, _ = categorical_policy_losses(policy, [transition], [np.ones(4)])
        self.assertAlmostEqual(losses[0].item(), -dec.stop_event['log_probability'], places=5)
        losses[0].backward()
        self.assertTrue(torch.isfinite(policy.stop.grad))
        self.assertNotEqual(policy.stop.grad.item(), 0.)

    def test_sampled_logprob_replay_and_update(self):
        v, f, x, g = self.geometry()
        policy = FakePolicy()
        dec = CategoricalFaceDecisions(policy, np.arange(4), x, 3, g, generator=torch.Generator().manual_seed(7))
        _, _, _, event = subdivide_longest_edges(v, f, np.ones(4, bool), True, np.arange(4), 3, dec)
        duplicate = CategoricalFaceDecisions(policy, np.arange(4), x, 3, g, generator=torch.Generator().manual_seed(7))
        _, _, _, duplicate_event = subdivide_longest_edges(v, f, np.ones(4, bool), True, np.arange(4), 3, duplicate)
        self.assertEqual(dec.decision_order, duplicate.decision_order)
        self.assertEqual(event['request_status'], duplicate_event['request_status'])
        transition = {'visible': np.arange(4), 'policy_graph': g, 'event': event,
                      'decision_order': np.array(dec.decision_order), 'remaining_budget': dec.remaining,
                      'round_budget': 3, 'allow_stop': False, 'stop_event': None}
        # Replay exactly the sampled categorical log probabilities. A rejected
        # request has zero direct credit; each executing edge adds passive returns.
        returns = np.ones(4)
        expected = 0.
        edges = {e['requester_face']: e for e in event['edge_events']}
        for face in dec.decision_order:
            credit = 1 + sum(int(n in event['passive_faces']) for n in edges[face]['affected_faces']) if face in edges else 0
            expected -= dec.log_probabilities[face] * credit
        losses, _ = categorical_policy_losses(policy, [transition], [returns])
        self.assertAlmostEqual(losses[0].item(), expected, places=5)
        old = policy.weight.detach().clone()
        optimizer = torch.optim.SGD(policy.parameters(), lr=.01)
        optimizer.zero_grad(); losses[0].backward(); optimizer.step()
        self.assertFalse(torch.equal(old, policy.weight))


if __name__ == '__main__':
    unittest.main()
