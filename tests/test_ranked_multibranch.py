import unittest

import numpy as np
import torch

from meshutils import subdivide_longest_edges
from nerf.refinement_candidates import RankedFaceDecisions, SequentialFaceDecisions


class BudgetModel(torch.nn.Module):
    def forward(self, features):
        budget = features[:, 6]
        return features[:, 0] * budget + features[:, 1] * (1 - budget) - 10


class GraphBudgetModel(torch.nn.Module):
    def encode(self, graph):
        return torch.from_numpy(graph['node_features'][:, :2].copy())
    def score(self, embeddings, budgets):
        budget = torch.as_tensor(budgets, dtype=torch.float32)
        return embeddings[:, 0] * budget + embeddings[:, 1] * (1 - budget) - 10


class RankedMultibranchTest(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0., 0., 0.], [3., 0., 0.], [1.5, .1, 0.], [0., -4., 0.],
                                  [10., 0., 0.], [12., 0., 0.], [10., 1., 0.],
                                  [20., 0., 0.], [22., 0., 0.], [20., 1., 0.]])
        self.faces = np.array([[0, 1, 2], [1, 0, 3], [4, 5, 6], [7, 8, 9]])
        self.features = np.zeros((4, 9), np.float32)
        self.features[:, :2] = [[5, 0], [4, 0], [0, 4], [0, 1]]
        self.ids = np.arange(4)

    def run_policy(self, model, budget, graph=None):
        dec = RankedFaceDecisions(model, self.ids, self.features, budget, graph=graph)
        result = subdivide_longest_edges(self.vertices, self.faces, np.ones(4, bool), True,
                                        self.ids, budget, dec)
        return dec, result[-1]

    def test_below_threshold_and_dynamic_rescoring(self):
        threshold = SequentialFaceDecisions(BudgetModel(), self.ids, self.features, 3, deterministic=True)
        _, _, _, event = subdivide_longest_edges(self.vertices, self.faces, np.ones(4, bool), True,
                                                 self.ids, 3, threshold)
        self.assertEqual(event['added_faces'], 0)
        ranked, event = self.run_policy(BudgetModel(), 3)
        self.assertTrue((ranked.initial_probabilities < .5).all())
        self.assertEqual(event['considered_faces'], [0, 2])
        self.assertEqual(ranked.decision_order, [0, 2])
        self.assertEqual(event['added_faces'], 3)
        self.assertEqual(ranked.remaining.tolist(), [3, -1, 1, -1])
        self.assertEqual(event['request_status'][1], 'budget_not_considered')
        self.assertTrue(np.isnan(ranked.log_probabilities).all())

    def test_conflict_budget_rejection_and_boundary(self):
        self.features[:, :2] = [[5, 5], [4, 4], [3, 3], [2, 2]]
        ranked, event = self.run_policy(BudgetModel(), 3)
        self.assertEqual(event['considered_faces'], [0, 1, 2])
        self.assertEqual(event['request_status'][1], 'conflict')
        self.assertEqual(ranked.remaining.tolist(), [3, 1, 1, -1])
        self.assertEqual(event['added_faces'], 3)
        ranked, event = self.run_policy(BudgetModel(), 1)
        self.assertEqual(event['request_status'][0], 'budget')
        self.assertEqual(event['request_status'][1], 'executed')
        self.assertEqual(event['request_status'][2], 'budget_not_considered')
        self.assertEqual(ranked.remaining.tolist(), [1, 1, -1, -1])
        self.assertEqual(event['added_faces'], 1)

    def test_graph_and_mlp_use_same_ranking_rule(self):
        mapping = np.array([2, 0, 3, 1])
        inverse = np.argsort(mapping)
        graph = {'face_to_node': mapping, 'node_to_face': inverse,
                 'node_features': self.features[inverse].copy()}
        mlp, mlp_event = self.run_policy(BudgetModel(), 3)
        gnn, gnn_event = self.run_policy(GraphBudgetModel(), 3, graph)
        self.assertEqual(mlp_event['considered_faces'], gnn_event['considered_faces'])
        self.assertEqual(mlp_event['request_status'], gnn_event['request_status'])
        np.testing.assert_allclose(mlp.probabilities, gnn.probabilities, equal_nan=True)
        np.testing.assert_array_equal(mlp.remaining, gnn.remaining)

    def test_same_shared_edge_is_only_split_once(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.],
                             [3., 0., 0.], [4., 0., 0.], [3., 1., 0.]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6]])
        ids = np.arange(3)
        features = np.zeros((3, 9), np.float32)
        features[:, :2] = [[3, 3], [2, 2], [1, 1]]
        decisions = RankedFaceDecisions(BudgetModel(), ids, features, 3)
        _, _, parents, event = subdivide_longest_edges(vertices, faces, np.ones(3, bool), True, ids, 3, decisions)
        self.assertEqual(event['request_status'], {0: 'executed', 1: 'already_split', 2: 'executed'})
        self.assertEqual(decisions.remaining.tolist(), [3, 1, 1])
        np.testing.assert_array_equal(np.bincount(parents), [2, 2, 2])

    def test_equal_scores_use_global_face_id(self):
        ids = np.array([3, 1, 2, 0])
        features = np.zeros((4, 9), np.float32)
        decisions = RankedFaceDecisions(BudgetModel(), ids, features, 1)
        _, _, _, event = subdivide_longest_edges(self.vertices, self.faces, np.ones(4, bool), True,
                                                 ids, 1, decisions)
        self.assertEqual(event['considered_faces'][:2], [0, 1])
        self.assertEqual(event['request_status'][0], 'budget')
        self.assertEqual(event['request_status'][1], 'executed')


if __name__ == '__main__':
    unittest.main()
