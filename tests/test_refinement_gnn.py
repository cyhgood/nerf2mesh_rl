import copy
import os
import tempfile
import time
import unittest

import numpy as np
import torch

from nerf.refinement_candidates import SequentialFaceDecisions, FEATURE_VERSION
from nerf.refinement_gnn import SparseFacePolicy, POLICY_VERSION, policy_graph_state
from nerf.refinement_multibranch import load_trajectory_policy
from nerf.refinement_multiround import face_policy_losses
from test_refinement_graph import graph


class GNNTest(unittest.TestCase):
    def setUp(self):
        self.v = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        self.f = np.array([[0, 1, 2], [0, 2, 3]])
        g, self.ids, self.features, _ = graph(self.v, self.f, pixels=[1, 0])
        self.state = policy_graph_state(g)
        self.policy = SparseFacePolicy()
        for parameter in self.policy.parameters():
            torch.nn.init.constant_(parameter, .05)

    def test_neighbour_and_budget_affect_logits(self):
        first = self.policy.logits(self.state, self.ids, np.array([1.]))
        altered = copy.deepcopy(self.state)
        altered['node_features'][1, 0] += 1.
        other = self.policy.logits(altered, self.ids, np.array([1.]))
        self.assertGreater(float((first - other).abs().max()), 1e-6)
        less = self.policy.logits(self.state, self.ids, np.array([.25]))
        self.assertGreater(float((first - less).abs().max()), 1e-5)
        altered_budget = copy.deepcopy(self.state)
        altered_budget['node_features'][:, 6] = 0
        torch.testing.assert_close(first, self.policy.logits(altered_budget, self.ids, np.array([1.])))
        wrong = copy.deepcopy(self.state)
        wrong['face_to_node'][self.ids[0]] = -1
        with self.assertRaisesRegex(ValueError, 'graph snapshot'):
            self.policy.logits(wrong, self.ids, np.array([1.]))

    def test_sampling_recompute_and_gradient(self):
        dec = SequentialFaceDecisions(self.policy, self.ids, self.features, 4,
                                      generator=torch.Generator().manual_seed(2), graph=self.state)
        dec(int(self.ids[0]), 1)
        logits = self.policy.logits(self.state, self.ids, dec.features[:, 6])
        logged = -torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.as_tensor(dec.actions, dtype=torch.float32), reduction='none')
        np.testing.assert_allclose(logged.detach().numpy(), dec.log_probabilities, rtol=1e-6)
        transition = {'visible': self.ids, 'features': dec.features, 'actions': dec.actions,
                      'decision_mask': dec.decided, 'policy_graph': self.state,
                      'event': {'passive_faces': [], 'edge_events': []}}
        loss = face_policy_losses(self.policy, [transition], [np.array([2., 0.])])[0]
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.policy.parameters() if p.grad is not None), 0.)
        self.assertIsNotNone(self.policy.message[0].weight.grad)

    def test_empty_neighbour_and_checkpoint(self):
        empty, no_ids, _, _ = graph(np.empty((0, 3)), np.empty((0, 3), dtype=np.int64), pixels=[])
        self.assertEqual(self.policy.encode(policy_graph_state(empty)).shape, (0, 32))
        self.assertEqual(len(no_ids), 0)
        isolated, ids, x, _ = graph(self.v[:3], np.array([[0, 1, 2]]))
        isolated = policy_graph_state(isolated)
        self.assertEqual(isolated['edge_index'].shape, (2, 0))
        self.assertTrue(torch.isfinite(self.policy.logits(isolated, ids, [1.])).all())
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'trajectory.pt')
            torch.save({'policy': self.policy.state_dict(), 'config': {
                'checkpoint_sha256': 'a', 'mesh_sha256': 'b', 'feature_version': FEATURE_VERSION,
                'policy_type': 'gnn', 'policy_architecture': POLICY_VERSION}}, path)
            loaded = load_trajectory_policy(path, 'a', 'b')
            self.assertIsInstance(loaded, SparseFacePolicy)
            torch.testing.assert_close(self.policy.logits(self.state, self.ids, [1.]), loaded.logits(self.state, self.ids, [1.]))
            value = torch.load(path, map_location='cpu')
            value['config']['policy_type'] = 'mlp'
            torch.save(value, path)
            with self.assertRaisesRegex(ValueError, 'architecture mismatch'):
                load_trajectory_policy(path, 'a', 'b')

    def test_large_graph_forward(self):
        side = 225
        y, x = np.mgrid[:side, :side]
        vertices = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))).astype(np.float32)
        a = (y[:-1, :-1] * side + x[:-1, :-1]).ravel()
        faces = np.concatenate((np.column_stack((a, a + 1, a + side)),
                                np.column_stack((a + 1, a + side + 1, a + side))))
        g, ids, _, _ = graph(vertices, faces)
        state = policy_graph_state(g)
        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            start = time.monotonic()
            with torch.no_grad():
                h = self.policy.encode(state)
                logits = self.policy.score(h[torch.as_tensor(state['face_to_node'][ids])], np.ones(len(ids)))
            elapsed = time.monotonic() - start
        finally:
            torch.set_num_threads(old_threads)
        self.assertEqual(len(logits), len(ids))
        self.assertTrue(torch.isfinite(logits).all())
        size = sum(value.nbytes for value in state.values() if isinstance(value, np.ndarray))
        print('GNN benchmark: {} faces, forward {:.2f}s, input {:.1f} MiB (excludes activations/peak)'.format(len(faces), elapsed, size / 2**20))


if __name__ == '__main__':
    unittest.main()
