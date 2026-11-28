import json
import os
import tempfile
import time
import unittest

import numpy as np

from meshutils import subdivide_longest_edges
from nerf.refinement_candidates import candidate_features
from nerf.refinement_graph import build_face_graph, save_face_graph, graph_budget_state


def graph(vertices, faces, pixels=None, sums=None, remaining=2, budget=2, self_loops=False):
    n = len(faces)
    pixels = np.ones(n, dtype=np.int64) if pixels is None else np.asarray(pixels)
    sums = pixels.astype(float) if sums is None else np.asarray(sums)
    return build_face_graph(vertices, faces, sums, pixels, (pixels > 0).astype(int), 1,
                            np.ones(len(vertices)), np.ones(len(vertices), dtype=bool),
                            remaining, budget, self_loops=self_loops)


class RefinementGraphTest(unittest.TestCase):
    def setUp(self):
        self.v = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
        self.f = np.array([[0, 1, 2], [0, 2, 3]])

    def test_shared_edge_parent_and_rebuilt_graph(self):
        before, candidates, features, info = graph(self.v, self.f, pixels=[1, 0])
        self.assertEqual(candidates.tolist(), [0])
        self.assertEqual(before['node_to_face'].tolist(), [0, 1])
        self.assertEqual(before['candidate_mask'].tolist(), [True, False])
        self.assertEqual(before['rgb_valid'].tolist(), [True, False])
        self.assertEqual(before['node_features'][1, -1], 1)
        self.assertEqual(before['edge_index'].tolist(), [[0, 1], [1, 0]])
        self.assertEqual(before['metadata']['boundary_edges'], 4)
        np.testing.assert_allclose(before['edge_features'][:, 0], np.sqrt(2))
        np.testing.assert_allclose(before['node_features'][before['face_to_node'][candidates], :9], features)
        v, f, parent, event = subdivide_longest_edges(self.v, self.f, [True, False], True, [0], 2)
        self.assertEqual(event['passive_faces'], [1])
        after, _, _, _ = graph(v, f, remaining=0)
        self.assertEqual(len(after['node_to_face']), 4)
        self.assertEqual(len(after['edge_index'][0]), 8)
        self.assertFalse(after['decision_mask'].any())
        self.assertTrue(np.all(after['node_features'][:, 6] == 0))
        self.assertEqual(np.bincount(parent).tolist(), [2, 2])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'graph.npz')
            save_face_graph(path, after, parent)
            with np.load(path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data['parent_face_id'], parent)
                self.assertEqual(json.loads(str(data['metadata_json']))['budget']['remaining'], 0)

    def test_boundaries_degenerate_nonmanifold_and_loops(self):
        v = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., -1., 0.], [0., 0., 1.],
                      [3., 0., 0.], [4., 0., 0.], [5., 0., 0.]])
        f = np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4], [5, 6, 7]])
        result, candidates, _, _ = graph(v, f, pixels=[0, 0, 0, 0], self_loops=True)
        self.assertEqual(result['metadata']['nonmanifold_edges'], 1)
        self.assertEqual(result['metadata']['boundary_edges'], 6)
        self.assertEqual(result['metadata']['invalid_faces'], 1)
        self.assertEqual(result['face_to_node'].tolist(), [0, 1, 2, -1])
        self.assertEqual(candidates.size, 0)
        self.assertEqual(result['edge_index'].tolist(), [[0, 1, 2], [0, 1, 2]])
        self.assertFalse(result['edge_feature_valid'].any())
        self.assertFalse(result['decision_mask'].any())
        empty, _, _, _ = graph(np.empty((0, 3)), np.empty((0, 3), dtype=int), pixels=[])
        self.assertEqual(empty['edge_index'].shape, (2, 0))

    def test_feature_and_budget_consistency(self):
        args = (self.v, self.f, np.array([3., 4.]), np.array([2, 2]), np.array([1, 1]), 2,
                np.arange(4, dtype=float), np.ones(4, dtype=bool))
        g, ids, x, info = build_face_graph(*args, 1, 4)
        expected_ids, expected_x, _ = candidate_features(*args, .25)
        np.testing.assert_array_equal(ids, expected_ids)
        np.testing.assert_allclose(x, expected_x)
        np.testing.assert_allclose(g['node_features'][:, 6], .25)
        self.assertEqual(info['candidate_faces'], 1)
        self.assertTrue(g['decision_mask'].any())
        updated, mask = graph_budget_state(g, 0, 4)
        self.assertFalse(mask.any())
        self.assertTrue(np.all(updated[:, 6] == 0))
        np.testing.assert_allclose(g['node_features'][:, 6], .25)
        with self.assertRaisesRegex(ValueError, 'budget'):
            build_face_graph(*args, 5, 4)
        bad_ema = np.array([np.nan, 1., 1., 1.])
        missing, _, _, _ = build_face_graph(self.v, self.f, np.ones(2), np.ones(2), np.ones(2), 1,
                                             bad_ema, np.ones(4, dtype=bool), 2, 2)
        self.assertFalse(missing['gradient_valid'].any())
        self.assertTrue(np.isfinite(missing['node_features']).all())

    def test_large_mesh_sparse_storage(self):
        side = 225
        y, x = np.mgrid[:side, :side]
        vertices = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))).astype(np.float32)
        a = (y[:-1, :-1] * side + x[:-1, :-1]).ravel()
        faces = np.concatenate((np.column_stack((a, a + 1, a + side)),
                                np.column_stack((a + 1, a + side + 1, a + side))))
        start = time.monotonic()
        result, _, _, _ = graph(vertices, faces)
        elapsed = time.monotonic() - start
        size = sum(value.nbytes for value in result.values() if isinstance(value, np.ndarray))
        self.assertGreater(len(faces), 100000)
        self.assertLess(result['edge_index'].shape[1], len(faces) * 3)
        self.assertLess(size, len(faces) * 500)
        print('sparse graph benchmark: {} faces, {:.2f}s, {:.1f} MiB arrays'.format(len(faces), elapsed, size / 2**20))


if __name__ == '__main__':
    unittest.main()
