import unittest

import numpy as np

from meshutils import subdivide_longest_edges


class LocalRefinementTest(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        self.faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    def test_shared_edge_splits_both_faces(self):
        vertices, faces, parents = subdivide_longest_edges(
            self.vertices, self.faces, [True, False]
        )
        self.assertEqual(vertices.shape, (5, 3))
        self.assertEqual(faces.shape, (4, 3))
        np.testing.assert_allclose(vertices[4], [0.5, 0.5, 0.0])
        np.testing.assert_array_equal(parents, [0, 0, 1, 1])
        self.assertTrue(np.all(np.any(faces == 4, axis=1)))
        normals = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                           vertices[faces[:, 2]] - vertices[faces[:, 0]])
        self.assertTrue(np.all(normals[:, 2] > 0))

        _, _, parents, event = subdivide_longest_edges(
            self.vertices, self.faces, [True, False], return_event=True
        )
        self.assertEqual(event['requested_faces'], [0])
        self.assertEqual(event['executed_edges'], [[0, 2]])
        self.assertEqual(event['request_status'], {0: 'executed'})
        self.assertEqual(event['passive_faces'], [1])
        np.testing.assert_array_equal(parents, [0, 0, 1, 1])

    def test_empty_candidates(self):
        vertices, faces, parents = subdivide_longest_edges(
            self.vertices, self.faces, [False, False]
        )
        np.testing.assert_array_equal(vertices, self.vertices)
        np.testing.assert_array_equal(faces, self.faces)
        np.testing.assert_array_equal(parents, [0, 1])

    def test_shared_edge_requested_twice(self):
        vertices, faces, parents = subdivide_longest_edges(
            self.vertices, self.faces, [True, True]
        )
        self.assertEqual(vertices.shape, (5, 3))
        self.assertEqual(faces.shape, (4, 3))
        np.testing.assert_array_equal(parents, [0, 0, 1, 1])
        _, _, _, event = subdivide_longest_edges(self.vertices, self.faces, [True, True], return_event=True)
        self.assertEqual(event['request_status'], {0: 'executed', 1: 'executed'})
        self.assertEqual(event['passive_faces'], [])

    def test_conflicting_edges_and_boundary(self):
        vertices = np.array([[0., 0., 0.], [3., 0., 0.], [0., 1., 0.],
                             [0., -4., 0.]])
        faces = np.array([[0, 1, 2], [1, 0, 3]], dtype=np.int32)
        result_vertices, result_faces, parents = subdivide_longest_edges(
            vertices, faces, [True, True]
        )
        self.assertEqual(result_vertices.shape, (6, 3))
        self.assertEqual(result_faces.shape, (4, 3))
        np.testing.assert_array_equal(parents, [0, 0, 1, 1])

        # The first face requests the common edge, so the second request conflicts.
        vertices[2] = [1.5, 0.1, 0.]
        result_vertices, result_faces, parents = subdivide_longest_edges(
            vertices, faces, [True, True]
        )
        self.assertEqual(result_vertices.shape, (5, 3))
        self.assertEqual(result_faces.shape, (4, 3))
        np.testing.assert_array_equal(parents, [0, 0, 1, 1])
        _, _, _, event = subdivide_longest_edges(vertices, faces, [True, True], return_event=True)
        self.assertEqual(event['request_status'], {0: 'executed', 1: 'conflict'})
        self.assertEqual(event['passive_faces'], [1])


if __name__ == '__main__':
    unittest.main()
