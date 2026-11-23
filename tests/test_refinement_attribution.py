import unittest
from types import SimpleNamespace

import numpy as np

from nerf.refinement_attribution import edge_outcomes
from nerf.refinement_experiment import _warmup_vertex_ema


class AttributionTest(unittest.TestCase):
    def test_shared_edge_two_sides_and_missing_observation(self):
        event = {'edge_events': [{'edge': [0, 2], 'requester_face': 0,
                                  'affected_faces': [0, 1], 'added_faces': 2}],
                 'passive_faces': [1]}
        result = edge_outcomes(event, 3, np.array([2, 0, 1]), np.array([.8, np.nan, .4]),
                               np.array([.6, np.nan, .3]), np.array([1, 3, 0]),
                               np.array([.5, .9, np.nan]), np.array([.4, .6, np.nan]))
        edge = result['edges'][0]
        self.assertEqual(edge['requester_face'], 0)
        self.assertEqual(edge['added_faces'], 2)
        self.assertFalse(edge['faces'][1]['rgb']['valid'])
        self.assertIsNone(edge['faces'][1]['rgb']['improvement'])
        self.assertTrue(edge['faces'][1]['passively_split'])
        self.assertAlmostEqual(edge['combined_silhouette']['pixel_weighted_improvement'], .25)
        self.assertEqual(result['groups']['unchanged_faces']['rgb']['valid_faces'], 1)
        self.assertEqual(result['groups']['unchanged_faces']['silhouette']['valid_faces'], 0)

    def test_zero_warmup_leaves_state_unchanged(self):
        state = {'ema': np.array([2.]), 'seen': np.array([True])}
        result = _warmup_vertex_ema(None, SimpleNamespace(poses=[0]), state, 0, .9, 42)
        self.assertEqual(result['steps'], 0)
        self.assertEqual(result['training_indices'], [])
        self.assertEqual(state['ema'][0], 2.)
        with self.assertRaises(ValueError):
            _warmup_vertex_ema(None, SimpleNamespace(poses=[0]), state, -1, .9, 42)


if __name__ == '__main__':
    unittest.main()
