import importlib.util
from pathlib import Path
import unittest

import numpy as np


spec = importlib.util.spec_from_file_location(
    'analyze_face_credit', Path(__file__).resolve().parents[1] / 'scripts' / 'analyze_face_credit.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def comparison_branch(method, seed, parent, edge):
    return {'method': method, 'random_seed': seed, 'rounds': [
        {'round': 1, 'actual_added_faces': 2,
         'event': {'edge_events': [{'requester_face': edge, 'affected_faces': [0, 1]}],
                   'parent_face_id': parent}},
        {'round': 2, 'actual_added_faces': 1,
         'event': {'edge_events': [{'requester_face': 2, 'affected_faces': [2]}],
                   'parent_face_id': [0, 1, 2, 2, 3]}}]}


class AnalyzeFaceCreditTest(unittest.TestCase):
    def test_overlap_follows_initial_ancestors(self):
        data = {'initial_faces': 2, 'branches': [
            comparison_branch('rl', None, [0, 0, 1, 1], 0),
            comparison_branch('random', 4, [0, 1, 0, 1], 1)]}
        report = analysis.compare_branches(data)['comparisons'][0]['rounds']
        self.assertEqual(report[0]['requester_initial_ancestor_overlap']['intersection'], 0)
        self.assertEqual(report[0]['affected_initial_ancestor_overlap']['jaccard'], 1)
        self.assertEqual(report[1]['requester_initial_ancestor_overlap']['intersection'], 0)

    def test_training_features_rewards_and_missing_returns(self):
        transition = {
            'round': 1, 'faces_before': 2,
            'event': {'edge_events': [{'requester_face': 0, 'affected_faces': [0, 1]}],
                      'passive_faces': [1]},
            'visible': np.array([0, 1]), 'features': np.log1p(np.ones((2, 3))),
            'feature_scales': np.array([.5, 10., .2]),
            'reward': np.array([.5, .2]), 'lineage_return': np.array([1., np.nan]),
            'reward_pixels': np.array([2, 1]), 'silhouette_pixels': np.array([4, 1]),
            'costs': np.array([2, 0]), 'sil_before': np.array([.6, .4]),
            'sil_after': np.array([.4, .3]), 'rgb_before': np.array([.5, .3]),
            'rgb_after': np.array([.2, .2]), 'parent_face_id': np.array([0, 0, 1]),
            'child_areas': np.array([1., 1., 2.]), 'probabilities': np.array([.8, .2]),
            'selection_order': np.array([0, 1]), 'considered_count': 1,
        }
        next_transition = dict(transition, round=2, faces_before=3,
                               visible=np.array([0, 1]), reward=np.array([.1, .2, .3]),
                               lineage_return=np.array([np.nan, .4, .5]))
        report = analysis.training_rounds({'config': {'weights': [2., 3., .1]},
                                           'history': [], 'transitions': [transition, next_transition]})
        first = report['rounds'][0]
        self.assertEqual(first['executed_requesters']['face_count'], 1)
        self.assertEqual(first['passively_split']['face_count'], 1)
        self.assertAlmostEqual(first['executed_requesters']['selection_rgb_mean_approx']['mean'], .5)
        self.assertEqual(first['executed_requesters']['assigned_uncovered_target_pixels'], 2)
        self.assertAlmostEqual(first['executed_requesters']['weighted_face_cost']['mean'], -.2)
        self.assertEqual(first['valid_local_but_invalid_return_count'], 1)
        self.assertEqual(first['parents_with_valid_local_and_invalid_next_descendant_including_possible_segment_boundaries'], 1)


if __name__ == '__main__':
    unittest.main()
