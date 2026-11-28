import unittest

import numpy as np

from nerf.refinement_gradient_diagnostics import gradient_diagnostics, three_phase_gradient_diagnostics


def observation(faces, errors, coverage, finite=None):
    faces = np.asarray(faces, dtype=np.int64)
    errors = np.asarray(errors, dtype=np.float64)
    if finite is None:
        finite = np.ones(len(faces), dtype=bool)
    return (faces, np.zeros(len(faces)), np.zeros(len(faces)), np.ones(len(faces)), (2, 4),
            np.zeros(len(faces)), errors, np.asarray(finite, bool), np.asarray(coverage, bool))


class GradientDiagnosticsTest(unittest.TestCase):
    def test_fixed_regions_losses_and_lineage(self):
        old = observation([0, 0, 0, 1, 1, 1, 2, 2], [.2, .6, 1, 2, 2, 2, 3, 3], [1] * 8)
        new = observation([0, -1, 2, 99, 2, 3, -1, -1], [.1, .5, .9, 1, 1.5, 0, .1, .1],
                          [1] * 8, [1, 1, 1, 1, 1, 0, 1, 1])
        summary, arrays = gradient_diagnostics([old], [new], 3, [1, 1, 1], [1, 0, 1],
                                                [1, 0, 0], [0, 1, 0], [0, 0, 1, 1], [4], [9])
        groups = summary['groups']
        self.assertAlmostEqual(summary['coverage_denominator_gradient_error_sum'], 13.8)
        self.assertAlmostEqual(groups['hybrid_candidates']['pre_gradient_error_coverage'], 7.8 / 13.8)
        self.assertAlmostEqual(groups['executed_requesters']['pre_gradient_error_coverage'], 1.8 / 13.8)
        self.assertEqual(groups['visible_valid']['pre_valid_pixels'], 8)
        self.assertEqual(groups['executed_requesters']['lost_coverage_pixels'], 1)
        self.assertEqual(groups['executed_requesters']['different_parent_pixels'], 1)
        self.assertEqual(groups['passive_neighbors']['invalid_correspondence_pixels'], 1)
        self.assertEqual(groups['passive_neighbors']['invalid_after_pixels'], 1)
        self.assertEqual(groups['unchanged']['no_child_mapping_faces'], 1)
        self.assertIsNone(groups['unchanged']['mean_improvement'])
        self.assertAlmostEqual(groups['executed_requesters']['mean_improvement'], .1)
        self.assertAlmostEqual(groups['passive_neighbors']['mean_improvement'], .5)
        self.assertEqual(arrays['paired_pixels'].tolist(), [3, 1, 0])
        self.assertAlmostEqual(groups['visible_valid']['pixel_weighted_improvement'], .2)
        self.assertEqual(summary['diagnostic_views'], [4])
        self.assertEqual(summary['independent_evaluation_views'], [9])
        doubled, _ = gradient_diagnostics([old, old], [new, new], 3, [1, 1, 1], [1, 0, 1],
                                           [1, 0, 0], [0, 1, 0], [0, 0, 1, 1], [4, 5], [9])
        self.assertEqual(doubled['groups']['visible_valid']['pre_valid_pixels'], 16)
        self.assertAlmostEqual(doubled['groups']['visible_valid']['pixel_weighted_improvement'], .2)

    def test_no_valid_gradients_and_invalid_parent(self):
        old = observation([0] * 8, [0] * 8, [0] * 8)
        new = observation([-1] * 8, [0] * 8, [0] * 8)
        summary, arrays = gradient_diagnostics([old], [new], 1, [1], [1], [1], [0], [-1], [0], [1])
        group = summary['groups']['visible_valid']
        self.assertEqual(group['pre_no_valid_gradient_faces'], 1)
        self.assertIsNone(group['pre_gradient_error_coverage'])
        self.assertIsNone(group['positive_improvement_fraction'])
        self.assertEqual(group['no_child_mapping_faces'], 1)
        self.assertTrue(np.isnan(arrays['pre_gradient_error'][0]))
        with self.assertRaisesRegex(ValueError, 'matching'):
            gradient_diagnostics([old], [], 1, [1], [1], [1], [0], [0], [0], [1])

    def test_three_phases_share_intersection_and_decompose_change(self):
        old = observation([0, 0, 0, 1, 1, 1, 2, 2], [.2, .6, 1, 2, 2, 2, 3, 3], [1] * 8)
        split = observation([0, -1, 2, 99, 2, 3, -1, -1], [.15, .7, .8, 1, 1.75, 0, .1, .1],
                            [1] * 8, [1, 1, 0, 1, 1, 1, 1, 1])
        final = observation([0, -1, 2, 99, 2, 3, -1, -1], [.1, .5, .9, 1, 1.5, 0, .1, .1],
                            [1] * 8, [1, 1, 1, 1, 1, 0, 1, 1])
        report, arrays = three_phase_gradient_diagnostics([old], [split], [final], 3, [1, 1, 1], [1, 0, 1],
                                                           [1, 0, 0], [0, 1, 0], [0, 0, 1, 1], [4], [9])
        self.assertEqual(arrays['three_phase_pixels'].tolist(), [2, 1, 0])
        self.assertAlmostEqual(arrays['topology_improvement'][0], -.025)
        self.assertAlmostEqual(arrays['optimization_improvement'][0], .125)
        self.assertAlmostEqual(arrays['topology_improvement'][0] + arrays['optimization_improvement'][0], .1)
        group = report['groups']['executed_requesters']
        self.assertEqual(group['after_split_coverage_and_mapping']['lost_coverage_pixels'], 1)
        self.assertEqual(group['after_optimization_coverage_and_mapping']['lost_coverage_pixels'], 1)
        self.assertEqual(group['valid_pixels'], 2)
        self.assertIsNone(report['groups']['unchanged']['after_split_error'])


if __name__ == '__main__':
    unittest.main()
