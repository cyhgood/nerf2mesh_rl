import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from nerf.refinement_texture import sobel_observation, texture_face_sums, fixed_gradient_regions, TEXTURE_FEATURE_VERSION
from nerf.refinement_graph import build_face_graph, add_texture_features
from nerf.refinement_gnn import TextureFacePolicy, TEXTURE_POLICY_VERSION, policy_graph_state
from nerf.refinement_multiround import local_rewards, _observe_silhouette
from nerf.refinement_experiment import _observe
from nerf.refinement_multibranch import load_trajectory_policy
from nerf.refinement_candidates import FEATURE_VERSION
from nerf.refinement_attribution import edge_outcomes


class TextureTest(unittest.TestCase):
    def setUp(self):
        self.target = torch.zeros(5, 5, 3)
        self.pred = torch.zeros_like(self.target)
        self.alpha = torch.ones(5, 5)
        self.face = torch.zeros(5, 5, dtype=torch.long)

    def observation(self, pred=None, target=None, alpha=None, face=None):
        pred = self.pred if pred is None else pred
        target = self.target if target is None else target
        alpha = self.alpha if alpha is None else alpha
        face = self.face if face is None else face
        return (face.numpy().reshape(-1), np.zeros(25), np.zeros(25), alpha.numpy().reshape(-1), (5, 5),
                *sobel_observation(pred, target, alpha, face))

    def test_constant_edge_and_background(self):
        strength, error, fg, covered = sobel_observation(self.pred, self.target, self.alpha, self.face)
        self.assertEqual(covered.sum(), 9)
        self.assertEqual(float(strength.max()), 0.)
        self.assertEqual(float(error.max()), 0.)
        target = self.target.clone()
        target[:, 3:] = 1
        strength, error, _, valid = sobel_observation(self.pred, target, self.alpha, self.face)
        self.assertGreater(strength[valid].max(), 0.)
        self.assertGreater(error[valid].max(), 0.)
        alpha = self.alpha.clone()
        alpha[:, 3:] = 0
        background = self.pred.clone()
        background[:, 3:] = torch.rand(5, 2, 3)
        strength, error, _, valid = sobel_observation(background, self.target, alpha, self.face)
        self.assertEqual(float(strength[valid].max()), 0.)
        self.assertEqual(float(error[valid].max()), 0.)
        self.assertFalse(valid.reshape(5, 5)[:, 2:].any())
        self.assertFalse(valid.reshape(5, 5)[0].any())
        pred = self.pred.clone()
        pred[2, 2, 0] = float('nan')
        _, _, _, valid = sobel_observation(pred, self.target, self.alpha, self.face)
        self.assertFalse(valid.any())

    def test_face_aggregation_fixed_regions_and_missing_reward(self):
        target = self.target.clone()
        target[:, 3:] = 1
        first = self.observation(target=target)
        sums_t, sums_e, counts = texture_face_sums([first, first], 2)
        self.assertEqual(counts.tolist(), [18, 0])
        self.assertGreater(sums_t[0], 0.)
        self.assertGreater(sums_e[0], 0.)
        # After topology changes and the new raster no longer owns pixels, old fixed regions still count.
        after = self.observation(pred=target, target=target, face=torch.full_like(self.face, -1))
        pixels, before, last = fixed_gradient_regions([first], [after], 2)
        self.assertEqual(pixels.tolist(), [9, 0])
        self.assertGreater(before[0], last[0])
        self.assertTrue(np.isnan(before[1]))
        rgb = np.array([1., 1.]); after_rgb = np.array([.9, .9]); sil = np.zeros(2)
        base = local_rewards(rgb, after_rgb, sil, sil, np.ones(2), np.zeros(2), [1, 1, .1, 0], pixels, before, last)
        rewarded = local_rewards(rgb, after_rgb, sil, sil, np.ones(2), np.zeros(2), [1, 1, .1, 2], pixels, before, last)
        self.assertTrue(np.isfinite(base).all())
        self.assertGreater(rewarded[0], base[0])
        self.assertAlmostEqual(rewarded[1], base[1])

    def test_shared_edge_gradient_attribution(self):
        event = {'edge_events': [{'edge': [0, 1], 'requester_face': 0, 'added_faces': 2,
                                  'affected_faces': [0, 1]}], 'passive_faces': [1]}
        counts = np.array([9, 0])
        first = np.array([.4, np.nan]); after = np.array([.1, np.nan])
        outcomes = edge_outcomes(event, 2, counts, first, after, counts, first, after, counts, first, after)
        sides = outcomes['edges'][0]['faces']
        self.assertAlmostEqual(sides[0]['gradient']['improvement'], .3)
        self.assertTrue(sides[1]['passively_split'])
        self.assertFalse(sides[1]['gradient']['valid'])
        self.assertIsNone(sides[1]['gradient']['improvement'])

    def test_graph_policy_and_checkpoints(self):
        vertices = np.array([[0., 0, 0], [1., 0, 0], [1., 1, 0], [0, 1., 0]])
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        graph, ids, _, _ = build_face_graph(vertices, faces, np.ones(2), np.ones(2), np.ones(2), 1,
                                             np.zeros(4), np.ones(4, bool), 2, 2)
        target = self.target.clone(); target[:, 3:] = 1
        add_texture_features(graph, [self.observation(target=target)])
        self.assertEqual(graph['node_features'].shape[1], 13)
        self.assertEqual(graph['texture_valid'].tolist(), [True, False])
        state = policy_graph_state(graph)
        self.assertEqual(state['texture_feature_version'], TEXTURE_FEATURE_VERSION)
        policy = TextureFacePolicy()
        for p in policy.parameters():
            torch.nn.init.constant_(p, .05)
        first = policy.logits(state, ids, [1.])
        state2 = dict(state, node_features=state['node_features'].copy())
        state2['node_features'][0, 10] += 1
        second = policy.logits(state2, ids, [1.])
        self.assertGreater(float((first - second).abs().sum()), 1e-6)
        optimizer = torch.optim.Adam(policy.parameters(), lr=.01)
        before = policy.node_encoder.weight.detach().clone()
        optimizer.zero_grad(); first.sum().backward(); optimizer.step()
        self.assertFalse(torch.equal(before, policy.node_encoder.weight))
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'policy.pt')
            cfg = {'checkpoint_sha256': 'a', 'mesh_sha256': 'b', 'feature_version': FEATURE_VERSION,
                   'policy_type': 'gnn', 'policy_architecture': TEXTURE_POLICY_VERSION,
                   'texture_feature_version': TEXTURE_FEATURE_VERSION}
            torch.save({'policy': policy.state_dict(), 'config': cfg}, path)
            restored = load_trajectory_policy(path, 'a', 'b')
            self.assertIsInstance(restored, TextureFacePolicy)
            torch.testing.assert_close(restored.logits(state, ids, [1.]), policy.logits(state, ids, [1.]))
            cfg['texture_feature_version'] = 'other'
            torch.save({'policy': policy.state_dict(), 'config': cfg}, path)
            with self.assertRaisesRegex(ValueError, 'feature version mismatch'):
                load_trajectory_policy(path, 'a', 'b')

    def test_train_eval_observation_feature_consistency(self):
        target = self.target.clone(); target[:, 3:] = 1
        images = torch.cat((target, self.alpha[..., None]), dim=-1)
        class Dataset:
            def collate(inner, index):
                return {'images': images.clone(), 'H': 5, 'W': 5}
        model = SimpleNamespace(eval=lambda: None, triangles_errors_id=self.face,
                                refinement_alpha=self.alpha)
        trainer = SimpleNamespace(model=model, criterion=torch.nn.SmoothL1Loss(reduction='none'),
                                  eval_step=lambda data: (self.pred, None, target, None))
        training = _observe(trainer, Dataset(), [0], texture=True)[0]
        evaluation = _observe_silhouette(trainer, Dataset(), [0], texture=True)[0]
        for a, b in zip(training[-4:], evaluation[-4:]):
            np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
