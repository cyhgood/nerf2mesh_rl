import random
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from nerf.refinement_loss_gradient_diagnostics import analyze_loss_gradients, parameter_groups, summarize
from nerf.refinement_experiment import _train_branch
from nerf.refinement_texture import stage1_image_gradient_loss, stage1_texture_weighted_gradient_loss
from nerf.silhouette_loss import stage1_foreground_alpha, stage1_silhouette_loss
from test_stage1_silhouette import load_methods


class GradientDiagnosticTest(unittest.TestCase):
    def test_grouping_sign_scale_and_invalid(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vertices_offsets = torch.nn.Parameter(torch.tensor([1., 2.]))
                self.color_net = torch.nn.Linear(1, 1, bias=False)
                self.sigma_net = torch.nn.Linear(1, 1, bias=False)
                self.individual_codes = torch.nn.Parameter(torch.tensor([1.]))
        model = Model()
        opt = torch.optim.SGD(model.parameters(), lr=.1)
        groups = parameter_groups(model, opt)
        self.assertEqual([n for n, _ in groups['appearance']], ['individual_codes', 'color_net.weight'])
        rgb = model.vertices_offsets.sum() + model.color_net.weight.sum() + model.individual_codes.sum()
        grad = -rgb
        model.vertices_offsets.grad = torch.tensor([3., 4.])
        result = analyze_loss_gradients(model, opt, rgb, grad, 1, .1, amp_scale=128.)
        for group in result.values():
            self.assertTrue(group['valid'])
            self.assertAlmostEqual(group['cosine'], -1.)
            self.assertAlmostEqual(group['image_gradient']['scaled_norm'], group['image_gradient']['norm'] * .1)
        torch.testing.assert_close(model.vertices_offsets.grad, torch.tensor([3., 4.]))
        missing = analyze_loss_gradients(model, opt, rgb, model.vertices_offsets.sum(), 1, 1.)
        self.assertEqual(missing['appearance']['image_gradient']['reason'], 'missing_parameter_gradient')
        zero = analyze_loss_gradients(model, opt, rgb, rgb * 0, 1, 1.)
        self.assertEqual(zero['vertex_offsets']['image_gradient']['reason'], 'norm_too_small')
        nonfinite = analyze_loss_gradients(model, opt, rgb, rgb * float('nan'), 1, 1.)
        self.assertEqual(nonfinite['vertex_offsets']['image_gradient']['reason'], 'nonfinite_loss')
        empty = analyze_loss_gradients(model, opt, rgb, grad, 0, 1.)
        self.assertEqual(empty['appearance']['image_gradient']['reason'], 'no_valid_gradient_pixels')
        self.assertEqual(summarize([{'groups': result}])['appearance']['negative_fraction'], 1.)

    def test_sampled_training_matches_without_diagnostics(self):
        namespace = {'torch': torch, 'stage1_foreground_alpha': stage1_foreground_alpha,
                     'stage1_silhouette_loss': stage1_silhouette_loss,
                     'stage1_image_gradient_loss': stage1_image_gradient_loss,
                     'stage1_texture_weighted_gradient_loss': stage1_texture_weighted_gradient_loss}
        train_step, = load_methods('nerf/utils.py', 'Trainer', ['train_step'], namespace)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vertices_offsets = torch.nn.Parameter(torch.tensor([.2]))
                self.encoder_color = torch.nn.Linear(1, 1, bias=False)
                self.color_net = torch.nn.Linear(1, 1, bias=False)
                self.specular_net = torch.nn.Linear(1, 1, bias=False)
            def render_stage1(self, rays_o, rays_d, mvp, H, W, bg_color=None, **kwargs):
                x = torch.linspace(.1, 1., H * W).reshape(-1, 1)
                alpha = torch.sigmoid(self.vertices_offsets).expand(H * W)
                color = torch.sigmoid(self.color_net(self.encoder_color(x)) + self.specular_net(x))
                return {'image': alpha[:, None] * color.expand(-1, 3) + (1 - alpha[:, None]) * bg_color,
                        'alpha': alpha}
            def update_triangles_errors(self, loss):
                pass
        torch.manual_seed(11)
        initial = Model().state_dict()
        def run(enabled, coefficient, scaled=False):
            model = Model(); model.load_state_dict(initial)
            opt = SimpleNamespace(stage=1, color_space='srgb', background='random', refine=True,
                                  lambda_lpips=0, lambda_lap=0, lambda_normal=0, lambda_edgelen=0,
                                  lambda_offsets=0, lambda_sil=0, lambda_tv=0, lambda_image_grad=coefficient)
            tr = SimpleNamespace(opt=opt, model=model, device=torch.device('cpu'), global_step=0,
                                 criterion=torch.nn.SmoothL1Loss(reduction='none'))
            tr.train_step = lambda data: train_step(tr, data)
            tr.scaler = (torch.amp.GradScaler('cpu', init_scale=128.) if scaled
                         else torch.cuda.amp.GradScaler(enabled=False))
            tr.optimizer_fn = lambda m: torch.optim.Adam(m.parameters(), lr=.01)
            tr.lr_scheduler_fn = lambda o: torch.optim.lr_scheduler.LambdaLR(o, lambda _: 1.)
            tr.post_train_step = lambda: None
            def collate(_):
                images = torch.zeros(25, 4); images[:, 3] = 1
                images.reshape(5, 5, 4)[:, 3:, :3] = 1
                return {'rays_o': torch.zeros(25, 3), 'rays_d': torch.ones(25, 3), 'index': [0],
                        'mvp': torch.eye(4).unsqueeze(0), 'H': 5, 'W': 5, 'images': images}
            stats = _train_branch(tr, SimpleNamespace(collate=collate), [0, 1, 2], 12,
                                  gradient_diagnostics={'interval': 2, 'max_samples': 2} if enabled else None)
            return tr, stats, (random.random(), np.random.rand(), torch.rand(1))
        cases = [(0., False), (.1, False)]
        if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
            cases.append((.1, True))
        for coefficient, scaled in cases:
            baseline, _, rng = run(False, coefficient, scaled)
            measured, stats, rng_measured = run(True, coefficient, scaled)
            for a, b in zip(baseline.model.parameters(), measured.model.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
                for key in baseline.optimizer.state[a]:
                    torch.testing.assert_close(baseline.optimizer.state[a][key], measured.optimizer.state[b][key], rtol=0, atol=0)
            self.assertEqual(baseline.scaler.state_dict(), measured.scaler.state_dict())
            self.assertEqual(baseline.lr_scheduler.state_dict(), measured.lr_scheduler.state_dict())
            self.assertEqual(rng[:2], rng_measured[:2]); torch.testing.assert_close(rng[2], rng_measured[2])
            samples = stats['loss_gradient_diagnostics']['samples']
            self.assertEqual([s['step_in_round'] for s in samples], [1, 3])
            self.assertEqual(samples[0]['amp_scale'], 128. if scaled else 1.)
            self.assertTrue(samples[0]['groups']['appearance']['valid'])
            self.assertAlmostEqual(samples[0]['groups']['appearance']['image_gradient']['scaled_norm'],
                                   coefficient * samples[0]['groups']['appearance']['image_gradient']['norm'])


if __name__ == '__main__':
    unittest.main()
