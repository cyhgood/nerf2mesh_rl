import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from nerf.refinement_texture import stage1_image_gradient_loss, sobel_observation
from nerf.silhouette_loss import stage1_foreground_alpha, stage1_silhouette_loss
from nerf.refinement_experiment import _train_branch
from test_stage1_silhouette import load_methods


class Stage1ImageGradientTest(unittest.TestCase):
    def test_renderer_ssaa_image_alignment(self):
        class Raster:
            def rasterize(self, ctx, vertices, triangles, shape):
                return torch.ones(1, *shape, 4), None
            def interpolate(self, attr, rast, triangles):
                return torch.ones(1, *rast.shape[1:3], attr.shape[-1]), None
            def antialias(self, value, rast, vertices, triangles, **kwargs):
                return value * .4 + vertices[:, :1, :1].reshape(1, 1, 1, 1) * .1
        def resize(value, size, mag='bilinear', min='bilinear'):
            mode = mag if size[0] >= value.shape[0] else min
            return F.interpolate(value.permute(2, 0, 1)[None], size=size, mode=mode)[0].permute(1, 2, 0)
        render, = load_methods('nerf/renderer.py', 'NeRFRenderer', ['render_stage1'],
                               {'torch': torch, 'F': F, 'dr': Raster(), 'scale_img_hwc': resize,
                                'scale_img_hw': lambda value, size, **kwargs: F.interpolate(value[None, None], size=size, mode='nearest')[0, 0]})
        for ssaa in (1, 2):
            with self.subTest(ssaa=ssaa):
                offset = torch.nn.Parameter(torch.zeros(3, 3))
                model = SimpleNamespace(opt=SimpleNamespace(ssaa=ssaa, fp16=False, pos_gradient_boost=1,
                                        enable_offset_nerf_grad=False, refine_multiround=False),
                                        vertices=torch.ones(3, 3), vertices_offsets=offset,
                                        triangles=torch.tensor([[0, 1, 2]], dtype=torch.int32),
                                        glctx=None, individual_dim=0,
                                        rgb=lambda xyz, dirs, code, shading: (torch.ones_like(xyz), None))
                rays = torch.ones(25, 3)
                outputs = render(model, rays, rays, torch.eye(4), 5, 5)
                self.assertEqual(outputs['image'].shape, (25, 3))
                loss, pixels = stage1_image_gradient_loss(outputs['image'].reshape(5, 5, 3),
                                                           torch.zeros(5, 5, 3), torch.ones(5, 5))
                self.assertEqual(pixels, 9)
                loss.backward()
                self.assertTrue(torch.isfinite(offset.grad).all())

    def test_known_edge_mask_and_empty(self):
        target = torch.zeros(5, 5, 3)
        target[:, 3:] = 1.
        pred = torch.zeros_like(target, requires_grad=True)
        alpha = torch.ones(5, 5)
        loss, pixels = stage1_image_gradient_loss(pred, target, alpha)
        _, error, valid, _ = sobel_observation(pred, target, alpha, torch.zeros(5, 5))
        self.assertEqual(pixels, 9)
        self.assertAlmostEqual(loss.item(), float(error[valid].mean()), places=7)
        self.assertGreater(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(pred.grad.abs().sum().item(), 0.)
        pred = torch.zeros_like(target, requires_grad=True)
        empty, pixels = stage1_image_gradient_loss(pred, target, torch.zeros(5, 5))
        self.assertEqual(pixels, 0)
        self.assertEqual(empty.item(), 0.)
        empty.backward()
        self.assertEqual(pred.grad.abs().sum().item(), 0.)
        mask = torch.ones(5, 5); mask[:, 3:] = 0
        background = torch.zeros_like(target); background[:, 3:] = torch.rand(5, 2, 3)
        masked, count = stage1_image_gradient_loss(background, torch.zeros_like(target), mask)
        self.assertEqual(count, 3)
        self.assertEqual(masked.item(), 0.)
        invalid = torch.zeros_like(target); invalid[2, 2, 0] = float('nan')
        _, count = stage1_image_gradient_loss(invalid, torch.zeros_like(target), alpha)
        self.assertEqual(count, 0)
        with self.assertRaisesRegex(ValueError, 'requires RGBA'):
            stage1_foreground_alpha(torch.zeros(25, 3), {}, (5, 5), 'cpu')

    def test_training_branch_and_fixed_background(self):
        namespace = {'torch': torch, 'stage1_silhouette_loss': stage1_silhouette_loss,
                     'stage1_foreground_alpha': stage1_foreground_alpha,
                     'stage1_image_gradient_loss': stage1_image_gradient_loss}
        train_step, = load_methods('nerf/utils.py', 'Trainer', ['train_step'], namespace)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vertices_offsets = torch.nn.Parameter(torch.tensor([.2]))
                self.color = torch.nn.Parameter(torch.tensor(.4))
                self.errors = None
            def render_stage1(self, rays_o, rays_d, mvp, H, W, bg_color=None, **kwargs):
                alpha = torch.sigmoid(self.vertices_offsets).expand(H * W)
                pattern = torch.linspace(0., 1., H * W).reshape(-1, 1).expand(-1, 3)
                image = alpha[:, None] * self.color * pattern + (1 - alpha[:, None]) * bg_color
                return {'image': image, 'alpha': alpha}
            def update_triangles_errors(self, loss):
                self.errors = loss.clone()
        opt = SimpleNamespace(stage=1, color_space='srgb', background='random', refine=True,
                              lambda_lpips=0, lambda_lap=0, lambda_normal=0, lambda_edgelen=0,
                              lambda_offsets=0, lambda_sil=0, lambda_tv=0, lambda_image_grad=0)
        model = Model()
        trainer = SimpleNamespace(opt=opt, model=model, device=torch.device('cpu'), global_step=0,
                                  criterion=torch.nn.SmoothL1Loss(reduction='none'))
        trainer.train_step = lambda data: train_step(trainer, data)
        images = torch.zeros(25, 4); images[:, 3] = 1.; images.reshape(5, 5, 4)[:, 3:, :3] = 1
        data = {'rays_o': torch.zeros(25, 3), 'rays_d': torch.ones(25, 3), 'index': [0],
                'mvp': torch.eye(4).unsqueeze(0), 'H': 5, 'W': 5, 'images': images}
        torch.manual_seed(0)
        _, _, base = trainer.train_step(data)
        self.assertIsNone(trainer.loss_image_grad)
        base_errors = model.errors.clone()
        base_grad = torch.autograd.grad(base, (model.color, model.vertices_offsets))
        opt.lambda_image_grad = 2.
        torch.manual_seed(0)
        _, _, total = trainer.train_step(data)
        torch.testing.assert_close(base_errors, model.errors)
        self.assertEqual(trainer.image_grad_valid_pixels, 9)
        torch.testing.assert_close(trainer.weighted_loss_image_grad, trainer.loss_image_grad * 2.)
        self.assertAlmostEqual(total.item() - base.item(), trainer.weighted_loss_image_grad.item(), places=6)
        total_grad = torch.autograd.grad(total, (model.color, model.vertices_offsets))
        self.assertTrue(all(torch.isfinite(v).all() for v in total_grad))
        self.assertTrue(any(not torch.allclose(a, b) for a, b in zip(base_grad, total_grad)))
        loss0 = trainer.loss_image_grad.clone()
        torch.manual_seed(9)
        trainer.train_step(data)
        torch.testing.assert_close(loss0, trainer.loss_image_grad, atol=1e-7, rtol=1e-6)
        trainer.scaler = torch.cuda.amp.GradScaler(enabled=False)
        trainer.optimizer_fn = lambda m: torch.optim.SGD(m.parameters(), lr=.1)
        trainer.lr_scheduler_fn = lambda opt: torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
        trainer.post_train_step = lambda: None
        stats = _train_branch(trainer, SimpleNamespace(collate=lambda _: data), [0, 0], 0)
        self.assertEqual(len(stats['loss_image_grad_steps']), 2)
        self.assertEqual(stats['image_grad_valid_pixels'], [9, 9])
        self.assertAlmostEqual(stats['weighted_loss_image_grad_steps'][0], stats['loss_image_grad_steps'][0] * 2.)


if __name__ == '__main__':
    unittest.main()
