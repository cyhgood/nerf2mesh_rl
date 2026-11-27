import ast
import pathlib
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from nerf.silhouette_loss import stage1_silhouette_loss
from nerf.refinement_experiment import _train_branch
from nerf.refinement_texture import stage1_image_gradient_loss

try:
    import nvdiffrast.torch as nvdiffrast
except ImportError:
    nvdiffrast = None

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_methods(path, class_name, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(body=methods, type_ignores=[])
    exec(compile(module, str(path), 'exec'), namespace)
    return [namespace[name] for name in names]


class SilhouetteTest(unittest.TestCase):
    def test_targets_and_finite_gradients(self):
        alpha = torch.tensor([.2, .8, .6, .1], requires_grad=True)
        images = torch.zeros(4, 4)
        images[:, 3] = torch.tensor([0., 1., 1., 0.])
        loss = stage1_silhouette_loss(alpha, images, {})
        self.assertAlmostEqual(loss.item(), .225, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(alpha.grad).all())
        self.assertTrue(torch.any(alpha.grad != 0))
        torch.testing.assert_close(stage1_silhouette_loss(alpha, images[:, :3],
                                                         {'foreground_mask': images[:, 3].reshape(2, 2)}), loss)
        with self.assertRaisesRegex(ValueError, 'requires RGBA'):
            stage1_silhouette_loss(alpha, images[:, :3], {})
        with self.assertRaisesRegex(ValueError, 'one value per'):
            stage1_silhouette_loss(alpha, images[:, :3], {'mask': torch.ones(1)})

    def test_train_step_and_branch(self):
        train_step, = load_methods('nerf/utils.py', 'Trainer', ['train_step'],
                                   {'torch': torch, 'stage1_silhouette_loss': stage1_silhouette_loss})
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vertices_offsets = torch.nn.Parameter(torch.tensor([.25]))
                self.rgb_errors = None
            def render_stage1(self, rays_o, rays_d, mvp, H, W, **kwargs):
                return {'image': torch.zeros(H * W, 3), 'alpha': self.vertices_offsets.expand(H * W)}
            def update_triangles_errors(self, loss):
                self.rgb_errors = loss
        model = Model()
        opt = SimpleNamespace(stage=1, color_space='srgb', background='white', refine=True,
                              lambda_lpips=0, lambda_lap=0, lambda_normal=0, lambda_edgelen=0,
                              lambda_offsets=0, lambda_sil=0, lambda_tv=0)
        trainer = SimpleNamespace(opt=opt, model=model, device=torch.device('cpu'), global_step=0,
                                  criterion=torch.nn.MSELoss(reduction='none'))
        trainer.train_step = lambda data: train_step(trainer, data)
        data = {'rays_o': torch.zeros(4, 3), 'rays_d': torch.zeros(4, 3), 'index': [0],
                'mvp': torch.eye(4).unsqueeze(0), 'H': 2, 'W': 2, 'images': torch.zeros(4, 3)}
        _, _, base_loss = trainer.train_step(data)
        self.assertIsNone(trainer.loss_sil)
        self.assertEqual(base_loss.item(), 0.)
        opt.lambda_sil = 2.
        with self.assertRaisesRegex(ValueError, 'requires RGBA'):
            trainer.train_step(data)
        data['mask'] = torch.ones(2, 2)
        _, _, loss = trainer.train_step(data)
        self.assertAlmostEqual(loss.item(), 1.5)
        self.assertAlmostEqual(trainer.loss_sil.item(), .75)
        self.assertEqual(model.rgb_errors.sum().item(), 0.)
        loss.backward()
        self.assertAlmostEqual(model.vertices_offsets.grad.item(), -2.)
        model.zero_grad()
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            _, _, amp_loss = trainer.train_step(data)
        amp_loss.backward()
        self.assertTrue(torch.isfinite(model.vertices_offsets.grad).all())
        self.assertAlmostEqual(model.vertices_offsets.grad.item(), -2.)
        trainer.scaler = torch.cuda.amp.GradScaler(enabled=False)
        trainer.optimizer_fn = lambda m: torch.optim.SGD(m.parameters(), lr=.1)
        trainer.lr_scheduler_fn = lambda opt: torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1)
        trainer.post_train_step = lambda: None
        dataset = SimpleNamespace(collate=lambda indices: data)
        metrics = _train_branch(trainer, dataset, [0, 0], 0)
        self.assertEqual(len(metrics['loss_sil_steps']), 2)
        self.assertLess(metrics['loss_sil_steps'][1], metrics['loss_sil_steps'][0])

    def test_renderer_output_downsample_and_gradient(self):
        # CPU stub exercises the real renderer's tensor path. It cannot validate nvdiffrast itself.
        class Raster:
            def rasterize(self, ctx, vertices, triangles, shape):
                return torch.ones(1, *shape, 4), None
            def interpolate(self, attr, rast, triangles):
                return torch.ones(1, *rast.shape[1:3], attr.shape[-1]), None
            def antialias(self, value, rast, vertices, triangles, **kwargs):
                return value * .4 + vertices[:, :1, :1].reshape(1, 1, 1, 1) * .1
        def scale_img_hwc(value, size, mag='bilinear', min='bilinear'):
            return F.interpolate(value.permute(2, 0, 1)[None], size=size, mode=mag if size[0] >= value.shape[0] else min)[0].permute(1, 2, 0)
        render, = load_methods('nerf/renderer.py', 'NeRFRenderer', ['render_stage1'],
                               {'torch': torch, 'F': F, 'dr': Raster(), 'scale_img_hwc': scale_img_hwc,
                                'scale_img_hw': lambda value, size, **kwargs: F.interpolate(value[None, None], size=size, mode='nearest')[0, 0]})
        for ssaa in (1, 2):
            with self.subTest(ssaa=ssaa):
                offsets = torch.nn.Parameter(torch.zeros(3, 3))
                model = SimpleNamespace(opt=SimpleNamespace(ssaa=ssaa, fp16=False, pos_gradient_boost=1,
                                        enable_offset_nerf_grad=False, refine_multiround=True),
                                        vertices=torch.ones(3, 3), vertices_offsets=offsets,
                                        triangles=torch.tensor([[0, 1, 2]], dtype=torch.int32),
                                        glctx=None, individual_dim=0,
                                        rgb=lambda xyz, dirs, code, shading: (torch.ones_like(xyz), None))
                rays = torch.ones(4, 3)
                camera = torch.eye(4)
                camera[2, 3] = 3
                outputs = render(model, rays, rays, torch.eye(4), 2, 2,
                                 depth_definition='camera_z', camera_pose=camera)
                self.assertEqual(outputs['alpha'].shape, (4,))
                torch.testing.assert_close(outputs['surface_depth'], torch.full((4,), 2.))
                self.assertTrue(outputs['surface_depth_hit'].all())
                ray_outputs = render(model, rays, rays, torch.eye(4), 2, 2,
                                     depth_definition='ray_distance', camera_pose=camera)
                torch.testing.assert_close(ray_outputs['surface_depth'], torch.full((4,), 6 ** .5))
                self.assertFalse(model.refinement_alpha.requires_grad)
                loss = stage1_silhouette_loss(outputs['alpha'], torch.zeros(4, 4), {})
                loss.backward()
                self.assertTrue(torch.isfinite(offsets.grad).all())
                self.assertGreater(offsets.grad.abs().sum().item(), 0.)

    @unittest.skipUnless(torch.cuda.is_available() and nvdiffrast is not None,
                         'CUDA and nvdiffrast required')
    def test_real_antialias_cuda_gradient(self):
        def scale_img_hwc(value, size, mag='bilinear', min='bilinear'):
            return F.interpolate(value.permute(2, 0, 1)[None], size=size,
                                 mode=mag if size[0] >= value.shape[0] else min)[0].permute(1, 2, 0)
        render, = load_methods('nerf/renderer.py', 'NeRFRenderer', ['render_stage1'],
                               {'torch': torch, 'F': F, 'dr': nvdiffrast, 'scale_img_hwc': scale_img_hwc,
                                'scale_img_hw': lambda value, size, **kwargs: F.interpolate(value[None, None], size=size, mode='nearest')[0, 0]})
        context = nvdiffrast.RasterizeCudaContext()
        vertices = torch.tensor([[-.72, -.61, 0.], [.67, -.53, 0.], [0., .73, 0.]], device='cuda')
        for ssaa in (1, 2):
            for fp16 in (False, True):
                with self.subTest(ssaa=ssaa, fp16=fp16):
                    offsets = torch.nn.Parameter(torch.zeros_like(vertices))
                    model = SimpleNamespace(opt=SimpleNamespace(ssaa=ssaa, fp16=fp16, pos_gradient_boost=1,
                                            enable_offset_nerf_grad=False, refine_multiround=True),
                                            vertices=vertices, vertices_offsets=offsets,
                                            triangles=torch.tensor([[0, 1, 2]], dtype=torch.int32, device='cuda'),
                                            glctx=context, individual_dim=0,
                                            rgb=lambda xyz, dirs, code, shading: (torch.ones_like(xyz), None))
                    rays = torch.ones(16 * 16, 3, device='cuda')
                    camera = torch.eye(4, device='cuda')
                    camera[2, 3] = 2.
                    outputs = render(model, rays, rays, torch.eye(4, device='cuda'), 16, 16,
                                     depth_definition='camera_z', camera_pose=camera)
                    self.assertEqual(outputs['alpha'].shape, (256,))
                    self.assertFalse(model.refinement_alpha.requires_grad)
                    self.assertTrue(outputs['surface_depth_hit'].any())
                    torch.testing.assert_close(outputs['surface_depth'][outputs['surface_depth_hit']],
                                               torch.full_like(outputs['surface_depth'][outputs['surface_depth_hit']], 2.))
                    depth_grad = torch.autograd.grad(outputs['surface_depth'][outputs['surface_depth_hit']].mean(),
                                                     offsets, retain_graph=True)[0]
                    self.assertTrue(torch.isfinite(depth_grad).all())
                    self.assertGreater(depth_grad.abs().sum().item(), 0.)
                    target = torch.zeros(16, 16, 3, device='cuda')
                    target[:, 8:] = 1.
                    image_loss, valid_count = stage1_image_gradient_loss(outputs['image'].reshape(16, 16, 3),
                                                                          target, torch.ones(16, 16, device='cuda'))
                    self.assertEqual(valid_count, 14 * 14)
                    image_grad = torch.autograd.grad(image_loss, offsets, retain_graph=True)[0]
                    self.assertTrue(torch.isfinite(image_grad).all())
                    loss = stage1_silhouette_loss(outputs['alpha'], torch.zeros(256, 4, device='cuda'), {})
                    loss.backward()
                    self.assertTrue(torch.isfinite(offsets.grad).all())
                    self.assertGreater(offsets.grad.abs().sum().item(), 0.)


if __name__ == '__main__':
    unittest.main()
