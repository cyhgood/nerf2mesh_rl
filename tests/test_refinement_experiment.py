import unittest
import csv
import json
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

from meshutils import subdivide_longest_edges
from nerf.refinement_experiment import (
    _child_observations, _fixed_region_means, _foreground_errors, _select_views,
    run_refinement_experiment,
)


class RefinementExperimentTest(unittest.TestCase):
    def test_view_selection_and_overlap(self):
        self.assertEqual(_select_views(10, 2), ([0, 1], [2, 3]))
        self.assertEqual(_select_views(5, 2, [3, 1], [4, 0]), ([3, 1], [4, 0]))
        for args in ((3, 2, None, None), (5, 2, [1], [1]),
                     (5, 2, [1, 1], [2]), (5, 2, [5], [2]),
                     (5, 2, [1], None)):
            with self.assertRaises(ValueError):
                _select_views(*args)
        self.assertEqual(_select_views(2, 1, [1], [1], True), ([1], [1]))

    def test_foreground_weighted_aggregate(self):
        before = [(np.array([0, -1, 0]), np.array([1., 3., 9.]), np.array([True, True, False])),
                  (np.array([1, -1]), np.array([8., 4.]), np.array([False, True]))]
        refined = [(np.array([-1, -1, 0]), np.array([2., 4., 0.]), None),
                   (np.array([1, -1]), np.array([0., 1.]), None)]
        control = [(np.array([0, 0, 0]), np.array([3., 5., 0.]), None),
                   (np.array([1, -1]), np.array([0., 2.]), None)]
        rows, overall = _foreground_errors(before, refined, control, [4, 5])
        self.assertEqual(rows[0]['foreground_pixels'], 2)
        self.assertEqual(rows[0]['before_error'], 2.)
        self.assertEqual(overall['foreground_pixels'], 3)
        self.assertAlmostEqual(overall['before_error'], 8. / 3)
        self.assertAlmostEqual(overall['refined_error'], 7. / 3)
        self.assertAlmostEqual(overall['control_error'], 10. / 3)

    def test_fixed_regions_include_pixels_that_leave_the_mesh(self):
        before = [(np.array([0, 0, 1, -1]), np.array([2., 4., 3., 9.]))]
        after = [(np.array([-1, 2, 3, 0]), np.array([10., 1., 2., 0.]))]
        counts, means = _fixed_region_means(before, after, 3)
        np.testing.assert_array_equal(counts, [2, 1, 0])
        np.testing.assert_allclose(means[:2], [5.5, 2.])
        self.assertTrue(np.isnan(means[2]))
        counts, means = _child_observations(after, 4)
        np.testing.assert_array_equal(counts, [1, 0, 1, 1])
        self.assertTrue(np.isnan(means[1]))

    def test_paired_experiment(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]])
                self.triangles = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int32)
                self.vertices_offsets = torch.nn.Parameter(torch.full((4, 3), 0.1))
                self.color = torch.nn.Parameter(torch.tensor(0.5))
                self.v_cumsum = np.array([0, 4])
                self.f_cumsum = np.array([0, 2])
                self.triangles_errors = torch.zeros(2)
                self.triangles_errors_cnt = torch.zeros(2)

            def refine_and_decimate(self, export_mesh):
                v, f, parent, self.refinement_event = subdivide_longest_edges(
                    (self.vertices + self.vertices_offsets).detach().numpy(),
                    self.triangles.numpy(), [True, False], return_event=True)
                self.vertices = torch.tensor(v)
                self.triangles = torch.tensor(f)
                self.vertices_offsets = torch.nn.Parameter(torch.zeros_like(self.vertices))
                self.parent_face_id = torch.tensor(parent)

        class Dataset:
            poses = [0, 1]
            images = True

            def collate(self, index):
                return index[0]

        with tempfile.TemporaryDirectory() as directory:
            model = Model()
            ckpt = os.path.join(directory, 'model.pth')
            mesh = os.path.join(directory, 'mesh.ply')
            torch.save({'stage': 1, 'model': model.state_dict()}, ckpt)
            open(mesh, 'w').close()
            output = os.path.join(directory, 'output')
            opt = SimpleNamespace(stage=1, refine_mode='local', refine=True, bound=1,
                                  ind_dim=0, gui=False, refine_experiment_steps=2,
                                  refine_experiment_views=2, refine_experiment_output=output,
                                  refine_selection_views=[0], refine_evaluation_views=[1],
                                  ckpt=ckpt, mesh=mesh, seed=42)
            samples = []
            observed_views = []
            trainer = SimpleNamespace(opt=opt, model=model, device=torch.device('cpu'),
                                      global_step=7, scaler=torch.cuda.amp.GradScaler(enabled=False),
                                      criterion=torch.nn.SmoothL1Loss(reduction='none'))
            trainer.optimizer_fn = lambda m: torch.optim.Adam(m.parameters(), lr=0.01)
            trainer.lr_scheduler_fn = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1)
            trainer.post_train_step = lambda: None
            trainer.log = lambda message: None

            def eval_step(data):
                observed_views.append(data)
                count = len(model.triangles)
                model.triangles_errors_id = torch.tensor([0, 1 if count == 2 else 2, -1])
                prediction = torch.full((3, 3), float(model.color.detach()))
                return prediction, None, torch.zeros_like(prediction), None

            def train_step(data):
                samples.append(data)
                return None, None, model.color.square() + model.vertices_offsets.square().sum()

            trainer.eval_step = eval_step
            trainer.train_step = train_step
            run_refinement_experiment(trainer, Dataset(), Dataset())
            self.assertEqual(samples[:2], samples[2:])
            self.assertEqual(observed_views, [0, 1, 1, 1])
            with open(os.path.join(output, 'event.json')) as file:
                event = json.load(file)
            self.assertEqual(event['parent_face_id'], [0, 0, 1, 1])
            self.assertEqual(event['passive_faces'], [1])
            self.assertEqual(event['selection_views'], [0])
            self.assertEqual(event['evaluation_views'], [1])
            self.assertEqual(event['largest_evaluation_face']['evaluation_pixels'], 1)
            self.assertEqual(event['foreground_errors_overall']['foreground_pixels'], 2)
            with open(os.path.join(output, 'faces.csv')) as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]['action_status'], 'executed')
            self.assertEqual(rows[1]['passively_split'], 'True')
            self.assertEqual(rows[0]['refined_error'], rows[0]['control_error'])


if __name__ == '__main__':
    unittest.main()
