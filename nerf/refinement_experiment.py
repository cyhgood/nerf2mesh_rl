"""Paired, fixed-view measurement of a single Stage 1 refinement event."""

import copy
import csv
import json
import os
import random

import numpy as np
import torch
from torch import nn


def _observe(trainer, dataset, view_ids):
    observations = []
    trainer.model.eval()
    with torch.no_grad():
        for view_id in view_ids:
            data = dataset.collate([view_id])
            prediction, _, target, _ = trainer.eval_step(data)
            loss = trainer.criterion(prediction, target).mean(-1)
            faces = trainer.model.triangles_errors_id.long()
            observations.append((faces.cpu().numpy().reshape(-1).copy(),
                                 loss.cpu().numpy().reshape(-1).copy()))
    return observations


def _fixed_region_means(before, after, face_count):
    counts = np.zeros(face_count, dtype=np.int64)
    sums = np.zeros(face_count, dtype=np.float64)
    for (old_faces, _), (_, losses) in zip(before, after):
        valid = old_faces >= 0
        counts += np.bincount(old_faces[valid], minlength=face_count)
        sums += np.bincount(old_faces[valid], weights=losses[valid], minlength=face_count)
    means = np.full(face_count, np.nan)
    np.divide(sums, counts, out=means, where=counts > 0)
    return counts, means


def _child_observations(observations, face_count):
    counts = np.zeros(face_count, dtype=np.int64)
    sums = np.zeros(face_count, dtype=np.float64)
    for faces, losses in observations:
        valid = faces >= 0
        counts += np.bincount(faces[valid], minlength=face_count)
        sums += np.bincount(faces[valid], weights=losses[valid], minlength=face_count)
    means = np.full(face_count, np.nan)
    np.divide(sums, counts, out=means, where=counts > 0)
    return counts, means


def _train_branch(trainer, dataset, indices, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    trainer.model.train()
    trainer.optimizer = trainer.optimizer_fn(trainer.model)
    trainer.lr_scheduler = trainer.lr_scheduler_fn(trainer.optimizer)
    for index in indices:
        trainer.global_step += 1
        trainer.optimizer.zero_grad()
        _, _, loss = trainer.train_step(dataset.collate([int(index)]))
        trainer.scaler.scale(loss).backward()
        trainer.post_train_step()
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()
        trainer.lr_scheduler.step()


def run_refinement_experiment(trainer, train_data, val_data):
    opt = trainer.opt
    if (opt.stage != 1 or opt.refine_mode != 'local' or not opt.refine
            or opt.bound > 1 or opt.ind_dim != 0 or opt.gui):
        raise ValueError('Experiment requires Stage 1, --refine, --refine_mode local, '
                         'bound <= 1, ind_dim=0 and non-GUI mode')
    if opt.refine_experiment_steps < 1 or opt.refine_experiment_views < 1:
        raise ValueError('Experiment steps and view count must be positive')
    if not opt.refine_experiment_output:
        raise ValueError('Set --refine_experiment_output to a new directory')
    if not os.path.isfile(opt.ckpt) or not opt.mesh or not os.path.isfile(opt.mesh):
        raise ValueError('Provide an explicit Stage 1 --ckpt and its matching --mesh')
    checkpoint = torch.load(opt.ckpt, map_location='cpu')
    if checkpoint.get('stage') != 1 or 'model' not in checkpoint:
        raise ValueError('The experiment requires a Stage 1 training checkpoint')
    current_state = trainer.model.state_dict()
    if ('vertices_offsets' not in checkpoint['model'] or
            any(name not in current_state or not torch.equal(value.cpu(), current_state[name].cpu())
                for name, value in checkpoint['model'].items())):
        raise ValueError('Stage 1 checkpoint was not loaded completely into the current model')
    if not np.isfinite(trainer.model.vertices.detach().cpu().numpy()).all():
        raise ValueError('Input mesh contains non-finite vertex positions')
    if val_data.images is None or train_data.images is None:
        raise ValueError('Training and validation views must have ground truth images')
    os.makedirs(opt.refine_experiment_output, exist_ok=False)

    view_ids = list(range(min(opt.refine_experiment_views, len(val_data.poses))))
    if not view_ids or not len(train_data.poses):
        raise ValueError('Training and validation datasets must contain views')
    face_count = len(trainer.model.triangles)
    before = _observe(trainer, val_data, view_ids)
    old_counts, before_mean = _fixed_region_means(before, before, face_count)

    # Select candidates using the same per-pixel RGB loss as Stage 1.
    trainer.model.triangles_errors.zero_()
    trainer.model.triangles_errors_cnt.zero_()
    for faces, losses in before:
        valid = faces >= 0
        sums = np.bincount(faces[valid], weights=losses[valid], minlength=face_count)
        counts = np.bincount(faces[valid], minlength=face_count)
        trainer.model.triangles_errors += torch.as_tensor(sums, device=trainer.device, dtype=torch.float32)
        trainer.model.triangles_errors_cnt += torch.as_tensor(counts, device=trainer.device, dtype=torch.float32)

    model_state = copy.deepcopy(trainer.model.state_dict())
    old_vertices = trainer.model.vertices.clone()
    old_triangles = trainer.model.triangles.clone()
    old_v_cumsum = trainer.model.v_cumsum.copy()
    old_f_cumsum = trainer.model.f_cumsum.copy()
    scaler_state = copy.deepcopy(trainer.scaler.state_dict())
    start_step = trainer.global_step
    indices = np.random.RandomState(opt.seed).choice(len(train_data.poses), opt.refine_experiment_steps).tolist()

    trainer.model.refine_and_decimate(export_mesh=False)
    parent = trainer.model.parent_face_id.cpu().numpy().copy()
    event = copy.deepcopy(trainer.model.refinement_event)
    event.update({'global_step': start_step, 'parent_face_id': parent.tolist(),
                  'validation_views': view_ids, 'training_indices': indices,
                  'steps': opt.refine_experiment_steps, 'seed': opt.seed,
                  'checkpoint': os.path.abspath(opt.ckpt), 'mesh': os.path.abspath(opt.mesh),
                  'selection_source': 'validation RGB SmoothL1',
                  'optimizer': 'both branches reset Adam and LambdaLR at the event'})
    opt.refine = False
    _train_branch(trainer, train_data, indices, opt.seed)
    refined = _observe(trainer, val_data, view_ids)

    trainer.model.vertices = old_vertices
    trainer.model.triangles = old_triangles
    trainer.model.vertices_offsets = nn.Parameter(torch.zeros_like(old_vertices))
    trainer.model.v_cumsum = old_v_cumsum
    trainer.model.f_cumsum = old_f_cumsum
    trainer.model.load_state_dict(model_state)
    # Match the refined branch, which folds the original offset into vertices.
    trainer.model.vertices = old_vertices + trainer.model.vertices_offsets.detach()
    trainer.model.vertices_offsets = nn.Parameter(torch.zeros_like(old_vertices))
    trainer.model.triangles_errors = torch.zeros(face_count, device=trainer.device)
    trainer.model.triangles_errors_cnt = torch.zeros(face_count, device=trainer.device)
    trainer.scaler.load_state_dict(scaler_state)
    trainer.global_step = start_step
    _train_branch(trainer, train_data, indices, opt.seed)
    control = _observe(trainer, val_data, view_ids)

    _, refined_mean = _fixed_region_means(before, refined, face_count)
    _, control_mean = _fixed_region_means(before, control, face_count)
    child_counts, child_means = _child_observations(refined, len(parent))
    children_per_face = np.bincount(parent, minlength=face_count)
    event['child_visibility'] = [
        {'child_face_id': i, 'parent_face_id': int(parent[i]), 'visible_pixels': int(child_counts[i]),
         'mean_error': float(child_means[i]) if child_counts[i] else None}
        for i in range(len(parent))
    ]
    with open(os.path.join(opt.refine_experiment_output, 'event.json'), 'x') as output:
        json.dump(event, output, indent=2, allow_nan=False)
    with open(os.path.join(opt.refine_experiment_output, 'faces.csv'), 'x', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['old_face_id', 'action_status', 'passively_split', 'child_count', 'evaluation_pixels',
                         'before_error', 'refined_error', 'control_error',
                         'refined_improvement', 'control_improvement', 'improvement_difference'])
        passive = set(event['passive_faces'])
        for face in range(face_count):
            status = event['request_status'].get(face, 'passive' if face in passive else 'hold')
            if not old_counts[face]:
                writer.writerow([face, status, face in passive, children_per_face[face], 0, '', '', '', '', '', ''])
                continue
            refined_gain = before_mean[face] - refined_mean[face]
            control_gain = before_mean[face] - control_mean[face]
            writer.writerow([face, status, face in passive, children_per_face[face], old_counts[face],
                             before_mean[face], refined_mean[face], control_mean[face],
                             refined_gain, control_gain, refined_gain - control_gain])
    trainer.log('[INFO] refinement experiment written to ' + opt.refine_experiment_output)
