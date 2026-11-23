"""Paired, fixed-view measurement of a single Stage 1 refinement event."""

import copy
import csv
import json
import os
import random

import numpy as np
import torch
from torch import nn


def _select_views(total, count, selection=None, evaluation=None, allow_overlap=False):
    if count < 1:
        raise ValueError('View count must be positive')
    if selection is None and evaluation is None:
        selection = list(range(count))
        evaluation = list(range(count, 2 * count))
    elif selection is None or evaluation is None:
        raise ValueError('Specify both selection and evaluation views')
    else:
        selection, evaluation = list(selection), list(evaluation)
    for views in (selection, evaluation):
        if not views or len(views) != len(set(views)) or any(i < 0 or i >= total for i in views):
            raise ValueError('View indices must be nonempty, unique, and within the validation dataset')
    if not allow_overlap and set(selection) & set(evaluation):
        raise ValueError('Selection and evaluation views overlap')
    return selection, evaluation


def _observe(trainer, dataset, view_ids):
    observations = []
    trainer.model.eval()
    with torch.no_grad():
        for view_id in view_ids:
            data = dataset.collate([view_id])
            images = data['images'] if isinstance(data, dict) else None
            foreground = None
            if images is not None and images.shape[-1] == 4:
                foreground = (images[..., 3] > 0).cpu().numpy().reshape(-1).copy()
            prediction, _, target, _ = trainer.eval_step(data)
            loss = trainer.criterion(prediction, target).mean(-1)
            faces = trainer.model.triangles_errors_id.long()
            observations.append((faces.cpu().numpy().reshape(-1).copy(),
                                 loss.cpu().numpy().reshape(-1).copy(), foreground))
    return observations


def _fixed_region_means(before, after, face_count):
    counts = np.zeros(face_count, dtype=np.int64)
    sums = np.zeros(face_count, dtype=np.float64)
    for old, new in zip(before, after):
        old_faces, losses = old[0], new[1]
        valid = old_faces >= 0
        counts += np.bincount(old_faces[valid], minlength=face_count)
        sums += np.bincount(old_faces[valid], weights=losses[valid], minlength=face_count)
    means = np.full(face_count, np.nan)
    np.divide(sums, counts, out=means, where=counts > 0)
    return counts, means


def _child_observations(observations, face_count):
    counts = np.zeros(face_count, dtype=np.int64)
    sums = np.zeros(face_count, dtype=np.float64)
    for observation in observations:
        faces, losses = observation[:2]
        valid = faces >= 0
        counts += np.bincount(faces[valid], minlength=face_count)
        sums += np.bincount(faces[valid], weights=losses[valid], minlength=face_count)
    means = np.full(face_count, np.nan)
    np.divide(sums, counts, out=means, where=counts > 0)
    return counts, means


def _foreground_errors(before, refined, control, view_ids):
    rows = []
    sums = np.zeros(3, dtype=np.float64)
    total = 0
    for view, initial, new, baseline in zip(view_ids, before, refined, control):
        mask = initial[2] if initial[2] is not None else initial[0] >= 0
        count = int(mask.sum())
        view_sums = np.array([np.sum(obs[1][mask], dtype=np.float64)
                              for obs in (initial, new, baseline)])
        sums += view_sums
        total += count
        means = (view_sums / count).tolist() if count else [None] * 3
        rows.append({'view_id': view, 'foreground_pixels': count,
                     'before_error': means[0], 'refined_error': means[1], 'control_error': means[2]})
    means = (sums / total).tolist() if total else [None] * 3
    return rows, {'foreground_pixels': total, 'before_error': means[0],
                  'refined_error': means[1], 'control_error': means[2]}


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


def _matched_budget_plans(vertices, faces, candidate_ids, errors, budget, seed):
    from meshutils import subdivide_longest_edges

    if budget < 1:
        raise ValueError('Face budget must be positive')
    candidate_ids = np.asarray(candidate_ids, dtype=np.int64)
    order_error = candidate_ids[np.argsort(-errors[candidate_ids], kind='stable')]
    order_random = np.random.RandomState(seed).permutation(candidate_ids)
    candidates = np.zeros(len(faces), dtype=bool)
    candidates[candidate_ids] = True
    plans = []
    for order in (order_error, order_random):
        _, _, _, event = subdivide_longest_edges(
            vertices, faces, candidates, return_event=True,
            candidate_order=order, max_added_faces=budget)
        plans.append(event)
    achievable = [set([0] + event['cumulative_added_faces']) for event in plans]
    matched = max(achievable[0] & achievable[1])
    return matched, order_error, order_random


def _run_random_comparison(trainer, train_data, val_data, before, selection_views,
                           evaluation_views, indices, model_state, old_vertices,
                           old_triangles, old_v_cumsum, old_f_cumsum, scaler_state, start_step):
    opt = trainer.opt
    budget = opt.refine_face_budget
    seeds = opt.refine_random_seeds
    if budget < 1 or not seeds or len(seeds) != len(set(seeds)):
        raise ValueError('Provide a positive face budget and unique random seeds')
    if any(seed < 0 or seed > 2**32 - 1 for seed in seeds):
        raise ValueError('Random seeds must be in [0, 2**32 - 1]')

    model = trainer.model
    face_count = len(old_triangles)
    selection_counts = model.triangles_errors_cnt.cpu().numpy().copy()
    selection_sums = model.triangles_errors.cpu().numpy().copy()
    errors = np.zeros(face_count, dtype=np.float64)
    np.divide(selection_sums, selection_counts, out=errors, where=selection_counts > 0)
    candidates = np.flatnonzero(selection_counts > 0)
    vertices = (old_vertices + model.vertices_offsets.detach()).cpu().numpy()
    faces = old_triangles.cpu().numpy()

    def restore():
        model.vertices = old_vertices.clone()
        model.triangles = old_triangles.clone()
        model.vertices_offsets = nn.Parameter(torch.zeros_like(old_vertices))
        model.v_cumsum = old_v_cumsum.copy()
        model.f_cumsum = old_f_cumsum.copy()
        model.load_state_dict(model_state)
        model.triangles_errors = torch.as_tensor(selection_sums.copy(), device=trainer.device)
        model.triangles_errors_cnt = torch.as_tensor(selection_counts.copy(), device=trainer.device)
        trainer.scaler.load_state_dict(scaler_state)
        trainer.global_step = start_step

    restore()
    model.vertices = model.vertices + model.vertices_offsets.detach()
    model.vertices_offsets = nn.Parameter(torch.zeros_like(model.vertices))
    opt.refine = False
    _train_branch(trainer, train_data, indices, opt.seed)
    control = _observe(trainer, val_data, evaluation_views)

    result = {'checkpoint': os.path.abspath(opt.ckpt), 'mesh': os.path.abspath(opt.mesh),
              'global_step': start_step, 'selection_views': selection_views,
              'evaluation_views': evaluation_views, 'training_indices': indices,
              'training_seed': opt.seed, 'steps': opt.refine_experiment_steps,
              'target_added_faces': budget, 'candidate_pool': 'faces visible in selection views',
              'budget_rule': 'largest common greedy prefix under target; same actual added faces',
              'foreground_mask': 'ground-truth alpha > 0, or baseline mesh coverage for RGB images',
              'error_metric': 'RGB mean SmoothL1', 'runs': []}
    for seed in seeds:
        matched, high_order, random_order = _matched_budget_plans(vertices, faces, candidates, errors, budget, seed)
        run = {'random_seed': seed, 'matched_added_faces': matched, 'branches': {}}
        for name, order in (('high_error', high_order), ('random', random_order)):
            restore()
            model.refine_and_decimate(export_mesh=False, candidate_order=order, max_added_faces=matched)
            event = copy.deepcopy(model.refinement_event)
            event['global_step'] = start_step
            event['parent_face_id'] = model.parent_face_id.cpu().numpy().tolist()
            actual = len(model.triangles) - face_count
            if actual != matched or actual != event['added_faces']:
                raise RuntimeError('Refinement did not produce the matched face budget')
            _train_branch(trainer, train_data, indices, opt.seed)
            after = _observe(trainer, val_data, evaluation_views)
            view_errors, overall = _foreground_errors(before, after, control, evaluation_views)
            run['branches'][name] = {'actual_added_faces': actual, 'event': event,
                                      'foreground_errors_by_view': view_errors,
                                      'foreground_errors_overall': overall}
        result['runs'].append(run)
        trainer.log('[INFO] random seed {}: target {} added faces, matched {}'.format(seed, budget, matched))

    with open(os.path.join(opt.refine_experiment_output, 'comparison.json'), 'x') as output:
        json.dump(result, output, indent=2, allow_nan=False)
    trainer.log('[INFO] refinement comparison written to ' + opt.refine_experiment_output)


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
    selection_views, evaluation_views = _select_views(
        len(val_data.poses), opt.refine_experiment_views,
        getattr(opt, 'refine_selection_views', None), getattr(opt, 'refine_evaluation_views', None),
        getattr(opt, 'refine_allow_view_overlap', False))
    if not len(train_data.poses):
        raise ValueError('Training and validation datasets must contain views')
    os.makedirs(opt.refine_experiment_output, exist_ok=False)
    face_count = len(trainer.model.triangles)
    selection_observations = _observe(trainer, val_data, selection_views)
    before = _observe(trainer, val_data, evaluation_views)
    old_counts, before_mean = _fixed_region_means(before, before, face_count)

    # Select candidates using the same per-pixel RGB loss as Stage 1.
    trainer.model.triangles_errors.zero_()
    trainer.model.triangles_errors_cnt.zero_()
    for faces, losses, _ in selection_observations:
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

    if getattr(opt, 'refine_multiround', False):
        from nerf.refinement_multiround import run_multiround
        run_multiround(trainer, train_data, val_data, selection_views, evaluation_views)
        return

    if getattr(opt, 'refine_rl', False):
        from nerf.refinement_rl import run_refinement_rl
        run_refinement_rl(trainer, train_data, val_data, selection_views, evaluation_views,
                          indices, model_state, old_vertices, old_triangles,
                          old_v_cumsum, old_f_cumsum, scaler_state, start_step)
        return

    if getattr(opt, 'refine_compare_random', False):
        _run_random_comparison(trainer, train_data, val_data, before, selection_views,
                               evaluation_views, indices, model_state, old_vertices,
                               old_triangles, old_v_cumsum, old_f_cumsum, scaler_state, start_step)
        return

    trainer.model.refine_and_decimate(export_mesh=False)
    parent = trainer.model.parent_face_id.cpu().numpy().copy()
    event = copy.deepcopy(trainer.model.refinement_event)
    event.update({'global_step': start_step, 'parent_face_id': parent.tolist(),
                  'validation_views': evaluation_views, 'selection_views': selection_views,
                  'evaluation_views': evaluation_views, 'training_indices': indices,
                  'steps': opt.refine_experiment_steps, 'seed': opt.seed,
                  'checkpoint': os.path.abspath(opt.ckpt), 'mesh': os.path.abspath(opt.mesh),
                  'selection_source': 'validation RGB SmoothL1',
                  'optimizer': 'both branches reset Adam and LambdaLR at the event'})
    opt.refine = False
    _train_branch(trainer, train_data, indices, opt.seed)
    refined = _observe(trainer, val_data, evaluation_views)

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
    control = _observe(trainer, val_data, evaluation_views)

    _, refined_mean = _fixed_region_means(before, refined, face_count)
    _, control_mean = _fixed_region_means(before, control, face_count)
    child_counts, child_means = _child_observations(refined, len(parent))
    children_per_face = np.bincount(parent, minlength=face_count)
    view_errors, overall_errors = _foreground_errors(before, refined, control, evaluation_views)
    event['foreground_mask'] = 'ground-truth alpha > 0, or baseline mesh coverage for RGB images'
    event['foreground_errors_by_view'] = view_errors
    event['foreground_errors_overall'] = overall_errors
    if old_counts.any():
        largest = int(np.argmax(old_counts))
        event['largest_evaluation_face'] = {
            'old_face_id': largest, 'evaluation_pixels': int(old_counts[largest]),
            'before_error': float(before_mean[largest]), 'refined_error': float(refined_mean[largest]),
            'control_error': float(control_mean[largest])}
        trainer.log('[INFO] largest evaluation face: ' + str(event['largest_evaluation_face']))
    else:
        event['largest_evaluation_face'] = None
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
