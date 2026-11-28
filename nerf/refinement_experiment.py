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


def _observe(trainer, dataset, view_ids, texture=False):
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
            observation = (faces.cpu().numpy().reshape(-1).copy(),
                           loss.cpu().numpy().reshape(-1).copy(), foreground)
            if texture:
                if images is None or images.shape[-1] != 4:
                    raise ValueError('Texture observation requires ground-truth RGBA images')
                from nerf.refinement_texture import sobel_observation
                observation += sobel_observation(prediction, target, images[..., 3], faces)
            observations.append(observation)
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


def _train_branch(trainer, dataset, indices, seed, vertex_state=None, ema_beta=0.9, gradient_diagnostics=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    trainer.model.train()
    trainer.optimizer = trainer.optimizer_fn(trainer.model)
    trainer.lr_scheduler = trainer.lr_scheduler_fn(trainer.optimizer)
    silhouette_losses = []
    depth_losses = []
    depth_valid_pixels = []
    image_gradient_losses = []
    weighted_image_gradient_losses = []
    image_gradient_valid_pixels = []
    texture_weighted_losses = []
    scaled_texture_weighted_losses = []
    texture_weight_stats = []
    gradient_samples = []
    for step, index in enumerate(indices):
        trainer.global_step += 1
        trainer.optimizer.zero_grad()
        sample_gradients = (gradient_diagnostics is not None and
                            step % gradient_diagnostics['interval'] == 0 and
                            len(gradient_samples) < gradient_diagnostics['max_samples'])
        if gradient_diagnostics is not None:
            trainer._gradient_diagnostic_requested = sample_gradients
        try:
            _, _, loss = trainer.train_step(dataset.collate([int(index)]))
        finally:
            if gradient_diagnostics is not None:
                trainer._gradient_diagnostic_requested = False
        if sample_gradients:
            from nerf.refinement_loss_gradient_diagnostics import analyze_loss_gradients
            components = getattr(trainer, '_gradient_diagnostic_losses', None)
            rgb, gradient, pixels, reason = components if components is not None else (None, None, None, 'training_loss_components_unavailable')
            groups = analyze_loss_gradients(trainer.model, trainer.optimizer, rgb, gradient, pixels,
                                            getattr(trainer.opt, 'lambda_image_grad', 0.),
                                            gradient_unavailable_reason=reason, amp_scale=trainer.scaler.get_scale())
            gradient_samples.append({'step_in_round': step + 1, 'global_step': trainer.global_step,
                                     'training_index': int(index), 'valid_gradient_pixels': pixels,
                                     'amp_scale': trainer.scaler.get_scale(),
                                     'rgb_loss': float(rgb.detach()) if rgb is not None and torch.isfinite(rgb.detach()).all() else None,
                                     'image_gradient_loss': float(gradient.detach()) if gradient is not None and torch.isfinite(gradient.detach()).all() else None,
                                     'groups': groups})
            trainer._gradient_diagnostic_losses = None
        if getattr(trainer, 'loss_sil', None) is not None:
            silhouette_losses.append(float(trainer.loss_sil.item()))
        if getattr(trainer, 'loss_stage1_depth', None) is not None:
            depth_losses.append(float(trainer.loss_stage1_depth.item()))
            depth_valid_pixels.append(trainer.stage1_depth_valid_count)
        if getattr(trainer, 'loss_image_grad', None) is not None:
            image_gradient_losses.append(float(trainer.loss_image_grad.item()))
            weighted_image_gradient_losses.append(float(trainer.weighted_loss_image_grad.item()))
            image_gradient_valid_pixels.append(trainer.image_grad_valid_pixels)
        if getattr(trainer, 'loss_texture_weighted_grad', None) is not None:
            texture_weighted_losses.append(float(trainer.loss_texture_weighted_grad.item()))
            scaled_texture_weighted_losses.append(float(trainer.scaled_loss_texture_weighted_grad.item()))
            texture_weight_stats.append(trainer.texture_gradient_weight_stats)
        trainer.scaler.scale(loss).backward()
        trainer.post_train_step()
        norms = None
        step_finite = True
        old_scale = trainer.scaler.get_scale()
        if vertex_state is not None:
            trainer.scaler.unscale_(trainer.optimizer)
            gradient = trainer.model.vertices_offsets.grad
            if gradient is not None:
                norms = gradient.detach().norm(dim=-1).cpu().numpy().copy()
            if not trainer.scaler.is_enabled():
                step_finite = all(torch.isfinite(parameter.grad).all().item()
                                  for group in trainer.optimizer.param_groups for parameter in group['params']
                                  if parameter.grad is not None)
        trainer.scaler.step(trainer.optimizer)
        trainer.scaler.update()
        if vertex_state is not None:
            from nerf.refinement_candidates import update_vertex_ema
            update_vertex_ema(vertex_state, norms, ema_beta,
                              step_finite and trainer.scaler.get_scale() >= old_scale)
        trainer.lr_scheduler.step()
    result = {'loss_sil_steps': silhouette_losses,
            'loss_sil_mean': float(np.mean(silhouette_losses)) if silhouette_losses else None,
            'loss_depth_steps': depth_losses, 'depth_valid_pixels': depth_valid_pixels,
            'loss_depth_mean': float(np.mean(depth_losses)) if depth_losses else None,
            'loss_image_grad_steps': image_gradient_losses,
            'weighted_loss_image_grad_steps': weighted_image_gradient_losses,
            'image_grad_valid_pixels': image_gradient_valid_pixels,
            'loss_image_grad_mean': float(np.mean(image_gradient_losses)) if image_gradient_losses else None,
            'weighted_loss_image_grad_mean': float(np.mean(weighted_image_gradient_losses)) if weighted_image_gradient_losses else None,
            'loss_texture_weighted_grad_steps': texture_weighted_losses,
            'scaled_loss_texture_weighted_grad_steps': scaled_texture_weighted_losses,
            'loss_texture_weighted_grad_mean': float(np.mean(texture_weighted_losses)) if texture_weighted_losses else None,
            'texture_gradient_weight_stats': texture_weight_stats}
    if gradient_diagnostics is not None:
        from nerf.refinement_loss_gradient_diagnostics import summarize, EPSILON, APPEARANCE_PREFIXES
        result['loss_gradient_diagnostics'] = {'config': gradient_diagnostics, 'samples': gradient_samples,
                                               'summary': summarize(gradient_samples),
                                               'definitions': {'rgb_coefficient': 1.,
                                                               'image_gradient_coefficient': getattr(trainer.opt, 'lambda_image_grad', 0.),
                                                               'rgb_loss': 'current training RGB criterion, mean over channels and image pixels; uses current training background',
                                                               'rgb_criterion': type(trainer.criterion).__name__,
                                                               'image_gradient_loss': 'ordinary unweighted foreground Sobel SmoothL1 on fixed white background; excludes invalid 3x3 stencils',
                                                               'color_space': getattr(trainer.opt, 'color_space', 'srgb'),
                                                               'cosine': 'unscaled gradients; invalid if a group parameter is disconnected, nonfinite or norm <= epsilon',
                                                               'epsilon': EPSILON, 'appearance_prefixes': list(APPEARANCE_PREFIXES),
                                                               'optional_appearance_parameter': 'individual_codes',
                                                               'amp': 'autograd.grad uses current loss scale, divides gradients by it; scaler state is untouched'}}
    return result


def _warmup_vertex_ema(trainer, dataset, vertex_state, steps, beta, seed):
    if steps < 0:
        raise ValueError('Gradient EMA warmup steps must be nonnegative')
    warmup_seed = (seed + 104729) % (2 ** 32)
    indices = np.random.RandomState(warmup_seed).choice(len(dataset.poses), steps).tolist()
    if steps:
        _train_branch(trainer, dataset, indices, warmup_seed, vertex_state, beta)
    return {'method': 'zero_initialization_then_optimization', 'steps': steps,
            'seed': warmup_seed, 'training_indices': indices,
            'observed_vertices': int(np.count_nonzero(vertex_state['seen']))}


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

    if getattr(opt, 'refine_multibranch_eval', False):
        from nerf.refinement_multibranch import run_multibranch
        run_multibranch(trainer, train_data, val_data, selection_views, evaluation_views,
                        model_state, old_vertices, old_triangles, old_v_cumsum,
                        old_f_cumsum, scaler_state, start_step)
        return

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
