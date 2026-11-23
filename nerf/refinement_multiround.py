"""Consecutive local mesh refinement with face-lineage policy returns."""

import copy
import json
import os

import numpy as np
import torch
from torch import nn
from scipy.ndimage import distance_transform_edt

from nerf.refinement_experiment import _fixed_region_means, _observe, _select_views, _train_branch, _warmup_vertex_ema
from nerf.refinement_rl import file_sha256
from nerf.refinement_candidates import (CandidatePolicy, FEATURE_VERSION, candidate_features,
                                        remap_vertex_ema, visibility_counts)
from nerf.refinement_attribution import edge_outcomes


POLICY_OBJECTIVE = 'candidate_bernoulli_ema_v3'


def edge_costs(event, face_count):
    costs = np.zeros(face_count, dtype=np.int64)
    for edge in event['edge_events']:
        costs[edge['requester_face']] += edge['added_faces']
    if costs.sum() != event['added_faces']:
        raise ValueError('Edge charges do not match added face count')
    return costs


def local_rewards(rgb_before, rgb_after, sil_before, sil_after, counts, costs, weights):
    valid = counts > 0
    reward = np.full(len(counts), np.nan)
    reward[valid] = (weights[0] * (sil_before[valid] - sil_after[valid]) +
                     weights[1] * (rgb_before[valid] - rgb_after[valid]) -
                     weights[2] * costs[valid])
    return reward


def lineage_returns(transitions, gamma, return_missing=False):
    """Renormalize over valid positive-area children; retain reward if none exist."""
    result = [None] * len(transitions)
    child_stats = [None] * len(transitions)
    for t in range(len(transitions) - 1, -1, -1):
        reward = np.asarray(transitions[t]['reward'], dtype=np.float64)
        current = reward.copy()
        if t + 1 < len(transitions):
            parents = np.asarray(transitions[t]['parent_face_id'], dtype=np.int64)
            areas = np.asarray(transitions[t]['child_areas'], dtype=np.float64)
            child_returns = result[t + 1]
            if len(parents) != len(child_returns) or len(areas) != len(parents):
                raise ValueError('Lineage does not match the next round mesh')
            valid_child = np.isfinite(child_returns) & np.isfinite(areas) & (areas > 0)
            missing = np.bincount(parents[~valid_child], minlength=len(reward))
            valid_counts = np.bincount(parents[valid_child], minlength=len(reward))
            child_stats[t] = {'missing_child_counts': missing, 'valid_child_counts': valid_counts}
            total = np.bincount(parents[valid_child], weights=areas[valid_child], minlength=len(reward))
            weighted = np.bincount(parents[valid_child], weights=areas[valid_child] * child_returns[valid_child], minlength=len(reward))
            valid = np.isfinite(reward) & (total > 0)
            current[valid] = reward[valid] + gamma * weighted[valid] / total[valid]
        result[t] = current
    return (result, child_stats) if return_missing else result


def face_policy_losses(policy, transitions, returns, baseline='none', scale='none', clip=0.0,
                       return_stats=False):
    """Weight sampled request/hold log-probabilities by per-face lineage return."""
    losses = []
    stats = []
    for transition, face_returns in zip(transitions, returns):
        visible = transition['visible']
        if not len(visible):
            losses.append(None)
            stats.append({'raw': _distribution([]), 'processed': _distribution([]), 'baseline': None, 'scale': None})
            continue
        values = face_returns[visible]
        valid = np.isfinite(values)
        if not valid.any():
            losses.append(None)
            stats.append({'raw': _distribution([]), 'processed': _distribution([]), 'baseline': None, 'scale': None})
            continue
        logits = policy(torch.from_numpy(transition['features']))
        sampled = torch.from_numpy(transition['actions'].astype(np.float32))
        log_probabilities = -torch.nn.functional.binary_cross_entropy_with_logits(logits, sampled, reduction='none')
        mask = torch.from_numpy(valid)
        raw = values[valid]
        baseline_value = float(raw.mean()) if baseline == 'mean' else 0.0
        advantages = raw - baseline_value
        scale_value = float(advantages.std()) if scale == 'std' else 1.0
        if scale_value < 1e-8:
            scale_value = 1.0
        advantages = advantages / scale_value
        if clip > 0:
            advantages = np.clip(advantages, -clip, clip)
        stats.append({'raw': _distribution(raw), 'processed': _distribution(advantages),
                      'baseline': baseline_value, 'scale': scale_value})
        weighted_returns = torch.as_tensor(advantages, dtype=logits.dtype)
        losses.append(-(log_probabilities[mask] * weighted_returns).sum())
    return (losses, stats) if return_stats else losses


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {'count': 0, 'sum': None, 'mean': None, 'min': None, 'p25': None,
                'median': None, 'p75': None, 'max': None}
    quantiles = np.quantile(values, [0, .25, .5, .75, 1])
    return {'count': len(values), 'sum': float(values.sum()), 'mean': float(values.mean()),
            **dict(zip(('min', 'p25', 'median', 'p75', 'max'), map(float, quantiles)))}


def reward_distributions(transition, weights, returns=None, face_ids=None):
    valid = ((transition['reward_pixels'] > 0) & np.isfinite(transition['reward']) &
             np.isfinite(transition['sil_before']) & np.isfinite(transition['sil_after']) &
             np.isfinite(transition['rgb_before']) & np.isfinite(transition['rgb_after']))
    if face_ids is not None:
        selected = np.zeros(len(valid), dtype=bool)
        selected[np.asarray(sorted(face_ids), dtype=np.int64)] = True
        valid &= selected
    silhouette = np.where(valid, weights[0] * (transition['sil_before'] - transition['sil_after']), np.nan)
    rgb = np.where(valid, weights[1] * (transition['rgb_before'] - transition['rgb_after']), np.nan)
    cost = np.where(valid, -weights[2] * transition['costs'], np.nan)
    return {'silhouette': _distribution(silhouette), 'rgb': _distribution(rgb),
            'face_cost': _distribution(cost), 'local_reward': _distribution(transition['reward']),
            'lineage_return': _distribution([] if returns is None else np.where(valid, returns, np.nan))}


def grouped_rewards(transition, weights, returns=None):
    event = transition['event']
    executed = {int(edge['requester_face']) for edge in event['edge_events']}
    passive = set(map(int, event['passive_faces']))
    unexecuted = {int(face) for face, status in event['request_status'].items()
                  if status in ('conflict', 'budget')}
    not_considered = {int(face) for face, status in event['request_status'].items()
                      if status == 'budget_not_considered'}
    result = {}
    for name, faces in [('executed', executed), ('passive', passive),
                        ('unexecuted', unexecuted), ('not_considered', not_considered)]:
        result[name] = {'faces': len(faces), 'same_valid_set': reward_distributions(transition, weights, returns, faces)}
    return result


def action_counts(transition):
    event = transition['event']
    statuses = list(event['request_status'].values())
    considered = event['considered_count']
    return {'requested_count': int(np.count_nonzero(transition['actions'])),
            'hold_count': int(len(transition['visible']) - np.count_nonzero(transition['actions'])),
            'considered_count': considered,
            'executed_request_count': statuses.count('executed'),
            'executed_edge_count': len(event['executed_edges']),
            'conflict_count': statuses.count('conflict'),
            'budget_rejected_count': statuses.count('budget'),
            'budget_not_considered_count': statuses.count('budget_not_considered')}


def _observe_silhouette(trainer, dataset, views, training_dataset=False):
    observations = []
    trainer.model.eval()
    with torch.no_grad():
        for view in views:
            data = dataset.collate([view])
            images = data['images']
            if training_dataset:
                images = images.reshape(data['H'], data['W'], -1)
                data['images'] = images
            if images.shape[-1] != 4:
                raise ValueError('Multi-round silhouette rewards require ground-truth alpha images')
            target_alpha = images[..., 3].detach().cpu().numpy().reshape(-1).copy()
            prediction, _, target, _ = trainer.eval_step(data)
            rgb = trainer.criterion(prediction, target).mean(-1).cpu().numpy().reshape(-1).copy()
            alpha = trainer.model.refinement_alpha.cpu().numpy().reshape(-1).copy()
            faces = trainer.model.triangles_errors_id.long().cpu().numpy().reshape(-1).copy()
            observations.append((faces, rgb, np.abs(alpha - target_alpha), target_alpha,
                                 (data['H'], data['W'])))
    return observations


def _silhouette_owners(faces, target_alpha, shape, radius):
    """Attach nearby uncovered target pixels to the closest pre-update covered pixel."""
    owners = faces.reshape(shape).copy()
    covered = owners >= 0
    if not covered.any():
        return owners.reshape(-1)
    distance, nearest = distance_transform_edt(~covered, return_indices=True)
    target = target_alpha.reshape(shape) > 0
    assign = (~covered) & target & (distance <= radius)
    owners[assign] = owners[tuple(axis[assign] for axis in nearest)]
    return owners.reshape(-1)


def _region_stats(before, after, face_count, channel, silhouette_radius=None):
    if silhouette_radius is not None:
        before_pairs = [(_silhouette_owners(item[0], item[3], item[4], silhouette_radius), item[channel])
                        for item in before]
    else:
        before_pairs = [(item[0], item[channel]) for item in before]
    after_pairs = [(item[0], item[channel]) for item in after]
    counts, first = _fixed_region_means(before_pairs, before_pairs, face_count)
    _, last = _fixed_region_means(before_pairs, after_pairs, face_count)
    return counts, first, last


def _areas(vertices, faces):
    triangle = vertices[faces]
    return 0.5 * np.linalg.norm(np.cross(triangle[:, 1] - triangle[:, 0],
                                          triangle[:, 2] - triangle[:, 0]), axis=1)


def _json_values(values):
    return [float(value) if np.isfinite(value) else None for value in values]


def _save(path, state):
    temp = path + '.tmp'
    torch.save(state, temp)
    os.replace(temp, path)


def run_multiround(trainer, train_data, val_data, selection_views, evaluation_views):
    opt = trainer.opt
    horizon = getattr(opt, 'refine_multiround_horizon', 3)
    radius = getattr(opt, 'refine_silhouette_radius', 8.0)
    advantage_baseline = getattr(opt, 'refine_advantage_baseline', 'mean')
    advantage_scale = getattr(opt, 'refine_advantage_scale', 'std')
    advantage_clip = getattr(opt, 'refine_advantage_clip', 5.0)
    candidate_fraction = getattr(opt, 'refine_candidate_fraction', 0.5)
    min_area = getattr(opt, 'refine_min_face_area', 1e-12)
    ema_beta = getattr(opt, 'refine_gradient_ema_beta', 0.9)
    warmup_steps = getattr(opt, 'refine_gradient_ema_warmup_steps', 0)
    if (not opt.refine_rl or opt.refine_compare_random or opt.refine_rl_eval or
            opt.refine_face_budget < 1 or opt.refine_multiround_rounds < 1 or horizon < 1 or opt.refine_rl_lr <= 0 or
            not 0 <= opt.refine_multiround_gamma <= 1):
        raise ValueError('Multi-round training requires --refine_rl, positive budget/rounds/lr and gamma in [0,1]')
    weights = [opt.refine_multiround_sil_weight, opt.refine_multiround_rgb_weight, opt.refine_multiround_face_cost]
    if any(not np.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError('Reward weights and face cost must be finite and nonnegative')
    if not np.isfinite(radius) or radius < 0:
        raise ValueError('Silhouette assignment radius must be finite and nonnegative')
    if (advantage_baseline not in ('none', 'mean') or advantage_scale not in ('none', 'std')
            or not np.isfinite(advantage_clip) or advantage_clip < 0):
        raise ValueError('Invalid advantage baseline, scale, or clip')
    if (not 0 < candidate_fraction <= 1 or not 0 <= ema_beta < 1 or
            not np.isfinite(min_area) or min_area < 0 or warmup_steps < 0):
        raise ValueError('Invalid candidate fraction, EMA decay or minimum area')
    reward_views = opt.refine_rl_reward_views or selection_views
    _select_views(len(val_data.poses), 1, selection_views, evaluation_views)
    if (not reward_views or len(reward_views) != len(set(reward_views)) or
            any(view < 0 or view >= len(train_data.poses) for view in reward_views)):
        raise ValueError('Reward views must be unique indices in the training dataset')
    config = {'checkpoint': os.path.abspath(opt.ckpt), 'mesh': os.path.abspath(opt.mesh),
              'checkpoint_sha256': file_sha256(opt.ckpt), 'mesh_sha256': file_sha256(opt.mesh),
              'selection_views': selection_views, 'reward_views': reward_views,
              'reward_view_dataset': 'train', 'evaluation_views': evaluation_views,
              'evaluation_view_dataset': 'val', 'seed': opt.seed, 'budget': opt.refine_face_budget,
              'steps': opt.refine_experiment_steps, 'rounds': opt.refine_multiround_rounds,
              'horizon': horizon, 'silhouette_radius': radius,
              'gamma': opt.refine_multiround_gamma, 'weights': weights, 'policy_lr': opt.refine_rl_lr,
              'silhouette_error': 'absolute antialiased raster alpha minus ground-truth alpha',
              'pixel_regions': 'pre-update face index; uncovered target pixels assigned to nearest face within radius for silhouette',
              'lineage_weights': 'child area after optimization normalized over each parent',
              'cost_allocation': 'all faces added by an edge charged once to its first executing requester',
              'policy_objective': POLICY_OBJECTIVE,
              'feature_version': FEATURE_VERSION,
              'advantage_config': {'baseline': advantage_baseline, 'scale': advantage_scale, 'clip': advantage_clip},
              'lineage_missing_rule': 'renormalize positive-area valid children; otherwise retain valid local reward'}
    excluded_options = {'refine_experiment_output', 'refine_multiround_resume', 'refine_multiround_rounds',
                        'refine_multiround_horizon', 'refine_silhouette_radius'}
    config['options'] = {key: value for key, value in vars(opt).items() if key not in excluded_options}
    model = trainer.model
    torch.manual_seed(opt.seed)
    policy = CandidatePolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=opt.refine_rl_lr)
    generator = torch.Generator().manual_seed(opt.seed)
    transitions = []
    history = []
    completed = False
    vertex_state = {'ema': np.zeros(len(model.vertices), dtype=np.float64),
                    'seen': np.zeros(len(model.vertices), dtype=bool)}
    vertex_mapping = np.repeat(np.arange(len(model.vertices))[:, None], 2, axis=1)
    ema_initialization = 'zero_initial_mesh'
    ema_warmup = None
    if opt.refine_multiround_resume:
        saved = torch.load(opt.refine_multiround_resume, map_location='cpu')
        old_config = dict(saved['config'])
        requested_config = dict(config)
        old_options = dict(old_config.get('options', {}))
        requested_options = dict(requested_config.get('options', {}))
        if 'refine_gradient_ema_warmup_steps' not in old_options and warmup_steps == 0:
            requested_options.pop('refine_gradient_ema_warmup_steps', None)
        old_config['options'], requested_config['options'] = old_options, requested_options
        old_config.pop('rounds')
        requested_config.pop('rounds')
        old_horizon = old_config.pop('horizon', None)
        old_radius = old_config.pop('silhouette_radius', None)
        old_objective = old_config.pop('policy_objective', None)
        requested_config.pop('policy_objective')
        old_feature_version = old_config.pop('feature_version', None)
        requested_config.pop('feature_version')
        requested_config.pop('horizon')
        requested_config.pop('silhouette_radius')
        if not saved['completed'] and old_objective != POLICY_OBJECTIVE:
            raise ValueError('Cannot resume an open segment from a different policy objective; '
                             'resume a completed checkpoint or start from Stage 1')
        if old_feature_version != FEATURE_VERSION and not getattr(opt, 'refine_reinitialize_legacy_policy', False):
            raise ValueError('Legacy policy has incompatible input dimensions; start from Stage 1 or explicitly use --refine_reinitialize_legacy_policy on a completed trajectory')
        if saved['completed']:
            old_config.pop('pixel_regions', None)
            requested_config.pop('pixel_regions', None)
        if saved['completed'] and old_objective != POLICY_OBJECTIVE:
            old_config.pop('advantage_config', None)
            requested_config.pop('advantage_config', None)
            old_config.pop('lineage_missing_rule', None)
            requested_config.pop('lineage_missing_rule', None)
            old_options = dict(old_config.get('options', {}))
            new_options = dict(requested_config.get('options', {}))
            for key in ('refine_multibranch_eval', 'refine_multibranch_policy',
                        'refine_multibranch_rounds', 'refine_multibranch_total_budget',
                        'refine_advantage_baseline', 'refine_advantage_scale', 'refine_advantage_clip'):
                if key not in old_options:
                    new_options.pop(key, None)
            for key in ('refine_candidate_fraction', 'refine_min_face_area',
                        'refine_gradient_ema_beta', 'refine_reinitialize_legacy_policy'):
                if key not in old_options:
                    new_options.pop(key, None)
            old_config['options'], requested_config['options'] = old_options, new_options
        if (old_config != requested_config or
                (not saved['completed'] and (old_horizon != horizon or old_radius != radius or
                                              old_objective != POLICY_OBJECTIVE))):
            raise ValueError('Resume checkpoint configuration or input file hashes differ')
        model.vertices = saved['vertices'].to(trainer.device)
        model.triangles = saved['triangles'].to(trainer.device)
        model.vertices_offsets = nn.Parameter(torch.zeros_like(model.vertices))
        model.v_cumsum = saved['v_cumsum']
        model.f_cumsum = saved['f_cumsum']
        model.load_state_dict(saved['model'])
        model.triangles_errors = torch.zeros(len(model.triangles), device=trainer.device)
        model.triangles_errors_cnt = torch.zeros(len(model.triangles), device=trainer.device)
        trainer.optimizer = trainer.optimizer_fn(model)
        trainer.optimizer.load_state_dict(saved['mesh_optimizer'])
        trainer.lr_scheduler = trainer.lr_scheduler_fn(trainer.optimizer)
        trainer.lr_scheduler.load_state_dict(saved['mesh_scheduler'])
        trainer.scaler.load_state_dict(saved['scaler'])
        trainer.global_step = saved['global_step']
        if old_feature_version == FEATURE_VERSION:
            policy.load_state_dict(saved['policy'])
        if old_objective == POLICY_OBJECTIVE and old_feature_version == FEATURE_VERSION:
            optimizer.load_state_dict(saved['policy_optimizer'])
            if any(key not in saved for key in ('vertex_gradient_ema', 'vertex_gradient_seen', 'vertex_mapping')):
                raise ValueError('Candidate-policy checkpoint is missing vertex EMA or mapping')
            vertex_state = {'ema': np.asarray(saved['vertex_gradient_ema'], dtype=np.float64),
                            'seen': np.asarray(saved['vertex_gradient_seen'], dtype=bool)}
            vertex_mapping = np.asarray(saved['vertex_mapping'], dtype=np.int64)
            if (len(vertex_state['ema']) != len(model.vertices) or len(vertex_state['seen']) != len(model.vertices)
                    or vertex_mapping.shape != (len(model.vertices), 2)
                    or not np.isfinite(vertex_state['ema']).all()):
                raise ValueError('Saved vertex EMA or mapping does not match the current mesh')
            ema_initialization = saved.get('ema_initialization', 'restored')
            ema_warmup = saved.get('ema_warmup', {'method': 'legacy_checkpoint', 'steps': 0})
        else:
            vertex_state = {'ema': np.zeros(len(model.vertices), dtype=np.float64),
                            'seen': np.zeros(len(model.vertices), dtype=bool)}
            vertex_mapping = np.repeat(np.arange(len(model.vertices))[:, None], 2, axis=1)
            ema_initialization = 'legacy_mesh_zero_ema_and_reinitialized_policy'
            ema_warmup = {'method': 'legacy_mesh_zero_reinitialize', 'steps': 0}
            trainer.log('[INFO] incompatible legacy policy: initialized new policy and zero vertex EMA')
        generator.set_state(saved['generator_state'])
        history = saved.get('history', [])
        transitions = saved['transitions']
        completed = saved['completed']
        if completed:
            history = history + transitions
            transitions = []

    def snapshot(finished):
        state = {'config': config, 'completed': finished, 'global_step': trainer.global_step,
                 'model': model.state_dict(), 'vertices': model.vertices.detach(),
                 'triangles': model.triangles.detach(), 'v_cumsum': model.v_cumsum,
                 'f_cumsum': model.f_cumsum, 'mesh_optimizer': trainer.optimizer.state_dict(),
                 'mesh_scheduler': trainer.lr_scheduler.state_dict(), 'scaler': trainer.scaler.state_dict(),
                 'policy': policy.state_dict(), 'policy_optimizer': optimizer.state_dict(),
                 'vertex_gradient_ema': vertex_state['ema'], 'vertex_gradient_seen': vertex_state['seen'],
                 'vertex_mapping': vertex_mapping, 'ema_initialization': ema_initialization,
                 'ema_warmup': ema_warmup,
                 'generator_state': generator.get_state(), 'history': history, 'transitions': transitions}
        _save(os.path.join(opt.refine_experiment_output, 'trajectory.pt'), state)

    opt.refine = False
    if not opt.refine_multiround_resume:
        ema_warmup = _warmup_vertex_ema(trainer, train_data, vertex_state, warmup_steps, ema_beta, opt.seed)
        ema_initialization = 'zero_initial_mesh_then_warmup' if warmup_steps else 'zero_initial_mesh'
    target_round = min(horizon, len(transitions) + opt.refine_multiround_rounds)
    for t in range(len(transitions), target_round):
        count = len(model.triangles)
        selection = _observe(trainer, val_data, selection_views)
        before = _observe_silhouette(trainer, train_data, reward_views, training_dataset=True)
        evaluation_before = _observe_silhouette(trainer, val_data, evaluation_views)
        sums, observations, view_counts = visibility_counts(selection, count)
        model.triangles_errors = torch.tensor(sums, dtype=torch.float32, device=trainer.device)
        model.triangles_errors_cnt = torch.tensor(observations, dtype=torch.float32, device=trainer.device)
        current_vertices = (model.vertices + model.vertices_offsets.detach()).cpu().numpy()
        visible, features, feature_info = candidate_features(
            current_vertices, model.triangles.cpu().numpy(), sums, observations, view_counts,
            len(selection_views), vertex_state['ema'], vertex_state['seen'], 1.0,
            candidate_fraction, min_area)
        feature_tensor = torch.from_numpy(features)
        with torch.no_grad():
            logits = policy(feature_tensor)
            probabilities = torch.sigmoid(logits)
            actions = (torch.rand(len(visible), generator=generator) < probabilities).numpy()
            requested_positions = np.flatnonzero(actions)
            requested_positions = requested_positions[np.argsort(-logits.numpy()[requested_positions], kind='stable')]
            requested = visible[requested_positions]
        model.refine_and_decimate(export_mesh=False, candidate_order=requested, max_added_faces=opt.refine_face_budget)
        event = copy.deepcopy(model.refinement_event)
        considered_count = event['considered_count']
        for face in requested[considered_count:]:
            event['request_status'][int(face)] = 'budget_not_considered'
        parents = model.parent_face_id.cpu().numpy().copy()
        if len(model.triangles) - count != event['added_faces']:
            raise RuntimeError('Mesh change does not match recorded face budget')
        vertex_state['ema'], vertex_state['seen'], vertex_mapping = remap_vertex_ema(
            vertex_state['ema'], vertex_state['seen'], event, len(model.vertices))
        costs = edge_costs(event, count)
        global_round = len(history) + t
        step_seed = (opt.seed + global_round) % (2 ** 32)
        indices = np.random.RandomState(step_seed).choice(len(train_data.poses), opt.refine_experiment_steps).tolist()
        _train_branch(trainer, train_data, indices, step_seed, vertex_state, ema_beta)
        after = _observe_silhouette(trainer, train_data, reward_views, training_dataset=True)
        evaluation_after = _observe_silhouette(trainer, val_data, evaluation_views)
        pixel_counts, rgb_before, rgb_after = _region_stats(before, after, count, 1)
        sil_counts, sil_before, sil_after = _region_stats(before, after, count, 2, radius)
        rewards = local_rewards(rgb_before, rgb_after, sil_before, sil_after, pixel_counts, costs, weights)
        outcomes = edge_outcomes(event, count, pixel_counts, rgb_before, rgb_after,
                                 sil_counts, sil_before, sil_after)
        child_vertices = (model.vertices + model.vertices_offsets.detach()).cpu().numpy()
        child_areas = _areas(child_vertices, model.triangles.cpu().numpy())
        _, eval_rgb_before, eval_rgb_after = _region_stats(evaluation_before, evaluation_after, count, 1)
        eval_sil_counts, eval_sil_before, eval_sil_after = _region_stats(evaluation_before, evaluation_after, count, 2, radius)
        eval_counts, _, _ = _region_stats(evaluation_before, evaluation_after, count, 1)
        valid_eval = eval_counts > 0
        valid_eval_sil = eval_sil_counts > 0
        evaluation = None
        if valid_eval.any():
            evaluation = {'pixels': int(eval_counts.sum()),
                          'rgb_before': float(np.average(eval_rgb_before[valid_eval], weights=eval_counts[valid_eval])),
                          'rgb_after': float(np.average(eval_rgb_after[valid_eval], weights=eval_counts[valid_eval])),
                          'sil_pixels': int(eval_sil_counts.sum()),
                          'sil_before': float(np.average(eval_sil_before[valid_eval_sil], weights=eval_sil_counts[valid_eval_sil])) if valid_eval_sil.any() else None,
                          'sil_after': float(np.average(eval_sil_after[valid_eval_sil], weights=eval_sil_counts[valid_eval_sil])) if valid_eval_sil.any() else None}
        transitions.append({'round': global_round + 1, 'faces_before': count, 'faces_after': len(model.triangles),
                            'global_step': trainer.global_step, 'training_indices': indices, 'event': event,
                            'visible': visible, 'features': features, 'feature_scales': np.asarray(feature_info['normalization_scales']),
                            'candidate_info': feature_info, 'vertex_mapping': vertex_mapping,
                            'actions': actions, 'probabilities': probabilities.numpy(),
                            'selection_order': requested_positions, 'considered_count': considered_count,
                            'parent_face_id': parents, 'child_areas': child_areas, 'costs': costs,
                            'reward_pixels': pixel_counts, 'silhouette_pixels': sil_counts, 'rgb_before': rgb_before, 'rgb_after': rgb_after,
                            'sil_before': sil_before, 'sil_after': sil_after, 'reward': rewards,
                            'edge_outcomes': outcomes,
                            'evaluation': evaluation})
        snapshot(False)
        trainer.log('[INFO] multi-round {}: {} -> {} faces'.format(t + 1, count, len(model.triangles)))

    if len(transitions) < horizon:
        for transition in transitions:
            event = transition['event']
            passive_faces = set(event['passive_faces'])
            record = {'round': transition['round'], 'segment_status': 'open',
                      'faces_before': transition['faces_before'], 'faces_after': transition['faces_after'],
                      'global_step': transition['global_step'], 'training_indices': transition['training_indices'],
                      'visible_face_ids': transition['visible'].tolist(), 'actions': transition['actions'].tolist(),
                      'probabilities': transition['probabilities'].tolist(),
                      'candidate_info': transition['candidate_info'],
                      'candidate_features': transition['features'].tolist(),
                      'vertex_mapping': transition['vertex_mapping'].tolist(),
                      'action_status': [event['request_status'].get(int(face), 'hold') for face in transition['visible']],
                      'passively_split': [int(face) in passive_faces for face in transition['visible']],
                      'event': event, 'parent_face_id': transition['parent_face_id'].tolist(),
                      'face_costs': transition['costs'].tolist(),
                      'reward_pixels': transition['reward_pixels'].tolist(),
                      'silhouette_pixels': transition['silhouette_pixels'].tolist(),
                      'rgb_before': _json_values(transition['rgb_before']), 'rgb_after': _json_values(transition['rgb_after']),
                      'sil_before': _json_values(transition['sil_before']), 'sil_after': _json_values(transition['sil_after']),
                      'local_reward': _json_values(transition['reward']), 'lineage_return': None,
                      'reward_distributions': reward_distributions(transition, weights),
                      'reward_groups': grouped_rewards(transition, weights),
                      'edge_outcomes': transition.get('edge_outcomes'),
                      'policy_loss': None, 'evaluation': transition['evaluation']}
            record.update(action_counts(transition))
            with open(os.path.join(opt.refine_experiment_output, 'round_{:04d}_pending.json'.format(transition['round'])), 'w') as output:
                json.dump(record, output, indent=2, allow_nan=False)
        with open(os.path.join(opt.refine_experiment_output, 'metadata.json'), 'w') as output:
            json.dump(dict(config, segment_status='open', segment_rounds=len(transitions),
                           ema_initialization=ema_initialization, ema_warmup=ema_warmup), output, indent=2)
        trainer.log('[INFO] open return segment: {}/{} rounds; resume trajectory.pt'.format(len(transitions), horizon))
        return

    returns, child_stats = lineage_returns(transitions, opt.refine_multiround_gamma, return_missing=True)
    losses, advantage_stats = face_policy_losses(policy, transitions, returns, advantage_baseline,
                                                 advantage_scale, advantage_clip, return_stats=True)
    valid_losses = [loss for loss in losses if loss is not None]
    if valid_losses:
        optimizer.zero_grad()
        torch.stack(valid_losses).sum().backward()
        optimizer.step()
    for t, (transition, values, loss) in enumerate(zip(transitions, returns, losses)):
        event = transition['event']
        passive_faces = set(event['passive_faces'])
        transition['lineage_return'] = values
        transition['policy_loss'] = float(loss.detach()) if loss is not None else None
        transition['child_observation_stats'] = child_stats[t]
        transition['advantage_stats'] = advantage_stats[t]
        record = {'round': transition['round'], 'faces_before': transition['faces_before'],
                  'faces_after': transition['faces_after'], 'global_step': transition['global_step'],
                  'training_indices': transition['training_indices'], 'event': event,
                  'executed_edges': len(event['executed_edges']), 'added_faces': event['added_faces'],
                  'policy_loss': transition['policy_loss'],
                  'reward_distributions': reward_distributions(transition, weights, values),
                  'reward_groups': grouped_rewards(transition, weights, values),
                  'edge_outcomes': transition.get('edge_outcomes'),
                  'advantage_stats': advantage_stats[t],
                  'missing_descendants': None if child_stats[t] is None else {
                      'missing_child_counts': child_stats[t]['missing_child_counts'].tolist(),
                      'valid_child_counts': child_stats[t]['valid_child_counts'].tolist(),
                      'parents_with_missing_children': int(np.count_nonzero(child_stats[t]['missing_child_counts'])),
                      'valid_reward_without_valid_children': int(np.count_nonzero(
                          np.isfinite(transition['reward']) & (child_stats[t]['valid_child_counts'] == 0)))},
                  'segment_horizon': horizon,
                  'segment_start_round': transitions[0]['round'],
                  'segment_terminal_round': transitions[-1]['round'],
                  'parent_face_id': transition['parent_face_id'].tolist(),
                  'child_areas': transition['child_areas'].tolist(), 'evaluation': transition['evaluation'],
                  'visible_face_ids': transition['visible'].tolist(), 'actions': transition['actions'].tolist(),
                  'action_status': [event['request_status'].get(int(face), 'hold')
                                    for face in transition['visible']],
                  'passively_split': [int(face) in passive_faces for face in transition['visible']],
                  'probabilities': transition['probabilities'].tolist(),
                  'candidate_info': transition['candidate_info'],
                  'candidate_features': transition['features'].tolist(),
                  'vertex_mapping': transition['vertex_mapping'].tolist(),
                  'feature_scales': transition['feature_scales'].tolist(), 'face_costs': transition['costs'].tolist(),
                  'reward_pixels': transition['reward_pixels'].tolist(),
                  'silhouette_pixels': transition['silhouette_pixels'].tolist(),
                  'rgb_before': _json_values(transition['rgb_before']), 'rgb_after': _json_values(transition['rgb_after']),
                  'sil_before': _json_values(transition['sil_before']), 'sil_after': _json_values(transition['sil_after']),
                  'local_reward': _json_values(transition['reward']), 'lineage_return': _json_values(values)}
        record.update(action_counts(transition))
        with open(os.path.join(opt.refine_experiment_output, 'round_{:04d}.json'.format(transition['round'])), 'w') as output:
            json.dump(record, output, indent=2, allow_nan=False)
    snapshot(True)
    with open(os.path.join(opt.refine_experiment_output, 'metadata.json'), 'w') as output:
        json.dump(dict(config, segment_status='complete', ema_initialization=ema_initialization, ema_warmup=ema_warmup,
                       segment_start_round=transitions[0]['round'],
                       segment_terminal_round=transitions[-1]['round']), output, indent=2, allow_nan=False)
    trainer.log('[INFO] multi-round trajectory saved to ' + opt.refine_experiment_output)
