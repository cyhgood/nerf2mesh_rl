"""Consecutive local mesh refinement with face-lineage policy returns."""

import copy
import json
import os

import numpy as np
import torch
from torch import nn
from scipy.ndimage import distance_transform_edt

from nerf.refinement_experiment import _fixed_region_means, _observe, _select_views, _train_branch
from nerf.refinement_rl import FacePolicy, face_features, file_sha256


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


def lineage_returns(transitions, gamma):
    """Propagate returns backwards; any missing positive-area child invalidates its parent."""
    result = [None] * len(transitions)
    for t in range(len(transitions) - 1, -1, -1):
        reward = np.asarray(transitions[t]['reward'], dtype=np.float64)
        current = reward.copy()
        if t + 1 < len(transitions):
            parents = np.asarray(transitions[t]['parent_face_id'], dtype=np.int64)
            areas = np.asarray(transitions[t]['child_areas'], dtype=np.float64)
            child_returns = result[t + 1]
            if len(parents) != len(child_returns) or len(areas) != len(parents):
                raise ValueError('Lineage does not match the next round mesh')
            total = np.bincount(parents, weights=areas, minlength=len(reward))
            missing = np.bincount(parents, weights=areas * ~np.isfinite(child_returns), minlength=len(reward))
            weighted = np.bincount(parents, weights=areas * np.nan_to_num(child_returns, nan=0.0), minlength=len(reward))
            valid = np.isfinite(reward) & (total > 0) & (missing == 0)
            current[:] = np.nan
            current[valid] = reward[valid] + gamma * weighted[valid] / total[valid]
        result[t] = current
    return result


def joint_policy_loss(policy, transitions, returns):
    """Score the complete sampled joint action against the segment-start objective."""
    valid = np.isfinite(returns[0])
    if not valid.any():
        return None, []
    objective = float(np.mean(returns[0][valid]))
    losses = []
    for transition in transitions:
        logits = policy(torch.from_numpy(transition['features']))
        actions = torch.from_numpy(transition['actions'].astype(np.float32))
        log_probs = -torch.nn.functional.binary_cross_entropy_with_logits(logits, actions, reduction='none')
        losses.append(-log_probs.sum() * objective)
    return objective, losses


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
    if (not opt.refine_rl or opt.refine_compare_random or opt.refine_rl_eval or
            opt.refine_face_budget < 1 or opt.refine_multiround_rounds < 1 or horizon < 1 or opt.refine_rl_lr <= 0 or
            not 0 <= opt.refine_multiround_gamma <= 1):
        raise ValueError('Multi-round training requires --refine_rl, positive budget/rounds/lr and gamma in [0,1]')
    weights = [opt.refine_multiround_sil_weight, opt.refine_multiround_rgb_weight, opt.refine_multiround_face_cost]
    if any(not np.isfinite(weight) or weight < 0 for weight in weights):
        raise ValueError('Reward weights and face cost must be finite and nonnegative')
    if not np.isfinite(radius) or radius < 0:
        raise ValueError('Silhouette assignment radius must be finite and nonnegative')
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
              'cost_allocation': 'all faces added by an edge charged once to its first executing requester'}
    excluded_options = {'refine_experiment_output', 'refine_multiround_resume', 'refine_multiround_rounds',
                        'refine_multiround_horizon', 'refine_silhouette_radius'}
    config['options'] = {key: value for key, value in vars(opt).items() if key not in excluded_options}
    model = trainer.model
    torch.manual_seed(opt.seed)
    policy = FacePolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=opt.refine_rl_lr)
    generator = torch.Generator().manual_seed(opt.seed)
    transitions = []
    history = []
    completed = False
    if opt.refine_multiround_resume:
        saved = torch.load(opt.refine_multiround_resume, map_location='cpu')
        old_config = dict(saved['config'])
        requested_config = dict(config)
        old_config.pop('rounds')
        requested_config.pop('rounds')
        old_horizon = old_config.pop('horizon', None)
        old_radius = old_config.pop('silhouette_radius', None)
        requested_config.pop('horizon')
        requested_config.pop('silhouette_radius')
        if saved['completed']:
            old_config.pop('pixel_regions', None)
            requested_config.pop('pixel_regions', None)
        if (old_config != requested_config or
                (not saved['completed'] and (old_horizon != horizon or old_radius != radius))):
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
        policy.load_state_dict(saved['policy'])
        optimizer.load_state_dict(saved['policy_optimizer'])
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
                 'generator_state': generator.get_state(), 'history': history, 'transitions': transitions}
        _save(os.path.join(opt.refine_experiment_output, 'trajectory.pt'), state)

    opt.refine = False
    target_round = min(horizon, len(transitions) + opt.refine_multiround_rounds)
    for t in range(len(transitions), target_round):
        count = len(model.triangles)
        selection = _observe(trainer, val_data, selection_views)
        before = _observe_silhouette(trainer, train_data, reward_views, training_dataset=True)
        evaluation_before = _observe_silhouette(trainer, val_data, evaluation_views)
        sums = np.zeros(count, dtype=np.float64)
        observations = np.zeros(count, dtype=np.float64)
        for faces, losses, _ in selection:
            mask = faces >= 0
            sums += np.bincount(faces[mask], weights=losses[mask], minlength=count)
            observations += np.bincount(faces[mask], minlength=count)
        model.triangles_errors = torch.tensor(sums, dtype=torch.float32, device=trainer.device)
        model.triangles_errors_cnt = torch.tensor(observations, dtype=torch.float32, device=trainer.device)
        current_vertices = (model.vertices + model.vertices_offsets.detach()).cpu().numpy()
        visible, features, scales = face_features(current_vertices, model.triangles.cpu().numpy(), sums, observations)
        feature_tensor = torch.from_numpy(features)
        with torch.no_grad():
            probabilities = torch.sigmoid(policy(feature_tensor))
            actions = torch.rand(len(visible), generator=generator) < probabilities
            requested = visible[actions.numpy()]
            requested = requested[torch.randperm(len(requested), generator=generator).numpy()]
        model.refine_and_decimate(export_mesh=False, candidate_order=requested, max_added_faces=opt.refine_face_budget)
        event = copy.deepcopy(model.refinement_event)
        parents = model.parent_face_id.cpu().numpy().copy()
        if len(model.triangles) - count != event['added_faces']:
            raise RuntimeError('Mesh change does not match recorded face budget')
        costs = edge_costs(event, count)
        global_round = len(history) + t
        step_seed = (opt.seed + global_round) % (2 ** 32)
        indices = np.random.RandomState(step_seed).choice(len(train_data.poses), opt.refine_experiment_steps).tolist()
        _train_branch(trainer, train_data, indices, step_seed)
        after = _observe_silhouette(trainer, train_data, reward_views, training_dataset=True)
        evaluation_after = _observe_silhouette(trainer, val_data, evaluation_views)
        pixel_counts, rgb_before, rgb_after = _region_stats(before, after, count, 1)
        sil_counts, sil_before, sil_after = _region_stats(before, after, count, 2, radius)
        rewards = local_rewards(rgb_before, rgb_after, sil_before, sil_after, pixel_counts, costs, weights)
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
                            'visible': visible, 'features': features, 'feature_scales': scales,
                            'actions': actions.numpy(), 'probabilities': probabilities.numpy(),
                            'parent_face_id': parents, 'child_areas': child_areas, 'costs': costs,
                            'reward_pixels': pixel_counts, 'silhouette_pixels': sil_counts, 'rgb_before': rgb_before, 'rgb_after': rgb_after,
                            'sil_before': sil_before, 'sil_after': sil_after, 'reward': rewards,
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
                      'action_status': [event['request_status'].get(int(face), 'hold') for face in transition['visible']],
                      'passively_split': [int(face) in passive_faces for face in transition['visible']],
                      'event': event, 'parent_face_id': transition['parent_face_id'].tolist(),
                      'face_costs': transition['costs'].tolist(),
                      'reward_pixels': transition['reward_pixels'].tolist(),
                      'silhouette_pixels': transition['silhouette_pixels'].tolist(),
                      'rgb_before': _json_values(transition['rgb_before']), 'rgb_after': _json_values(transition['rgb_after']),
                      'sil_before': _json_values(transition['sil_before']), 'sil_after': _json_values(transition['sil_after']),
                      'local_reward': _json_values(transition['reward']), 'lineage_return': None,
                      'policy_loss': None, 'evaluation': transition['evaluation']}
            with open(os.path.join(opt.refine_experiment_output, 'round_{:04d}_pending.json'.format(transition['round'])), 'w') as output:
                json.dump(record, output, indent=2, allow_nan=False)
        with open(os.path.join(opt.refine_experiment_output, 'metadata.json'), 'w') as output:
            json.dump(dict(config, segment_status='open', segment_rounds=len(transitions)), output, indent=2)
        trainer.log('[INFO] open return segment: {}/{} rounds; resume trajectory.pt'.format(len(transitions), horizon))
        return

    returns = lineage_returns(transitions, opt.refine_multiround_gamma)
    objective, losses = joint_policy_loss(policy, transitions, returns)
    if losses:
        optimizer.zero_grad()
        torch.stack(losses).sum().backward()
        optimizer.step()
    else:
        losses = [None] * len(transitions)
    for t, (transition, values, loss) in enumerate(zip(transitions, returns, losses)):
        event = transition['event']
        passive_faces = set(event['passive_faces'])
        transition['lineage_return'] = values
        transition['policy_loss'] = float(loss.detach()) if loss is not None else None
        record = {'round': transition['round'], 'faces_before': transition['faces_before'],
                  'faces_after': transition['faces_after'], 'global_step': transition['global_step'],
                  'training_indices': transition['training_indices'], 'event': event,
                  'requested_count': int(transition['actions'].sum()),
                  'executed_edges': len(event['executed_edges']), 'added_faces': event['added_faces'],
                  'policy_loss': transition['policy_loss'],
                  'segment_objective': objective, 'segment_horizon': horizon,
                  'segment_start_round': transitions[0]['round'],
                  'segment_terminal_round': transitions[-1]['round'],
                  'parent_face_id': transition['parent_face_id'].tolist(),
                  'child_areas': transition['child_areas'].tolist(), 'evaluation': transition['evaluation'],
                  'visible_face_ids': transition['visible'].tolist(), 'actions': transition['actions'].tolist(),
                  'action_status': [event['request_status'].get(int(face), 'hold')
                                    for face in transition['visible']],
                  'passively_split': [int(face) in passive_faces for face in transition['visible']],
                  'probabilities': transition['probabilities'].tolist(),
                  'feature_scales': transition['feature_scales'].tolist(), 'face_costs': transition['costs'].tolist(),
                  'reward_pixels': transition['reward_pixels'].tolist(),
                  'silhouette_pixels': transition['silhouette_pixels'].tolist(),
                  'rgb_before': _json_values(transition['rgb_before']), 'rgb_after': _json_values(transition['rgb_after']),
                  'sil_before': _json_values(transition['sil_before']), 'sil_after': _json_values(transition['sil_after']),
                  'local_reward': _json_values(transition['reward']), 'lineage_return': _json_values(values)}
        with open(os.path.join(opt.refine_experiment_output, 'round_{:04d}.json'.format(transition['round'])), 'w') as output:
            json.dump(record, output, indent=2, allow_nan=False)
    snapshot(True)
    with open(os.path.join(opt.refine_experiment_output, 'metadata.json'), 'w') as output:
        json.dump(dict(config, segment_status='complete', segment_start_round=transitions[0]['round'],
                       segment_terminal_round=transitions[-1]['round']), output, indent=2, allow_nan=False)
    trainer.log('[INFO] multi-round trajectory saved to ' + opt.refine_experiment_output)
