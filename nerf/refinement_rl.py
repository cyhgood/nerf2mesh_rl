"""Single-step REINFORCE experiment on a fixed Stage 1 starting mesh."""

import copy
import hashlib
import json
import os

import numpy as np
import torch
from torch import nn

from nerf.refinement_experiment import _fixed_region_means, _foreground_errors, _observe, _select_views, _train_branch


FEATURE_DESCRIPTION = ('mean RGB SmoothL1, visible pixel count, triangle area; '
                       'each transformed as min(log1p(value / positive median), 5)')


class FacePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(3, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, features):
        return self.network(features).squeeze(-1)


def face_features(vertices, faces, error_sums, counts):
    visible = np.flatnonzero(counts > 0)
    if not len(visible):
        raise ValueError('No faces observed in the selection views')
    triangles = vertices[faces[visible]]
    areas = 0.5 * np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                          triangles[:, 2] - triangles[:, 0]), axis=1)
    values = np.stack((error_sums[visible] / counts[visible], counts[visible], areas), axis=1)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Non-finite or negative face feature')
    scales = np.array([np.median(column[column > 0]) if np.any(column > 0) else 1.0
                       for column in values.T])
    features = np.minimum(np.log1p(values / scales), 5.0).astype(np.float32)
    return visible, features, scales


def executed_rewards(event, parent, before, after, control, face_count):
    counts, before_error = _fixed_region_means(before, before, face_count)
    _, after_error = _fixed_region_means(before, after, face_count)
    _, control_error = _fixed_region_means(before, control, face_count)
    children = np.bincount(parent, minlength=face_count)
    rewards = {}
    for face, status in event['request_status'].items():
        face = int(face)
        if status == 'executed' and children[face] > 1 and counts[face] > 0:
            rewards[face] = float(control_error[face] - after_error[face])
    return rewards, counts, before_error, after_error, control_error


def policy_update(policy, optimizer, features, order, considered_count, reward):
    """REINFORCE for a Plackett-Luce ordering stopped by the mesh budget."""
    if not considered_count or reward is None:
        return None
    logits = policy(features)[order]
    denominators = torch.logcumsumexp(logits.flip(0), dim=0).flip(0)
    log_probability = (logits[:considered_count] - denominators[:considered_count]).sum()
    loss = -log_probability * float(reward)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def policy_order(logits, generator, mode):
    if mode == 'sample':
        uniform = torch.rand(logits.shape, generator=generator, dtype=torch.float64)
        uniform.clamp_(min=torch.finfo(torch.float64).tiny)
        return torch.argsort(logits.double() - torch.log(-torch.log(uniform)), descending=True)
    if mode == 'rank':
        return torch.argsort(logits, descending=True, stable=True)
    if mode == 'threshold':
        return torch.nonzero(torch.sigmoid(logits) >= 0.5, as_tuple=False).view(-1)
    raise ValueError('Unknown policy selection mode')


def fixed_region_reward(before, after, control, face_count):
    counts, _ = _fixed_region_means(before, before, face_count)
    _, after_error = _fixed_region_means(before, after, face_count)
    _, control_error = _fixed_region_means(before, control, face_count)
    observed = counts > 0
    if not observed.any():
        return None
    return float(np.average(control_error[observed] - after_error[observed], weights=counts[observed]))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_policy(path, policy, optimizer, generator, round_number, provenance):
    temporary_path = path + '.tmp'
    torch.save({'policy': policy.state_dict(), 'optimizer': optimizer.state_dict(),
                'generator_state': generator.get_state(), 'round': round_number,
                'provenance': provenance, 'feature_description': FEATURE_DESCRIPTION,
                'algorithm': 'plackett_luce_budget_v1'}, temporary_path)
    os.replace(temporary_path, path)


def load_policy(path, policy, optimizer, generator, provenance):
    checkpoint = torch.load(path, map_location='cpu')
    for key in ('checkpoint_sha256', 'mesh_sha256'):
        if checkpoint['provenance'][key] != provenance[key]:
            raise ValueError('Policy checkpoint {} does not match this experiment'.format(key))
    if checkpoint.get('feature_description') != FEATURE_DESCRIPTION:
        raise ValueError('Policy checkpoint feature definition differs')
    policy.load_state_dict(checkpoint['policy'])
    if checkpoint.get('algorithm') == 'plackett_luce_budget_v1':
        optimizer.load_state_dict(checkpoint['optimizer'])
    generator.set_state(checkpoint['generator_state'])
    return checkpoint['round']


def run_refinement_rl(trainer, train_data, val_data, selection_views, evaluation_views,
                      indices, model_state, old_vertices, old_triangles,
                      old_v_cumsum, old_f_cumsum, scaler_state, start_step):
    opt = trainer.opt
    if opt.refine_compare_random or opt.refine_face_budget < 1 or opt.refine_rl_rounds < 1 or opt.refine_rl_lr <= 0:
        raise ValueError('RL requires a positive face budget, rounds and learning rate, without random comparison')
    if opt.refine_rl_eval and not opt.refine_rl_checkpoint:
        raise ValueError('RL evaluation requires --refine_rl_checkpoint')
    reward_views = opt.refine_rl_reward_views or selection_views
    _select_views(len(val_data.poses), 1, selection_views, evaluation_views)
    _select_views(len(val_data.poses), 1, reward_views, evaluation_views)
    face_count = len(old_triangles)
    model = trainer.model
    sums = model.triangles_errors.detach().cpu().numpy().copy()
    counts = model.triangles_errors_cnt.detach().cpu().numpy().copy()
    actual_vertices = (old_vertices + model.vertices_offsets.detach()).cpu().numpy()
    visible, features_array, scales = face_features(actual_vertices, old_triangles.cpu().numpy(), sums, counts)
    features = torch.from_numpy(features_array)
    provenance = {'checkpoint': os.path.abspath(opt.ckpt), 'mesh': os.path.abspath(opt.mesh),
                  'checkpoint_sha256': file_sha256(opt.ckpt), 'mesh_sha256': file_sha256(opt.mesh)}
    torch.manual_seed(opt.seed)
    policy = FacePolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=opt.refine_rl_lr)
    generator = torch.Generator().manual_seed(opt.seed)
    start_round = 0
    if opt.refine_rl_checkpoint:
        start_round = load_policy(opt.refine_rl_checkpoint, policy, optimizer, generator, provenance)

    def restore():
        model.vertices = old_vertices.clone()
        model.triangles = old_triangles.clone()
        model.vertices_offsets = nn.Parameter(torch.zeros_like(old_vertices))
        model.v_cumsum = old_v_cumsum.copy()
        model.f_cumsum = old_f_cumsum.copy()
        model.load_state_dict(model_state)
        model.triangles_errors = torch.as_tensor(sums.copy(), device=trainer.device)
        model.triangles_errors_cnt = torch.as_tensor(counts.copy(), device=trainer.device)
        trainer.scaler.load_state_dict(scaler_state)
        trainer.global_step = start_step

    before_reward = _observe(trainer, val_data, reward_views)
    before_eval = _observe(trainer, val_data, evaluation_views)
    restore()
    model.vertices = model.vertices + model.vertices_offsets.detach()
    model.vertices_offsets = nn.Parameter(torch.zeros_like(model.vertices))
    opt.refine = False
    _train_branch(trainer, train_data, indices, opt.seed)
    control_reward = _observe(trainer, val_data, reward_views)
    control_eval = _observe(trainer, val_data, evaluation_views)

    metadata = dict(provenance, selection_views=selection_views, reward_views=reward_views,
                    evaluation_views=evaluation_views, training_indices=indices, seed=opt.seed,
                    face_budget=opt.refine_face_budget, optimization_steps=opt.refine_experiment_steps,
                    policy_learning_rate=optimizer.param_groups[0]['lr'],
                    policy_mode='evaluation' if opt.refine_rl_eval else 'training',
                    loaded_policy=os.path.abspath(opt.refine_rl_checkpoint) if opt.refine_rl_checkpoint else None,
                    loaded_policy_sha256=file_sha256(opt.refine_rl_checkpoint) if opt.refine_rl_checkpoint else None,
                    stage1_global_step=start_step, feature_description=FEATURE_DESCRIPTION,
                    feature_scales=scales.tolist(), algorithm='Plackett-Luce budget-stopped REINFORCE',
                    reward='pixel-weighted control minus refined RGB SmoothL1 on all fixed pre-update face pixels',
                    action_order='Gumbel-sampled weighted order for training; rank or threshold for evaluation',
                    score_interpretation='probability is legacy sigmoid score; initial_selection_weight is softmax first-draw probability',
                    eval_mode=getattr(opt, 'refine_rl_eval_mode', 'rank'),
                    legacy_checkpoint_optimizer='reset when loaded checkpoint predates plackett_luce_budget_v1',
                    checkpoint_limit='input mesh and Stage 1 checkpoint topology correspondence is not verified')
    with open(os.path.join(opt.refine_experiment_output, 'rl_metadata.json'), 'x') as output:
        json.dump(metadata, output, indent=2, allow_nan=False)

    for iteration in range(opt.refine_rl_rounds):
        restore()
        with torch.no_grad():
            logits = policy(features)
            probabilities = torch.sigmoid(logits)
            selection_weights = torch.softmax(logits, dim=0)
            mode = getattr(opt, 'refine_rl_eval_mode', 'rank') if opt.refine_rl_eval else 'sample'
            order = policy_order(logits, generator, mode)
            requested = visible[order.numpy()]
        model.refine_and_decimate(export_mesh=False, candidate_order=requested,
                                  max_added_faces=opt.refine_face_budget)
        event = copy.deepcopy(model.refinement_event)
        parent = model.parent_face_id.cpu().numpy().copy()
        actual_added = len(model.triangles) - face_count
        if actual_added != event['added_faces']:
            raise RuntimeError('Recorded face budget does not match the resulting topology')
        considered_count = event['considered_count']
        for face in requested[considered_count:]:
            event['request_status'][int(face)] = 'budget_not_considered'
        considered_faces = set(map(int, requested[:considered_count]))
        _train_branch(trainer, train_data, indices, opt.seed)
        after_reward = _observe(trainer, val_data, reward_views)
        after_eval = _observe(trainer, val_data, evaluation_views)
        rewards, reward_counts, baseline, after, control = executed_rewards(
            event, parent, before_reward, after_reward, control_reward, face_count)
        global_reward = fixed_region_reward(before_reward, after_reward, control_reward, face_count)
        loss = None if opt.refine_rl_eval else policy_update(policy, optimizer, features, order, considered_count, global_reward)
        by_view, overall = _foreground_errors(before_eval, after_eval, control_eval, evaluation_views)
        statuses = event['request_status']
        summary = np.quantile(probabilities.numpy(), [0, 0.25, 0.5, 0.75, 1]).tolist()
        record = {'round': start_round + iteration + 1, 'global_step': start_step,
                  'selection_mode': mode, 'visible_candidate_count': len(visible),
                  'ranked_candidate_count': len(order), 'considered_count': considered_count,
                  'sigmoid_score_quantiles': dict(zip(['min', 'q25', 'median', 'q75', 'max'], summary)),
                  'initial_selection_weight_min': float(selection_weights.min()),
                  'initial_selection_weight_max': float(selection_weights.max()),
                  'requested_count': considered_count, 'hold_count': int(len(visible) - considered_count),
                  'executed_count': sum(s == 'executed' for s in statuses.values()),
                  'executed_edge_count': len(event['executed_edges']),
                  'conflict_count': sum(s == 'conflict' for s in statuses.values()),
                  'budget_skipped_count': sum(s in ('budget', 'budget_not_considered') for s in statuses.values()),
                  'actual_added_faces': actual_added, 'mean_reward': float(np.mean(list(rewards.values()))) if rewards else None,
                  'global_reward': global_reward, 'reward_count': len(rewards), 'policy_loss': loss, 'event': event,
                  'parent_face_id': parent.tolist(), 'foreground_errors_by_view': by_view,
                  'foreground_errors_overall': overall,
                  'decisions': [{'face_id': int(face), 'probability': float(probabilities[i]),
                                 'requested': int(face) in considered_faces,
                                 'initial_selection_weight': float(selection_weights[i]),
                                 'status': statuses.get(int(face), 'passive' if int(face) in event['passive_faces'] else 'hold'),
                                 'passively_split': int(face) in event['passive_faces'],
                                 'reward_pixels': int(reward_counts[face]),
                                 'before_error': float(baseline[face]) if reward_counts[face] else None,
                                 'after_error': float(after[face]) if reward_counts[face] else None,
                                 'control_error': float(control[face]) if reward_counts[face] else None,
                                 'reward': rewards.get(int(face))} for i, face in enumerate(visible)]}
        path = os.path.join(opt.refine_experiment_output, 'round_{:04d}.json'.format(iteration + 1))
        with open(path, 'x') as output:
            json.dump(record, output, indent=2, allow_nan=False)
        if not opt.refine_rl_eval:
            save_policy(os.path.join(opt.refine_experiment_output, 'policy.pt'), policy, optimizer,
                        generator, start_round + iteration + 1, provenance)
        trainer.log('[INFO] RL round {}: {} added faces, reward {}, loss {}'.format(
            record['round'], actual_added, record['mean_reward'], loss))
