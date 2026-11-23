"""Compare fixed-policy, high-error and random consecutive refinements."""

import copy
import json
import os

import numpy as np
import torch
from torch import nn
from scipy.ndimage import binary_erosion, distance_transform_edt

from nerf.refinement_experiment import _select_views, _train_branch, _warmup_vertex_ema
from nerf.refinement_multiround import _observe_silhouette, _region_stats
from nerf.refinement_rl import FacePolicy, face_features, file_sha256
from nerf.refinement_candidates import (CandidatePolicy, FEATURE_VERSION, SequentialFaceDecisions,
                                        candidate_features, remap_vertex_ema, visibility_counts)
from nerf.refinement_attribution import edge_outcomes


def load_trajectory_policy(path, checkpoint_hash, mesh_hash):
    trajectory = torch.load(path, map_location='cpu')
    config = trajectory.get('config', {})
    if config.get('checkpoint_sha256') != checkpoint_hash or config.get('mesh_sha256') != mesh_hash:
        raise ValueError('Policy trajectory does not match the Stage 1 checkpoint and mesh hashes')
    if 'policy' not in trajectory:
        raise ValueError('Trajectory checkpoint does not contain policy weights')
    policy = CandidatePolicy() if config.get('feature_version') == FEATURE_VERSION else FacePolicy()
    if isinstance(policy, CandidatePolicy):
        options = config.get('options', {})
        policy.candidate_settings = {'fraction': options.get('refine_candidate_fraction', 0.5),
                                     'min_area': options.get('refine_min_face_area', 1e-12),
                                     'ema_beta': options.get('refine_gradient_ema_beta', 0.9),
                                     'training_ema_warmup_steps': options.get('refine_gradient_ema_warmup_steps', 0)}
        policy.training_objective = config.get('policy_objective', 'unknown')
    try:
        policy.load_state_dict(trajectory['policy'])
    except RuntimeError as error:
        raise ValueError('Policy checkpoint has incompatible feature dimensions') from error
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


def full_frame_metrics(observations, views, expected_shapes):
    per_view = []
    rgb_total = 0.0
    sil_total = 0.0
    pixels_total = 0
    if len(observations) != len(views) or len(views) != len(expected_shapes):
        raise ValueError('Evaluation views do not match the fixed pixel set')
    for view, observation, shape in zip(views, observations, expected_shapes):
        if tuple(observation[4]) != tuple(shape):
            raise ValueError('Evaluation image dimensions changed between branches')
        rgb = observation[1]
        sil = observation[2]
        pixels = int(np.prod(shape))
        if len(rgb) != pixels or len(sil) != pixels:
            raise ValueError('Evaluation output does not match the fixed pixel set')
        rgb_sum = float(np.sum(rgb, dtype=np.float64))
        sil_sum = float(np.sum(sil, dtype=np.float64))
        rgb_total += rgb_sum
        sil_total += sil_sum
        pixels_total += pixels
        per_view.append({'view_id': int(view), 'pixels': pixels,
                         'rgb_error': rgb_sum / pixels, 'silhouette_error': sil_sum / pixels})
    if not pixels_total:
        raise ValueError('Evaluation pixel set is empty')
    return {'per_view': per_view, 'overall': {'pixels': pixels_total,
                                               'rgb_error': rgb_total / pixels_total,
                                               'silhouette_error': sil_total / pixels_total}}


def _save_json(path, value):
    temporary = path + '.tmp'
    with open(temporary, 'w') as output:
        json.dump(value, output, indent=2, allow_nan=False)
    os.replace(temporary, path)


def selection_face_observations(observations, face_count, vertices, faces, radius):
    sums = np.zeros(face_count, dtype=np.float64)
    counts = np.zeros(face_count, dtype=np.int64)
    near_counts = np.zeros(face_count, dtype=np.int64)
    min_distance = np.full(face_count, np.inf)
    contour_views = 0
    for face_ids, losses, _, target_alpha, shape in observations:
        valid = face_ids >= 0
        sums += np.bincount(face_ids[valid], weights=losses[valid], minlength=face_count)
        counts += np.bincount(face_ids[valid], minlength=face_count)
        foreground = target_alpha.reshape(shape) > 0.5
        boundary = foreground & ~binary_erosion(foreground, border_value=0)
        if not boundary.any():
            continue
        contour_views += 1
        distance = distance_transform_edt(~boundary).reshape(-1)
        near = valid & (distance <= radius)
        near_counts += np.bincount(face_ids[near], minlength=face_count)
        np.minimum.at(min_distance, face_ids[valid], distance[valid])
    means = np.full(face_count, np.nan)
    np.divide(sums, counts, out=means, where=counts > 0)
    min_distance[~np.isfinite(min_distance)] = np.nan
    triangle = vertices[faces]
    areas = 0.5 * np.linalg.norm(np.cross(triangle[:, 1] - triangle[:, 0],
                                          triangle[:, 2] - triangle[:, 0]), axis=1)
    return sums, counts, means, areas, near_counts, min_distance, contour_views


def run_multibranch(trainer, train_data, val_data, selection_views, evaluation_views,
                    model_state, old_vertices, old_triangles, old_v_cumsum,
                    old_f_cumsum, scaler_state, start_step):
    opt = trainer.opt
    contour_radius = getattr(opt, 'refine_silhouette_radius', 8.0)
    if opt.refine_multiround or opt.refine_rl or opt.refine_compare_random or opt.refine_rl_eval:
        raise ValueError('Multi-branch evaluation must be used without other experiment modes')
    if opt.refine_multibranch_rounds < 1 or opt.refine_face_budget < 1:
        raise ValueError('Evaluation rounds and per-round face budget must be positive')
    total_budget = opt.refine_multibranch_total_budget
    if total_budget < 0:
        raise ValueError('Total face budget must be nonnegative')
    if total_budget == 0:
        total_budget = opt.refine_multibranch_rounds * opt.refine_face_budget
    seeds = opt.refine_random_seeds
    if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 or seed >= 2**32 for seed in seeds):
        raise ValueError('Random selection seeds must be unique integers in [0, 2**32 - 1]')
    if not os.path.isfile(opt.refine_multibranch_policy):
        raise ValueError('Provide --refine_multibranch_policy with a trajectory.pt file')
    _select_views(len(val_data.poses), 1, selection_views, evaluation_views)
    checkpoint_hash, mesh_hash = file_sha256(opt.ckpt), file_sha256(opt.mesh)
    policy = load_trajectory_policy(opt.refine_multibranch_policy, checkpoint_hash, mesh_hash)
    initial_policy = copy.deepcopy(policy.state_dict())
    model = trainer.model
    warmup_steps = getattr(opt, 'refine_gradient_ema_warmup_steps', 0)
    warmup_beta = policy.candidate_settings['ema_beta'] if isinstance(policy, CandidatePolicy) else getattr(opt, 'refine_gradient_ema_beta', 0.9)
    warmup_state = {'ema': np.zeros(len(model.vertices), dtype=np.float64),
                    'seen': np.zeros(len(model.vertices), dtype=bool)}
    checkpoint_step = start_step
    if isinstance(policy, CandidatePolicy) and warmup_steps != policy.candidate_settings['training_ema_warmup_steps']:
        trainer.log('[INFO] evaluation EMA warmup differs from the policy training configuration')
    opt.refine = False
    ema_warmup = _warmup_vertex_ema(trainer, train_data, warmup_state, warmup_steps, warmup_beta, opt.seed)
    if warmup_steps:
        model_state = copy.deepcopy(model.state_dict())
        old_vertices, old_triangles = model.vertices.clone(), model.triangles.clone()
        old_v_cumsum, old_f_cumsum = model.v_cumsum.copy(), model.f_cumsum.copy()
        scaler_state, start_step = copy.deepcopy(trainer.scaler.state_dict()), trainer.global_step
    initial = _observe_silhouette(trainer, val_data, evaluation_views)
    shapes = [item[4] for item in initial]
    initial_metrics = full_frame_metrics(initial, evaluation_views, shapes)

    def restore():
        model.vertices = old_vertices.clone()
        model.triangles = old_triangles.clone()
        model.vertices_offsets = nn.Parameter(torch.zeros_like(old_vertices))
        model.v_cumsum = old_v_cumsum.copy()
        model.f_cumsum = old_f_cumsum.copy()
        model.load_state_dict(model_state)
        model.triangles_errors = torch.zeros(len(old_triangles), device=trainer.device)
        model.triangles_errors_cnt = torch.zeros(len(old_triangles), device=trainer.device)
        trainer.scaler.load_state_dict(scaler_state)
        trainer.global_step = start_step

    # Draw samples once so every branch optimizes exactly the same sequence.
    training_indices = []
    training_seeds = []
    for round_index in range(opt.refine_multibranch_rounds):
        seed = (opt.seed + round_index) % (2 ** 32)
        training_seeds.append(seed)
        training_indices.append(np.random.RandomState(seed).choice(
            len(train_data.poses), opt.refine_experiment_steps).tolist())

    result = {'checkpoint': os.path.abspath(opt.ckpt), 'checkpoint_sha256': checkpoint_hash,
              'mesh': os.path.abspath(opt.mesh), 'mesh_sha256': mesh_hash,
              'policy_checkpoint': os.path.abspath(opt.refine_multibranch_policy),
              'policy_checkpoint_sha256': file_sha256(opt.refine_multibranch_policy),
              'selection_views': selection_views, 'evaluation_views': evaluation_views,
              'evaluation_pixel_set': 'all pixels in each fixed validation view',
              'evaluation_shapes': [list(shape) for shape in shapes],
              'rgb_metric': 'RGB mean SmoothL1 on white background',
              'silhouette_metric': 'absolute antialiased raster alpha minus ground-truth alpha',
              'initial_faces': len(old_triangles), 'initial_global_step': start_step,
              'checkpoint_global_step': checkpoint_step, 'ema_warmup': ema_warmup,
              'initial_metrics': initial_metrics, 'rounds': opt.refine_multibranch_rounds,
              'per_round_budget': opt.refine_face_budget, 'total_budget': total_budget,
              'optimization_steps': opt.refine_experiment_steps,
              'training_indices': training_indices, 'training_seeds': training_seeds,
              'rl_selection': 'descending shared policy score among visible faces',
              'high_error_selection': 'descending average RGB error among visible faces',
              'random_selection': 'seeded permutation of visible faces', 'branches': []}
    result['face_observations'] = ('NPZ arrays indexed by pre-update face ID; contour distance is the minimum '
                                   'pixel distance to a ground-truth alpha > 0.5 boundary over selection views')
    result['contour_radius_pixels'] = contour_radius
    result['rl_policy_feature_version'] = FEATURE_VERSION if isinstance(policy, CandidatePolicy) else 'legacy_three_features'
    result['rl_vertex_ema_initialization'] = 'zeros on starting Stage 1 mesh followed by recorded warmup optimization steps'
    if isinstance(policy, CandidatePolicy):
        result['rl_selection'] = 'area order; sequential sigmoid threshold decisions using current remaining budget'
        result['rl_candidate_settings'] = policy.candidate_settings
        result['rl_training_objective'] = policy.training_objective

    branches = [('rl', None), ('high_error', None)] + [('random', seed) for seed in seeds]
    _save_json(os.path.join(opt.refine_experiment_output, 'multibranch_metadata.json'), result)
    opt.refine = False
    for method, random_seed in branches:
        restore()
        vertex_state = {'ema': warmup_state['ema'].copy(), 'seen': warmup_state['seen'].copy()}
        branch = {'method': method, 'random_seed': random_seed,
                  'initial_faces': len(model.triangles), 'initial_global_step': trainer.global_step,
                  'initial_metrics': initial_metrics, 'rounds': []}
        cumulative = 0
        random_generator = np.random.RandomState(random_seed) if method == 'random' else None
        branch_name = '{}_{}'.format(method, random_seed) if random_seed is not None else method
        for round_index in range(opt.refine_multibranch_rounds):
            face_count = len(model.triangles)
            observations = _observe_silhouette(trainer, val_data, selection_views)
            vertices = (model.vertices + model.vertices_offsets.detach()).cpu().numpy()
            faces = model.triangles.cpu().numpy()
            sums, counts, means, areas, near_counts, min_distance, contour_views = selection_face_observations(
                observations, face_count, vertices, faces, contour_radius)
            model.triangles_errors = torch.tensor(sums, dtype=torch.float32, device=trainer.device)
            model.triangles_errors_cnt = torch.tensor(counts, dtype=torch.float32, device=trainer.device)
            visible = np.flatnonzero(counts > 0)
            candidate_info = None
            decisions = None
            candidate_mask = np.zeros(face_count, dtype=bool)
            candidate_feature_values = np.full((face_count, 9), np.nan, dtype=np.float32)
            request_probability = np.full(face_count, np.nan, dtype=np.float32)
            round_budget = min(opt.refine_face_budget, total_budget - cumulative)
            if method == 'rl' and (len(visible) or isinstance(policy, CandidatePolicy)):
                if isinstance(policy, CandidatePolicy):
                    _, _, view_counts = visibility_counts(observations, face_count)
                    visible, features, candidate_info = candidate_features(
                        vertices, faces, sums, counts, view_counts, len(selection_views),
                        vertex_state['ema'], vertex_state['seen'], 1.0 if round_budget else 0.0,
                        policy.candidate_settings['fraction'], policy.candidate_settings['min_area'])
                    candidate_mask[visible] = True
                    decisions = SequentialFaceDecisions(policy, visible, features, round_budget, deterministic=True)
                    order = visible
                else:
                    visible, features, _ = face_features(vertices, faces, sums, counts)
                    candidate_mask[visible] = True
                    with torch.no_grad():
                        scores = policy(torch.from_numpy(features)).numpy()
                    order = visible[np.argsort(-scores, kind='stable')]
            elif method == 'high_error':
                order = visible[np.argsort(-(sums[visible] / counts[visible]), kind='stable')]
            elif method == 'random':
                order = random_generator.permutation(visible)
            else:
                order = visible
            if decisions is None:
                model.refine_and_decimate(export_mesh=False, candidate_order=order, max_added_faces=round_budget)
            else:
                model.refine_and_decimate(export_mesh=False, candidate_order=order,
                                          max_added_faces=round_budget, decision_fn=decisions)
                request_probability[visible] = decisions.probabilities
                candidate_feature_values[visible] = decisions.features
            event = copy.deepcopy(model.refinement_event)
            actual = len(model.triangles) - face_count
            if actual != event['added_faces'] or actual > round_budget:
                raise RuntimeError('Recorded face count does not match the budgeted topology')
            cumulative += actual
            event['global_step'] = trainer.global_step
            event['parent_face_id'] = model.parent_face_id.cpu().numpy().tolist()
            if method == 'rl' and isinstance(policy, CandidatePolicy):
                vertex_state['ema'], vertex_state['seen'], vertex_mapping = remap_vertex_ema(
                    vertex_state['ema'], vertex_state['seen'], event, len(model.vertices))
                event['vertex_mapping'] = vertex_mapping.tolist()
            status = np.full(face_count, 'not_visible', dtype='<U24')
            status[counts > 0] = 'not_candidate' if method == 'rl' and isinstance(policy, CandidatePolicy) else 'budget_not_considered'
            status[visible] = 'hold' if method == 'rl' and isinstance(policy, CandidatePolicy) else 'budget_not_considered'
            status[order] = 'budget_not_considered'
            for face in order[:event['considered_count']]:
                status[face] = event['request_status'][int(face)]
            passive = np.zeros(face_count, dtype=bool)
            passive[event['passive_faces']] = True
            executed = np.zeros(face_count, dtype=bool)
            executed[[edge['requester_face'] for edge in event['edge_events']]] = True
            observations_file = 'faces_{}_round_{:04d}.npz'.format(branch_name, round_index + 1)
            _train_branch(trainer, train_data, training_indices[round_index], training_seeds[round_index],
                          vertex_state if method == 'rl' and isinstance(policy, CandidatePolicy) else None,
                          policy.candidate_settings['ema_beta'] if isinstance(policy, CandidatePolicy) else 0.9)
            after_selection = _observe_silhouette(trainer, val_data, selection_views)
            rgb_counts, rgb_before, rgb_after = _region_stats(observations, after_selection, face_count, 1)
            sil_counts, sil_before, sil_after = _region_stats(observations, after_selection, face_count, 2, contour_radius)
            outcomes = edge_outcomes(event, face_count, rgb_counts, rgb_before, rgb_after,
                                     sil_counts, sil_before, sil_after)
            np.savez_compressed(os.path.join(opt.refine_experiment_output, observations_file),
                                face_id=np.arange(face_count), rgb_mean=means, visible_pixels=counts,
                                area=areas, contour_near_pixels=near_counts,
                                min_contour_distance_pixels=min_distance, status=status,
                                passive=passive, executed_requester=executed, candidate=candidate_mask,
                                candidate_features=candidate_feature_values, request_probability=request_probability,
                                decision_mask=decisions.decided if decisions is not None else np.array([], dtype=bool),
                                decision_face_ids=visible if decisions is not None else np.array([], dtype=np.int64),
                                remaining_budget=decisions.remaining if decisions is not None else np.array([], dtype=np.int64),
                                decision_log_probability=decisions.log_probabilities if decisions is not None else np.array([], dtype=np.float64),
                                rgb_evaluation_pixels=rgb_counts, rgb_before=rgb_before, rgb_after=rgb_after,
                                silhouette_evaluation_pixels=sil_counts, silhouette_before=sil_before, silhouette_after=sil_after)
            metrics = full_frame_metrics(_observe_silhouette(trainer, val_data, evaluation_views),
                                         evaluation_views, shapes)
            branch['rounds'].append({'round': round_index + 1, 'faces_before': face_count,
                                     'faces_after': len(model.triangles), 'visible_candidates': len(visible),
                                     'requested_count': int(np.count_nonzero(decisions.actions)) if decisions is not None else len(order),
                                     'hold_count': int(np.count_nonzero(decisions.decided & ~decisions.actions)) if decisions is not None else len(visible) - len(order),
                                     'executed_request_count': sum(
                                         event['request_status'][int(face)] == 'executed'
                                         for face in order[:event['considered_count']]),
                                     'conflict_count': sum(
                                         event['request_status'][int(face)] == 'conflict'
                                         for face in order[:event['considered_count']]),
                                     'budget_unexecuted_count': sum(
                                         event['request_status'][int(face)] in ('budget', 'budget_not_considered') for face in order),
                                     'round_budget': round_budget, 'actual_added_faces': actual,
                                     'face_observations': observations_file, 'contour_views': contour_views,
                                     'candidate_info': candidate_info,
                                     'edge_outcomes': outcomes,
                                     'cumulative_added_faces': cumulative, 'unfilled_budget': round_budget - actual,
                                     'training_indices': training_indices[round_index],
                                     'training_seed': training_seeds[round_index], 'event': event,
                                     'metrics': metrics})
            _save_json(os.path.join(opt.refine_experiment_output, 'branch_{}.json'.format(branch_name)), branch)
            trainer.log('[INFO] {} seed {} round {}: {} added faces'.format(
                method, random_seed, round_index + 1, actual))
        branch['actual_added_faces'] = cumulative
        branch['unused_total_budget'] = total_budget - cumulative
        branch['final_metrics'] = branch['rounds'][-1]['metrics']
        _save_json(os.path.join(opt.refine_experiment_output, 'branch_{}.json'.format(branch_name)), branch)
        result['branches'].append(branch)

    if any(not torch.equal(value, policy.state_dict()[key]) for key, value in initial_policy.items()):
        raise RuntimeError('Evaluation policy weights changed')
    _save_json(os.path.join(opt.refine_experiment_output, 'multibranch_comparison.json'), result)
    trainer.log('[INFO] multi-branch comparison written to ' + opt.refine_experiment_output)
