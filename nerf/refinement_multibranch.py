"""Compare fixed-policy, high-error and random consecutive refinements."""

import copy
import json
import os

import numpy as np
import torch
from torch import nn
from scipy.ndimage import binary_erosion, distance_transform_edt

from nerf.refinement_experiment import _select_views, _train_branch
from nerf.refinement_multiround import _observe_silhouette
from nerf.refinement_rl import FacePolicy, face_features, file_sha256


def load_trajectory_policy(path, checkpoint_hash, mesh_hash):
    trajectory = torch.load(path, map_location='cpu')
    config = trajectory.get('config', {})
    if config.get('checkpoint_sha256') != checkpoint_hash or config.get('mesh_sha256') != mesh_hash:
        raise ValueError('Policy trajectory does not match the Stage 1 checkpoint and mesh hashes')
    if 'policy' not in trajectory:
        raise ValueError('Trajectory checkpoint does not contain policy weights')
    policy = FacePolicy()
    policy.load_state_dict(trajectory['policy'])
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
    initial = _observe_silhouette(trainer, val_data, evaluation_views)
    shapes = [item[4] for item in initial]
    initial_metrics = full_frame_metrics(initial, evaluation_views, shapes)
    model = trainer.model

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

    branches = [('rl', None), ('high_error', None)] + [('random', seed) for seed in seeds]
    _save_json(os.path.join(opt.refine_experiment_output, 'multibranch_metadata.json'), result)
    opt.refine = False
    for method, random_seed in branches:
        restore()
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
            if method == 'rl' and len(visible):
                visible, features, _ = face_features(vertices, model.triangles.cpu().numpy(), sums, counts)
                with torch.no_grad():
                    scores = policy(torch.from_numpy(features)).numpy()
                order = visible[np.argsort(-scores, kind='stable')]
            elif method == 'high_error':
                order = visible[np.argsort(-(sums[visible] / counts[visible]), kind='stable')]
            elif method == 'random':
                order = random_generator.permutation(visible)
            else:
                order = visible
            round_budget = min(opt.refine_face_budget, total_budget - cumulative)
            model.refine_and_decimate(export_mesh=False, candidate_order=order, max_added_faces=round_budget)
            event = copy.deepcopy(model.refinement_event)
            actual = len(model.triangles) - face_count
            if actual != event['added_faces'] or actual > round_budget:
                raise RuntimeError('Recorded face count does not match the budgeted topology')
            cumulative += actual
            event['global_step'] = trainer.global_step
            event['parent_face_id'] = model.parent_face_id.cpu().numpy().tolist()
            status = np.full(face_count, 'not_visible', dtype='<U24')
            status[visible] = 'budget_not_considered'
            for face in order[:event['considered_count']]:
                status[face] = event['request_status'][int(face)]
            passive = np.zeros(face_count, dtype=bool)
            passive[event['passive_faces']] = True
            executed = np.zeros(face_count, dtype=bool)
            executed[[edge['requester_face'] for edge in event['edge_events']]] = True
            observations_file = 'faces_{}_round_{:04d}.npz'.format(branch_name, round_index + 1)
            np.savez_compressed(os.path.join(opt.refine_experiment_output, observations_file),
                                face_id=np.arange(face_count), rgb_mean=means, visible_pixels=counts,
                                area=areas, contour_near_pixels=near_counts,
                                min_contour_distance_pixels=min_distance, status=status,
                                passive=passive, executed_requester=executed)
            _train_branch(trainer, train_data, training_indices[round_index], training_seeds[round_index])
            metrics = full_frame_metrics(_observe_silhouette(trainer, val_data, evaluation_views),
                                         evaluation_views, shapes)
            branch['rounds'].append({'round': round_index + 1, 'faces_before': face_count,
                                     'faces_after': len(model.triangles), 'visible_candidates': len(visible),
                                     'round_budget': round_budget, 'actual_added_faces': actual,
                                     'face_observations': observations_file, 'contour_views': contour_views,
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
