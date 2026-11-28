"""Compare fixed-policy, high-error and random consecutive refinements."""

import copy
import json
import os

import numpy as np
import torch
from torch import nn
from scipy.ndimage import binary_erosion, distance_transform_edt

from nerf.refinement_experiment import _select_views, _train_branch, _warmup_vertex_ema
from nerf.refinement_multiround import _observe_silhouette, _region_stats, edge_costs, local_rewards
from nerf.refinement_rl import FacePolicy, face_features, file_sha256
from nerf.refinement_candidates import (CandidatePolicy, FEATURE_VERSION, SequentialFaceDecisions, RankedFaceDecisions, CategoricalFaceDecisions,
                                        candidate_features, remap_vertex_ema, visibility_counts)
from nerf.refinement_attribution import edge_outcomes
from nerf.refinement_gnn import SparseFacePolicy, TextureFacePolicy, RankedTextureFacePolicy, RANK_POLICY_VERSION, POLICY_VERSION, TEXTURE_POLICY_VERSION, policy_graph_state
from nerf.refinement_texture import TEXTURE_FEATURE_VERSION, fixed_gradient_regions, texture_face_sums


def probability_distribution(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {'count': 0, 'min': None, 'p25': None, 'median': None, 'p75': None, 'max': None, 'mean': None}
    return {'count': len(values), **dict(zip(('min', 'p25', 'median', 'p75', 'max'),
                                             np.quantile(values, [0, .25, .5, .75, 1]).tolist())),
            'mean': float(values.mean())}


def load_trajectory_policy(path, checkpoint_hash, mesh_hash):
    trajectory = torch.load(path, map_location='cpu')
    config = trajectory.get('config', {})
    if config.get('checkpoint_sha256') != checkpoint_hash or config.get('mesh_sha256') != mesh_hash:
        raise ValueError('Policy trajectory does not match the Stage 1 checkpoint and mesh hashes')
    if 'policy' not in trajectory:
        raise ValueError('Trajectory checkpoint does not contain policy weights')
    kind = config.get('policy_type', 'mlp')
    if kind not in ('mlp', 'gnn'):
        raise ValueError('Unknown policy architecture in trajectory: {}'.format(kind))
    if kind == 'gnn':
        architecture = config.get('policy_architecture')
        if architecture not in (POLICY_VERSION, TEXTURE_POLICY_VERSION, RANK_POLICY_VERSION) or config.get('feature_version') != FEATURE_VERSION:
            raise ValueError('GNN checkpoint architecture or feature version mismatch')
        if architecture in (TEXTURE_POLICY_VERSION, RANK_POLICY_VERSION) and config.get('texture_feature_version') != TEXTURE_FEATURE_VERSION:
            raise ValueError('Texture GNN feature version mismatch; retrain a compatible policy')
        if architecture == RANK_POLICY_VERSION and config.get('action_mode') != 'categorical':
            raise ValueError('Categorical GNN checkpoint action version mismatch')
        if architecture != RANK_POLICY_VERSION and config.get('action_mode', 'bernoulli') != 'bernoulli':
            raise ValueError('GNN checkpoint action distribution mismatch')
        policy = RankedTextureFacePolicy() if architecture == RANK_POLICY_VERSION else TextureFacePolicy() if architecture == TEXTURE_POLICY_VERSION else SparseFacePolicy()
        policy.allow_stop = config.get('allow_stop', False)
    else:
        if config.get('policy_architecture', 'candidate_mlp_v1') != 'candidate_mlp_v1':
            raise ValueError('MLP checkpoint architecture mismatch')
        policy = CandidatePolicy() if config.get('feature_version') == FEATURE_VERSION else FacePolicy()
    if isinstance(policy, CandidatePolicy):
        options = config.get('options', {})
        policy.candidate_settings = {'fraction': options.get('refine_candidate_fraction', 0.5),
                                     'min_area': options.get('refine_min_face_area', 1e-12),
                                     'ema_beta': options.get('refine_gradient_ema_beta', 0.9),
                                     'training_ema_warmup_steps': options.get('refine_gradient_ema_warmup_steps', 0),
                                     'self_loops': options.get('refine_graph_self_loops', False)}
        policy.candidate_settings.update(mode=options.get('refine_candidate_mode', 'area'),
                                         area_fraction=options.get('refine_area_fraction', .4),
                                         texture_fraction=options.get('refine_texture_fraction', .1))
        policy.training_objective = config.get('policy_objective', 'unknown')
        policy.training_reward_weights = config.get('weights')
    try:
        policy.load_state_dict(trajectory['policy'])
    except RuntimeError as error:
        raise ValueError('Policy checkpoint has incompatible feature dimensions') from error
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


def full_frame_metrics(observations, views, expected_shapes, gradient_masks=None):
    per_view = []
    rgb_total = 0.0
    sil_total = 0.0
    pixels_total = 0
    gradient_total = 0.0
    gradient_pixels_total = 0
    if len(observations) != len(views) or len(views) != len(expected_shapes):
        raise ValueError('Evaluation views do not match the fixed pixel set')
    if gradient_masks is not None and len(gradient_masks) != len(views):
        raise ValueError('Gradient masks do not match the fixed evaluation views')
    for index, (view, observation, shape) in enumerate(zip(views, observations, expected_shapes)):
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
        row = {'view_id': int(view), 'pixels': pixels,
               'rgb_error': rgb_sum / pixels, 'silhouette_error': sil_sum / pixels}
        if gradient_masks is not None:
            if len(observation) != 10:
                raise ValueError('Gradient evaluation requires Sobel error and target-only mask')
            mask = np.asarray(gradient_masks[index], dtype=bool).reshape(-1)
            if len(mask) != pixels or not np.array_equal(mask, observation[9]):
                raise ValueError('Evaluation target foreground mask changed between branches')
            # Do not silently drop a different set of pixels for a nonfinite rendering.
            if np.any(mask & (~observation[7] | ~np.isfinite(observation[6]))):
                raise ValueError('Nonfinite rendered RGB in the fixed gradient evaluation region')
            count = int(mask.sum())
            value = float(np.sum(observation[6][mask], dtype=np.float64))
            gradient_total += value
            gradient_pixels_total += count
            row.update(gradient_error=value / count if count else None, gradient_valid_pixels=count)
        per_view.append(row)
    if not pixels_total:
        raise ValueError('Evaluation pixel set is empty')
    overall = {'pixels': pixels_total, 'rgb_error': rgb_total / pixels_total,
               'silhouette_error': sil_total / pixels_total}
    if gradient_masks is not None:
        overall.update(gradient_error=gradient_total / gradient_pixels_total if gradient_pixels_total else None,
                       gradient_valid_pixels=gradient_pixels_total)
    return {'per_view': per_view, 'overall': overall}


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
    for observation in observations:
        face_ids, losses, _, target_alpha, shape = observation[:5]
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
    diagnose_gradients = getattr(opt, 'refine_face_gradient_diagnostics', False)
    loss_diagnostics = None
    if getattr(opt, 'refine_loss_gradient_diagnostics', False):
        loss_diagnostics = {'interval': getattr(opt, 'refine_loss_gradient_interval', 10),
                            'max_samples': getattr(opt, 'refine_loss_gradient_max_samples', 5)}
        if loss_diagnostics['interval'] < 1 or loss_diagnostics['max_samples'] < 1:
            raise ValueError('Loss gradient diagnostic interval and maximum must be positive')
    policy_mode = getattr(opt, 'refine_multibranch_policy_mode', 'threshold')
    if policy_mode not in ('threshold', 'rank'):
        raise ValueError('Unknown multi-branch policy evaluation mode')
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
    if policy_mode == 'rank' and not isinstance(policy, CandidatePolicy):
        raise ValueError('Budget ranking requires a current MLP or GNN candidate policy checkpoint')
    primary_policy = policy
    comparison_path = getattr(opt, 'refine_multibranch_compare_policy', '')
    comparison_policy = load_trajectory_policy(comparison_path, checkpoint_hash, mesh_hash) if comparison_path else None
    if policy_mode != 'rank' and (isinstance(policy, RankedTextureFacePolicy) or isinstance(comparison_policy, RankedTextureFacePolicy)):
        raise ValueError('Categorical checkpoint evaluation requires --refine_multibranch_policy_mode rank; it uses categorical greedy actions')
    if comparison_policy is not None:
        if (not isinstance(policy, CandidatePolicy) or not isinstance(comparison_policy, CandidatePolicy) or
                type(policy) == type(comparison_policy)):
            raise ValueError('Policy comparison requires two different current policy architectures')
        for key in ('min_area', 'ema_beta'):
            if policy.candidate_settings[key] != comparison_policy.candidate_settings[key]:
                raise ValueError('MLP and GNN candidate/EMA settings must match: ' + key)
    if getattr(opt, 'refine_observation_graph', False) and not isinstance(policy, CandidatePolicy):
        raise ValueError('Observation graph recording requires the current CandidatePolicy checkpoint')
    initial_policy = copy.deepcopy(policy.state_dict())
    use_texture_observation = (isinstance(policy, TextureFacePolicy) or isinstance(comparison_policy, TextureFacePolicy) or
                               getattr(policy, 'candidate_settings', {}).get('mode') == 'hybrid' or
                               getattr(comparison_policy, 'candidate_settings', {}).get('mode') == 'hybrid' or
                               getattr(opt, 'refine_multiround_grad_weight', 0.) > 0)
    collect_selection_texture = use_texture_observation or diagnose_gradients
    hybrid_reference = None
    if diagnose_gradients:
        for source in (primary_policy, comparison_policy):
            config = getattr(source, 'candidate_settings', {})
            if config.get('mode') == 'hybrid':
                hybrid_reference = {'area_fraction': config['area_fraction'], 'texture_fraction': config['texture_fraction'],
                                    'min_area': config['min_area'], 'source': 'hybrid policy checkpoint'}
                break
        if hybrid_reference is None:
            hybrid_reference = {'area_fraction': getattr(opt, 'refine_area_fraction', .4),
                                'texture_fraction': getattr(opt, 'refine_texture_fraction', .1),
                                'min_area': getattr(opt, 'refine_min_face_area', 1e-12), 'source': 'command line defaults'}
    comparison_initial = copy.deepcopy(comparison_policy.state_dict()) if comparison_policy is not None else None
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
    initial = _observe_silhouette(trainer, val_data, evaluation_views, texture=True, texture_target_mask=True)
    shapes = [item[4] for item in initial]
    gradient_masks = [item[9].copy() for item in initial]
    initial_metrics = full_frame_metrics(initial, evaluation_views, shapes, gradient_masks)
    gradient_mask_path = os.path.join(opt.refine_experiment_output, 'evaluation_gradient_masks.npz')
    np.savez_compressed(gradient_mask_path, **{'view_{}'.format(view): mask.reshape(shape)
                                               for view, mask, shape in zip(evaluation_views, gradient_masks, shapes)})

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
              'gradient_metric': {'definition': 'mean SmoothL1 of Sobel / 8 derivatives over RGB and x/y, then pixels',
                                  'color_space': getattr(opt, 'color_space', 'srgb'),
                                  'background': 'fixed white',
                                  'mask': 'fixed target-only: finite RGB and alpha >= 0.999 across entire 3x3 stencil; independent of mesh coverage',
                                  'boundary': 'replicate convolution padding; boundary centers excluded by stencil',
                                  'normalization': 'Sobel / 8; average six channel/direction errors per pixel, then pixel-weighted mean across views',
                                  'nonfinite_rendering': 'raise error instead of changing the fixed pixel set',
                                  'mask_file': os.path.abspath(gradient_mask_path),
                                  'mask_sha256': file_sha256(gradient_mask_path)},
              'initial_faces': len(old_triangles), 'initial_global_step': start_step,
              'checkpoint_global_step': checkpoint_step, 'ema_warmup': ema_warmup,
              'initial_metrics': initial_metrics, 'rounds': opt.refine_multibranch_rounds,
              'per_round_budget': opt.refine_face_budget, 'total_budget': total_budget,
              'optimization_steps': opt.refine_experiment_steps,
              'selection_region_reward_weights': [getattr(opt, 'refine_multiround_sil_weight', 1.0),
                                                  getattr(opt, 'refine_multiround_rgb_weight', 1.0),
                                                  getattr(opt, 'refine_multiround_face_cost', 0.001),
                                                  getattr(opt, 'refine_multiround_grad_weight', 0.)],
              'selection_region_reward_note': 'local reward on pre-split selection-view regions; no policy updates or lineage return in evaluation',
              'training_indices': training_indices, 'training_seeds': training_seeds,
              'policy_evaluation_mode': policy_mode,
              'primary_policy_action_mode': 'categorical_greedy' if isinstance(primary_policy, RankedTextureFacePolicy) else policy_mode,
              'face_gradient_diagnostics': {'enabled': diagnose_gradients, 'hybrid_reference': hybrid_reference,
                                            'views': selection_views, 'dataset': 'validation selection views'},
              'rl_selection': 'descending shared policy score among visible faces',
              'high_error_selection': 'descending average RGB error among visible faces',
              'random_selection': 'seeded permutation of visible faces', 'branches': []}
    if comparison_policy is not None:
        result['comparison_policy_checkpoint'] = os.path.abspath(comparison_path)
        result['comparison_policy_checkpoint_sha256'] = file_sha256(comparison_path)
    result['face_observations'] = ('NPZ arrays indexed by pre-update face ID; contour distance is the minimum '
                                   'pixel distance to a ground-truth alpha > 0.5 boundary over selection views')
    result['contour_radius_pixels'] = contour_radius
    result['rl_policy_feature_version'] = FEATURE_VERSION if isinstance(policy, CandidatePolicy) else 'legacy_three_features'
    result['rl_vertex_ema_initialization'] = 'zeros on starting Stage 1 mesh followed by recorded warmup optimization steps'
    if isinstance(policy, CandidatePolicy):
        result['rl_selection'] = ('categorical greedy: highest current candidate/stop logit at each budget' if isinstance(policy, RankedTextureFacePolicy) else
                                  'highest current probability; rescore all untried candidates when budget changes; always request'
                                  if policy_mode == 'rank' else
                                  'area order; sequential sigmoid >= 0.5 decisions using current remaining budget')
        result['rl_candidate_settings'] = policy.candidate_settings
        result['rl_training_objective'] = policy.training_objective

    branches = [('rl', None)] + ([('rl_compare', None)] if comparison_policy is not None else []) + [('high_error', None)] + [('random', seed) for seed in seeds]
    _save_json(os.path.join(opt.refine_experiment_output, 'multibranch_metadata.json'), result)
    opt.refine = False
    for method, random_seed in branches:
        policy = comparison_policy if method == 'rl_compare' else primary_policy
        is_rl = method in ('rl', 'rl_compare')
        restore()
        vertex_state = {'ema': warmup_state['ema'].copy(), 'seen': warmup_state['seen'].copy()}
        branch = {'method': method, 'random_seed': random_seed,
                  'policy_evaluation_mode': ('categorical_greedy' if isinstance(policy, RankedTextureFacePolicy) else policy_mode if isinstance(policy, CandidatePolicy) else 'legacy_rank') if is_rl else None,
                  'policy_type': ('gnn' if isinstance(policy, SparseFacePolicy) else 'mlp') if is_rl else None,
                  'policy_architecture': (RANK_POLICY_VERSION if isinstance(policy, RankedTextureFacePolicy) else TEXTURE_POLICY_VERSION if isinstance(policy, TextureFacePolicy) else POLICY_VERSION if isinstance(policy, SparseFacePolicy) else 'candidate_mlp_v1') if is_rl else None,
                  'candidate_rule': policy.candidate_settings if is_rl and isinstance(policy, CandidatePolicy) else ('all visible faces' if not is_rl else 'legacy visible faces'),
                  'policy_training_reward_weights': getattr(policy, 'training_reward_weights', None) if is_rl else None,
                  'initial_faces': len(model.triangles), 'initial_global_step': trainer.global_step,
                  'initial_metrics': initial_metrics, 'rounds': []}
        cumulative = 0
        random_generator = np.random.RandomState(random_seed) if method == 'random' else None
        branch_name = '{}_{}'.format(method, random_seed) if random_seed is not None else method
        for round_index in range(opt.refine_multibranch_rounds):
            face_count = len(model.triangles)
            observations = _observe_silhouette(trainer, val_data, selection_views, texture=collect_selection_texture)
            vertices = (model.vertices + model.vertices_offsets.detach()).cpu().numpy()
            faces = model.triangles.cpu().numpy()
            sums, counts, means, areas, near_counts, min_distance, contour_views = selection_face_observations(
                observations, face_count, vertices, faces, contour_radius)
            model.triangles_errors = torch.tensor(sums, dtype=torch.float32, device=trainer.device)
            model.triangles_errors_cnt = torch.tensor(counts, dtype=torch.float32, device=trainer.device)
            visible = np.flatnonzero(counts > 0)
            candidate_info = None
            decisions = None
            graph_files = None
            candidate_mask = np.zeros(face_count, dtype=bool)
            candidate_feature_values = np.full((face_count, 9), np.nan, dtype=np.float32)
            request_probability = np.full(face_count, np.nan, dtype=np.float32)
            initial_probability = np.full(face_count, np.nan, dtype=np.float32)
            round_budget = min(opt.refine_face_budget, total_budget - cumulative)
            if is_rl and (len(visible) or isinstance(policy, CandidatePolicy)):
                if isinstance(policy, CandidatePolicy):
                    _, _, view_counts = visibility_counts(observations, face_count)
                    settings = policy.candidate_settings
                    candidate_kw = dict(candidate_mode=settings['mode'], area_fraction=settings['area_fraction'],
                                        texture_fraction=settings['texture_fraction'])
                    if settings['mode'] == 'hybrid':
                        _, tex_error, tex_pixels = texture_face_sums(observations, face_count)
                        candidate_kw.update(texture_error_sum=tex_error, texture_pixels=tex_pixels)
                    graph_state = None
                    if getattr(opt, 'refine_observation_graph', False) or isinstance(policy, SparseFacePolicy):
                        from nerf.refinement_graph import build_face_graph, save_face_graph
                        graph, visible, features, candidate_info = build_face_graph(
                            vertices, faces, sums, counts, view_counts, len(selection_views),
                            vertex_state['ema'], vertex_state['seen'], round_budget, round_budget,
                            policy.candidate_settings['fraction'], policy.candidate_settings['min_area'],
                            policy.candidate_settings['self_loops'] if isinstance(policy, SparseFacePolicy) else getattr(opt, 'refine_graph_self_loops', False), **candidate_kw)
                        if isinstance(policy, SparseFacePolicy):
                            if isinstance(policy, TextureFacePolicy):
                                from nerf.refinement_graph import add_texture_features
                                add_texture_features(graph, observations)
                            graph_state = policy_graph_state(graph)
                        if getattr(opt, 'refine_observation_graph', False):
                            graph['metadata']['policy'] = 'shared ' + ('gnn' if isinstance(policy, SparseFacePolicy) else 'mlp') + ' policy'
                            graph_files = {phase: os.path.abspath(os.path.join(opt.refine_experiment_output,
                                           'graph_{}_round_{:04d}_{}.npz'.format(branch_name, round_index + 1, phase)))
                                           for phase in ('before', 'after')}
                            save_face_graph(graph_files['before'], graph)
                    else:
                        visible, features, candidate_info = candidate_features(
                            vertices, faces, sums, counts, view_counts, len(selection_views),
                            vertex_state['ema'], vertex_state['seen'], 1.0 if round_budget else 0.0,
                            policy.candidate_settings['fraction'], policy.candidate_settings['min_area'], **candidate_kw)
                    candidate_mask[visible] = True
                    if isinstance(policy, RankedTextureFacePolicy):
                        decisions = CategoricalFaceDecisions(policy, visible, features, round_budget, graph_state,
                                                             deterministic=True, allow_stop=policy.allow_stop)
                    elif policy_mode == 'rank':
                        decisions = RankedFaceDecisions(policy, visible, features, round_budget, graph=graph_state)
                        initial_probability[visible] = decisions.initial_probabilities
                    else:
                        decisions = SequentialFaceDecisions(policy, visible, features, round_budget, deterministic=True, graph=graph_state)
                        initial_probability[visible] = decisions.score_all(round_budget)
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
                if isinstance(decisions, CategoricalFaceDecisions):
                    initial_probability[visible] = decisions.initial_probabilities
            event = copy.deepcopy(model.refinement_event)
            actual = len(model.triangles) - face_count
            if actual != event['added_faces'] or actual > round_budget:
                raise RuntimeError('Recorded face count does not match the budgeted topology')
            cumulative += actual
            event['global_step'] = trainer.global_step
            event['parent_face_id'] = model.parent_face_id.cpu().numpy().tolist()
            if is_rl and isinstance(policy, CandidatePolicy):
                vertex_state['ema'], vertex_state['seen'], vertex_mapping = remap_vertex_ema(
                    vertex_state['ema'], vertex_state['seen'], event, len(model.vertices))
                event['vertex_mapping'] = vertex_mapping.tolist()
            status = np.full(face_count, 'not_visible', dtype='<U24')
            status[counts > 0] = 'not_candidate' if is_rl and isinstance(policy, CandidatePolicy) else 'budget_not_considered'
            status[visible] = 'hold' if is_rl and isinstance(policy, CandidatePolicy) else 'budget_not_considered'
            status[order] = 'budget_not_considered'
            considered_faces = event.get('considered_faces', order[:event['considered_count']])
            for face in order:
                status[face] = event['request_status'][int(face)]
            passive = np.zeros(face_count, dtype=bool)
            passive[event['passive_faces']] = True
            executed = np.zeros(face_count, dtype=bool)
            executed[[edge['requester_face'] for edge in event['edge_events']]] = True
            observations_file = 'faces_{}_round_{:04d}.npz'.format(branch_name, round_index + 1)
            after_split = (_observe_silhouette(trainer, val_data, selection_views, texture=True)
                           if diagnose_gradients else None)
            training_losses = _train_branch(trainer, train_data, training_indices[round_index], training_seeds[round_index],
                          vertex_state if is_rl and isinstance(policy, CandidatePolicy) else None,
                          policy.candidate_settings['ema_beta'] if isinstance(policy, CandidatePolicy) else 0.9,
                          gradient_diagnostics=loss_diagnostics)
            loss_diagnostic_report = training_losses.pop('loss_gradient_diagnostics', None)
            after_selection = _observe_silhouette(trainer, val_data, selection_views, texture=collect_selection_texture)
            if graph_files is not None:
                post_sums, post_pixels, post_views = visibility_counts(after_selection, len(model.triangles))
                post_kw = dict(candidate_mode=settings['mode'], area_fraction=settings['area_fraction'], texture_fraction=settings['texture_fraction'])
                if settings['mode'] == 'hybrid':
                    _, post_texture, post_texture_pixels = texture_face_sums(after_selection, len(model.triangles))
                    post_kw.update(texture_error_sum=post_texture, texture_pixels=post_texture_pixels)
                post_graph, _, _, _ = build_face_graph(
                    (model.vertices + model.vertices_offsets.detach()).cpu().numpy(), model.triangles.cpu().numpy(),
                    post_sums, post_pixels, post_views, len(selection_views), vertex_state['ema'], vertex_state['seen'],
                    round_budget - actual, round_budget, policy.candidate_settings['fraction'],
                    policy.candidate_settings['min_area'], policy.candidate_settings['self_loops'] if isinstance(policy, SparseFacePolicy) else getattr(opt, 'refine_graph_self_loops', False), **post_kw)
                if isinstance(policy, TextureFacePolicy):
                    add_texture_features(post_graph, after_selection)
                post_graph['metadata']['policy'] = 'shared ' + ('gnn' if isinstance(policy, SparseFacePolicy) else 'mlp') + ' policy'
                save_face_graph(graph_files['after'], post_graph,
                                model.parent_face_id.cpu().numpy())
            rgb_counts, rgb_before, rgb_after = _region_stats(observations, after_selection, face_count, 1)
            sil_counts, sil_before, sil_after = _region_stats(observations, after_selection, face_count, 2, contour_radius)
            if use_texture_observation:
                grad_counts, grad_before, grad_after = fixed_gradient_regions(observations, after_selection, face_count)
                texture_sum, texture_error_sum, texture_counts = texture_face_sums(observations, face_count)
            else:
                grad_counts = np.zeros(face_count, dtype=np.int64)
                grad_before = grad_after = np.full(face_count, np.nan)
                texture_sum = texture_error_sum = np.zeros(face_count)
                texture_counts = np.zeros(face_count, dtype=np.int64)
            diagnostic_files = None
            if diagnose_gradients:
                from nerf.refinement_gradient_diagnostics import gradient_diagnostics, three_phase_gradient_diagnostics
                _, diagnostic_texture_sum, diagnostic_texture_counts = texture_face_sums(observations, face_count)
                _, _, diagnostic_views = visibility_counts(observations, face_count)
                diagnostic_ids, _, diagnostic_candidate_info = candidate_features(
                    vertices, faces, sums, counts, diagnostic_views, len(selection_views),
                    np.zeros(len(vertices)), np.zeros(len(vertices), dtype=bool),
                    1.0 if round_budget else 0.0, min_area=hybrid_reference['min_area'],
                    candidate_mode='hybrid', area_fraction=hybrid_reference['area_fraction'],
                    texture_fraction=hybrid_reference['texture_fraction'],
                    texture_error_sum=diagnostic_texture_sum, texture_pixels=diagnostic_texture_counts)
                hybrid_mask = np.zeros(face_count, dtype=bool)
                hybrid_mask[diagnostic_ids] = True
                valid_faces = (counts > 0) & np.isfinite(sums) & np.isfinite(areas) & (areas > hybrid_reference['min_area'])
                diagnostic, diagnostic_arrays = gradient_diagnostics(
                    observations, after_selection, face_count, valid_faces, hybrid_mask,
                    executed, passive, event['parent_face_id'], selection_views, evaluation_views)
                diagnostic['three_phase'], three_phase_arrays = three_phase_gradient_diagnostics(
                    observations, after_split, after_selection, face_count, valid_faces, hybrid_mask,
                    executed, passive, event['parent_face_id'], selection_views, evaluation_views)
                diagnostic_arrays.update(three_phase_arrays)
                diagnostic['method'] = method
                diagnostic['random_seed'] = random_seed
                diagnostic['round'] = round_index + 1
                diagnostic['color_space'] = getattr(opt, 'color_space', 'srgb')
                diagnostic['hybrid_reference'] = hybrid_reference
                diagnostic['hybrid_candidate_info'] = diagnostic_candidate_info
                diagnostic['actual_candidate_rule'] = branch['candidate_rule']
                diagnostic['actual_added_faces'] = actual
                diagnostic_arrays['actual_policy_candidate'] = candidate_mask if is_rl else (counts > 0)
                diagnostic_arrays['parent_face_id'] = np.asarray(event['parent_face_id'], dtype=np.int64)
                diagnostic_files = {'summary': 'gradient_diagnostics_{}_round_{:04d}.json'.format(branch_name, round_index + 1),
                                    'faces': 'gradient_diagnostics_{}_round_{:04d}.npz'.format(branch_name, round_index + 1)}
                _save_json(os.path.join(opt.refine_experiment_output, diagnostic_files['summary']), diagnostic)
                np.savez_compressed(os.path.join(opt.refine_experiment_output, diagnostic_files['faces']), **diagnostic_arrays)
            outcomes = edge_outcomes(event, face_count, rgb_counts, rgb_before, rgb_after,
                                     sil_counts, sil_before, sil_after, grad_counts, grad_before, grad_after)
            face_cost = edge_costs(event, face_count)
            face_reward = local_rewards(rgb_before, rgb_after, sil_before, sil_after, rgb_counts, face_cost,
                                        result['selection_region_reward_weights'], grad_counts, grad_before, grad_after,
                                        sil_counts=sil_counts)
            np.savez_compressed(os.path.join(opt.refine_experiment_output, observations_file),
                                face_id=np.arange(face_count), rgb_mean=means, visible_pixels=counts,
                                area=areas, contour_near_pixels=near_counts,
                                min_contour_distance_pixels=min_distance, status=status,
                                passive=passive, executed_requester=executed, candidate=candidate_mask,
                                candidate_features=candidate_feature_values, request_probability=request_probability,
                                initial_probability=initial_probability,
                                decision_order=np.asarray(considered_faces, dtype=np.int64),
                                decision_mask=decisions.decided if decisions is not None else np.array([], dtype=bool),
                                decision_face_ids=visible if decisions is not None else np.array([], dtype=np.int64),
                                remaining_budget=decisions.remaining if decisions is not None else np.array([], dtype=np.int64),
                                decision_log_probability=decisions.log_probabilities if decisions is not None else np.array([], dtype=np.float64),
                                rgb_evaluation_pixels=rgb_counts, rgb_before=rgb_before, rgb_after=rgb_after,
                                silhouette_evaluation_pixels=sil_counts, silhouette_before=sil_before, silhouette_after=sil_after,
                                face_cost=face_cost, selection_region_local_reward=face_reward,
                                gradient_pixels=grad_counts, gradient_before=grad_before, gradient_after=grad_after,
                                texture_strength_sum=texture_sum, texture_error_sum=texture_error_sum,
                                texture_pixels=texture_counts)
            metrics = full_frame_metrics(_observe_silhouette(trainer, val_data, evaluation_views,
                                                              texture=True, texture_target_mask=True),
                                         evaluation_views, shapes, gradient_masks)
            branch['rounds'].append({'round': round_index + 1, 'faces_before': face_count,
                                     'policy_evaluation_mode': ('categorical_greedy' if isinstance(policy, RankedTextureFacePolicy) else policy_mode if isinstance(policy, CandidatePolicy) else 'legacy_rank') if is_rl else None,
                                     'stop_event': getattr(decisions, 'stop_event', None),
                                     'stopped_count': sum(event['request_status'][int(face)] == 'stopped' for face in order),
                                     'initial_candidate_probability_distribution': probability_distribution(initial_probability) if decisions is not None else None,
                                     'attempted_probability_distribution': probability_distribution(decisions.probabilities[decisions.decided]) if decisions is not None else None,
                                     'faces_after': len(model.triangles), 'visible_candidates': len(visible),
                                     'considered_count': event['considered_count'],
                                     'requested_count': int(np.count_nonzero(decisions.actions)) if decisions is not None else len(order),
                                     'hold_count': int(np.count_nonzero(decisions.decided & ~decisions.actions)) if decisions is not None else len(visible) - len(order),
                                     'executed_request_count': sum(
                                         event['request_status'][int(face)] == 'executed'
                                         for face in considered_faces),
                                     'conflict_count': sum(
                                         event['request_status'][int(face)] == 'conflict'
                                         for face in considered_faces),
                                     'already_split_count': sum(event['request_status'][int(face)] == 'already_split' for face in considered_faces),
                                     'budget_rejected_count': sum(event['request_status'][int(face)] == 'budget' for face in considered_faces),
                                     'budget_not_considered_count': sum(event['request_status'][int(face)] == 'budget_not_considered' for face in order),
                                     'budget_unexecuted_count': sum(
                                         event['request_status'][int(face)] in ('budget', 'budget_not_considered') for face in order),
                                     'round_budget': round_budget, 'actual_added_faces': actual,
                                     'face_observations': observations_file, 'contour_views': contour_views,
                                     'candidate_info': candidate_info,
                                     'observation_graphs': graph_files,
                                     'face_gradient_diagnostics': diagnostic_files,
                                     'edge_outcomes': outcomes,
                                     'cumulative_added_faces': cumulative, 'unfilled_budget': round_budget - actual,
                                     'training_indices': training_indices[round_index],
                                     'training_seed': training_seeds[round_index], 'event': event,
                                     'training_losses': training_losses,
                                     'metrics': metrics})
            if loss_diagnostic_report is not None:
                diagnostic_file = 'loss_gradient_diagnostics_{}_round_{:04d}.json'.format(branch_name, round_index + 1)
                loss_diagnostic_report.update(method=method, branch=branch_name, round=round_index + 1,
                                              views='current training images only; independent evaluation views excluded')
                _save_json(os.path.join(opt.refine_experiment_output, diagnostic_file), loss_diagnostic_report)
                branch['rounds'][-1]['loss_gradient_diagnostics'] = {'file': diagnostic_file,
                                                                     'summary': loss_diagnostic_report['summary']}
            _save_json(os.path.join(opt.refine_experiment_output, 'branch_{}.json'.format(branch_name)), branch)
            trainer.log('[INFO] {} seed {} round {}: {} added faces'.format(
                method, random_seed, round_index + 1, actual))
        branch['actual_added_faces'] = cumulative
        branch['unused_total_budget'] = total_budget - cumulative
        branch['final_metrics'] = branch['rounds'][-1]['metrics']
        _save_json(os.path.join(opt.refine_experiment_output, 'branch_{}.json'.format(branch_name)), branch)
        result['branches'].append(branch)

    if any(not torch.equal(value, primary_policy.state_dict()[key]) for key, value in initial_policy.items()):
        raise RuntimeError('Evaluation policy weights changed')
    if comparison_policy is not None and any(not torch.equal(value, comparison_policy.state_dict()[key]) for key, value in comparison_initial.items()):
        raise RuntimeError('Comparison policy weights changed')
    _save_json(os.path.join(opt.refine_experiment_output, 'multibranch_comparison.json'), result)
    trainer.log('[INFO] multi-branch comparison written to ' + opt.refine_experiment_output)
