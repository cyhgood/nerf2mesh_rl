#!/usr/bin/env python3
"""Summarize existing face-credit training and multibranch comparison results.

Only load trajectory checkpoints from a trusted source. This script reads result
files and writes a compact JSON report to stdout; it does not change the results.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch


def load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:  # PyTorch versions used with Python 3.9
        return torch.load(path, map_location='cpu')


def array(values, dtype=None):
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=dtype)


def distribution(values):
    values = array(values, np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return {'count': 0}
    q = np.quantile(values, [0, .1, .25, .5, .75, .9, .95, .99, 1])
    return {'count': int(len(values)), 'mean': float(values.mean()),
            'sum': float(values.sum()),
            'quantiles': dict(zip(['min', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'max'],
                                  map(float, q)))}


def concentration(values):
    values = np.abs(array(values, np.float64).reshape(-1))
    values = values[np.isfinite(values)]
    if not len(values) or values.sum() == 0:
        return {'count': int(len(values)), 'top_1pct_absolute_share': None,
                'top_10pct_absolute_share': None}
    values.sort()
    return {'count': int(len(values)),
            'top_1pct_absolute_share': float(values[-max(1, int(np.ceil(len(values) * .01))):].sum() / values.sum()),
            'top_10pct_absolute_share': float(values[-max(1, int(np.ceil(len(values) * .10))):].sum() / values.sum())}


def execution_faces(event):
    edges = event.get('edge_events', [])
    requesters = {int(edge['requester_face']) for edge in edges}
    affected = {int(face) for edge in edges for face in edge.get('affected_faces', [])}
    return requesters, affected


def overlap(left, right):
    union = left | right
    return {'left_count': len(left), 'right_count': len(right), 'intersection': len(left & right),
            'jaccard': len(left & right) / len(union) if union else None}


def branch_lineages(branch, initial_faces):
    roots = np.arange(initial_faces, dtype=np.int64)
    result = []
    for item in branch.get('rounds', []):
        event = item['event']
        requesters, affected = execution_faces(event)
        for face in requesters | affected:
            if face < 0 or face >= len(roots):
                raise ValueError('Face index outside current branch topology')
        parent = array(event['parent_face_id'], np.int64)
        if np.any(parent < 0) or np.any(parent >= len(roots)):
            raise ValueError('Invalid parent_face_id in comparison results')
        result.append({'round': int(item['round']),
                       'executed_requester_roots': set(map(int, roots[list(requesters)])),
                       'affected_roots': set(map(int, roots[list(affected)])),
                       'executed_requesters': len(requesters), 'affected_faces': len(affected),
                       'actual_added_faces': int(item['actual_added_faces'])})
        roots = roots[parent]
    return result


def compare_branches(data):
    branches = data['branches']
    reference = next((branch for branch in branches if branch.get('method') == 'rl'), None)
    if reference is None:
        raise ValueError('Comparison contains no RL branch')
    initial_faces = int(data['initial_faces'])
    rl = branch_lineages(reference, initial_faces)
    compared = []
    for branch in branches:
        if branch is reference:
            continue
        other = branch_lineages(branch, initial_faces)
        rounds = []
        for left, right in zip(rl, other):
            if left['round'] != right['round']:
                raise ValueError('Comparison branch round numbers differ')
            rounds.append({'round': left['round'],
                           'requester_initial_ancestor_overlap': overlap(left['executed_requester_roots'], right['executed_requester_roots']),
                           'affected_initial_ancestor_overlap': overlap(left['affected_roots'], right['affected_roots']),
                           'rl_actual_added_faces': left['actual_added_faces'],
                           'other_actual_added_faces': right['actual_added_faces']})
        if len(rl) != len(other):
            raise ValueError('Comparison branches contain different numbers of rounds')
        compared.append({'method': branch['method'], 'random_seed': branch.get('random_seed'),
                         'final_metrics': branch.get('final_metrics'), 'rounds': rounds})
    return {'initial_faces': initial_faces, 'comparisons': compared,
            'rl_final_metrics': reference.get('final_metrics'),
            'interpretation': 'Round 1 indices are identical; subsequent overlaps compare initial-mesh ancestors, not current face IDs. Multiple descendants collapse to one ancestor.',
            'feature_limitation': 'Comparison logs do not contain per-face RGB, area, visibility counts or silhouettes, and later branch meshes are not saved.'}


def training_rounds(checkpoint):
    transitions = list(checkpoint.get('history', [])) + list(checkpoint.get('transitions', []))
    weights = checkpoint.get('config', {}).get('weights')
    if not isinstance(weights, (list, tuple)) or len(weights) != 3:
        raise ValueError('Trajectory configuration does not contain three reward weights')
    reports = []
    for t, transition in enumerate(transitions):
        event = transition['event']
        requesters, affected = execution_faces(event)
        total_faces = int(transition['faces_before'])
        visible = array(transition['visible'], np.int64)
        features = array(transition['features'], np.float64)
        scales = array(transition['feature_scales'], np.float64)
        if features.shape != (len(visible), 3) or scales.shape != (3,):
            raise ValueError('Unexpected training feature dimensions')
        positions = np.full(total_faces, -1, dtype=np.int64)
        positions[visible] = np.arange(len(visible))
        reward = array(transition['reward'], np.float64)
        returns = array(transition.get('lineage_return', np.full(total_faces, np.nan)), np.float64)
        counts = array(transition['reward_pixels'], np.int64)
        sil_count = array(transition['silhouette_pixels'], np.int64)
        costs = array(transition['costs'], np.float64)
        sil_delta = array(transition['sil_before'], np.float64) - array(transition['sil_after'], np.float64)
        rgb_delta = array(transition['rgb_before'], np.float64) - array(transition['rgb_after'], np.float64)

        def group(face_set):
            ids = np.array(sorted(face_set), dtype=np.int64)
            if np.any(ids < 0) or np.any(ids >= total_faces):
                raise ValueError('Execution face index outside training topology')
            rows = positions[ids]
            rows = rows[rows >= 0]
            decoded = np.expm1(features[rows]) * scales
            extra = np.maximum(sil_count[ids] - counts[ids], 0)
            return {'face_count': len(ids), 'visible_in_selection': len(rows),
                    'silhouette_improvement': distribution(sil_delta[ids]),
                    'rgb_improvement': distribution(rgb_delta[ids]),
                    'actual_face_cost': distribution(costs[ids]),
                    'weighted_silhouette': distribution(weights[0] * sil_delta[ids]),
                    'weighted_rgb': distribution(weights[1] * rgb_delta[ids]),
                    'weighted_face_cost': distribution(-weights[2] * costs[ids]),
                    'local_reward': distribution(reward[ids]), 'lineage_return': distribution(returns[ids]),
                    'return_absolute_concentration': concentration(returns[ids]),
                    'selection_rgb_mean_approx': distribution(decoded[:, 0]),
                    'selection_pixels_approx': distribution(decoded[:, 1]),
                    'face_area_approx': distribution(decoded[:, 2]),
                    'feature_clipped_counts': (features[rows] >= 5 - 1e-6).sum(axis=0).tolist(),
                    'faces_with_uncovered_target_pixels_assigned': int(np.count_nonzero(extra)),
                    'assigned_uncovered_target_pixels': int(extra.sum())}

        report = {'round': int(transition['round']), 'faces_before': total_faces,
                  'policy_loss': transition.get('policy_loss'), 'executed_requesters': group(requesters),
                  'affected_including_passive': group(affected),
                  'passively_split': group(set(event.get('passive_faces', []))),
                  'all_faces_return_absolute_concentration': concentration(returns),
                  'valid_local_reward_count': int(np.isfinite(reward).sum()),
                  'valid_local_but_invalid_return_count': int((np.isfinite(reward) & ~np.isfinite(returns)).sum())}
        if t + 1 < len(transitions):
            next_transition = transitions[t + 1]
            parent = array(transition['parent_face_id'], np.int64)
            areas = array(transition['child_areas'], np.float64)
            next_returns = array(next_transition.get('lineage_return', np.full(len(parent), np.nan)), np.float64)
            if len(parent) == len(next_returns) == len(areas) and len(parent) and parent.max() < total_faces:
                invalid = np.bincount(parent, weights=areas * ~np.isfinite(next_returns), minlength=total_faces) > 0
                report['parents_with_valid_local_and_invalid_next_descendant_including_possible_segment_boundaries'] = int((invalid & np.isfinite(reward)).sum())
                lost = np.isfinite(reward) & ~np.isfinite(returns)
                report['invalid_parent_returns_with_invalid_next_child'] = int((invalid & lost).sum())
                next_counts = array(next_transition['reward_pixels'], np.int64)
                if len(next_counts) == len(parent):
                    unobserved = np.bincount(parent, weights=areas * (next_counts == 0), minlength=total_faces) > 0
                    report['invalid_parent_returns_with_unobserved_next_child'] = int((unobserved & lost).sum())
            else:
                report['descendant_check'] = 'unavailable: missing or incompatible parent mapping'
        # This is an objective-term proxy, not a parameter-gradient norm.
        if 'selection_order' in transition and 'considered_count' in transition:
            order = array(transition['selection_order'], np.int64)
            probabilities = array(transition['probabilities'], np.float64)[order]
            denominator = np.cumsum(probabilities[::-1])[::-1]
            count = int(transition['considered_count'])
            with np.errstate(divide='ignore', invalid='ignore'):
                log_prob = np.log(probabilities[:count]) - np.log(denominator[:count])
            score = returns[visible[order[:count]]] * log_prob
            report['absolute_objective_term_proxy_concentration'] = concentration(score)
            report['objective_term_proxy_nonfinite'] = int((~np.isfinite(score)).sum())
        reports.append(report)
    return {'rounds': reports,
            'feature_limitation': 'Features are reconstructed from log-normalized stored inputs; clipped features (value 5) are lower bounds. They describe selection-view observations, not comparison branches.',
            'gradient_limitation': 'Return and scalar objective-term concentration are proxies. Actual parameter-gradient attribution requires behavior-policy parameters per training segment, which are not stored for all historical rounds.',
            'missing_return_limitation': 'Invalid descendants include both unobserved faces and recursively invalid returns. Adjacent transitions can belong to different segments, so next-descendant counts may include cases that were not dropped from the earlier segment.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-dir', required=True)
    parser.add_argument('--comparison-dir', required=True)
    args = parser.parse_args()
    trajectory_path = os.path.join(args.training_dir, 'trajectory.pt')
    checkpoint = load_checkpoint(trajectory_path)
    with open(os.path.join(args.comparison_dir, 'multibranch_comparison.json')) as source:
        comparison = json.load(source)
    config = checkpoint.get('config', {})
    for key in ('checkpoint_sha256', 'mesh_sha256'):
        if not config.get(key) or config.get(key) != comparison.get(key):
            raise ValueError('Training and comparison {} missing or mismatched'.format(key))
    digest = hashlib.sha256()
    with open(trajectory_path, 'rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    comparison_policy_hash = comparison.get('policy_checkpoint_sha256')
    report = {'source_hashes': {key: config[key] for key in ('checkpoint_sha256', 'mesh_sha256')},
              'training_trajectory_sha256': digest.hexdigest(),
              'comparison_policy_sha256': comparison_policy_hash,
              'comparison_used_this_trajectory': digest.hexdigest() == comparison_policy_hash if comparison_policy_hash else None,
              'training': training_rounds(checkpoint), 'comparison': compare_branches(comparison)}
    json.dump(report, sys.stdout, indent=2, allow_nan=False)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
