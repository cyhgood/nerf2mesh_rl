"""Shared candidate selection, geometric features and vertex-gradient state."""

import numpy as np
import torch
from torch import nn


FEATURE_VERSION = 'area_rgb_vertex_ema_curvature_visibility_budget_v1'
FEATURE_NAMES = ('area_log', 'rgb_log', 'vertex_gradient_ema_log', 'normal_angle_log',
                 'view_fraction', 'pixel_count_log', 'round_budget_fraction',
                 'gradient_missing', 'curvature_missing')
FEATURE_DEFINITIONS = {
    'area_log': 'triangle area in world coordinates',
    'rgb_log': 'selection-view sum of per-pixel RGB SmoothL1 divided by visible pixel count',
    'vertex_gradient_ema_log': 'mean of three vertex offset-gradient norm EMAs; missing if any vertex unseen',
    'normal_angle_log': 'mean dihedral angle in radians over valid manifold neighbors',
    'view_fraction': 'number of selection views seeing the face divided by number of selection views',
    'pixel_count_log': 'total visible pixels over selection views',
    'round_budget_fraction': 'remaining budget divided by per-round budget immediately before each decision',
    'gradient_missing': 'one when at least one vertex EMA has not been observed',
    'curvature_missing': 'one when no valid manifold neighbor exists',
    'normalization': 'log features use log1p(value / candidate positive median), clipped to 5',
}


class CandidatePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(len(FEATURE_NAMES), 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, features):
        return self.network(features).squeeze(-1)


class SequentialFaceDecisions:
    """Record exactly the state and Bernoulli action used before each edge attempt."""
    def __init__(self, policy, candidates, features, budget, generator=None, deterministic=False, graph=None):
        self.policy = policy
        self.positions = {int(face): index for index, face in enumerate(candidates)}
        self.features = features.copy()
        self.budget = budget
        self.generator = generator
        self.deterministic = deterministic
        self.actions = np.zeros(len(candidates), dtype=bool)
        self.decided = np.zeros(len(candidates), dtype=bool)
        self.probabilities = np.full(len(candidates), np.nan)
        self.log_probabilities = np.full(len(candidates), np.nan)
        self.remaining = np.full(len(candidates), -1, dtype=np.int64)
        self.graph = graph
        if graph is not None:
            face_ids = np.asarray(candidates, dtype=np.int64)
            if np.any(face_ids < 0) or np.any(face_ids >= len(graph['face_to_node'])):
                raise ValueError('Candidate face is absent from the policy graph')
            indices = graph['face_to_node'][face_ids]
            if (np.any(indices < 0) or np.any(indices >= len(graph['node_to_face'])) or
                    not np.array_equal(graph['node_to_face'][indices], face_ids)):
                raise ValueError('Candidate face is absent from or mismatched with the policy graph')
            self.candidate_nodes = torch.as_tensor(indices, dtype=torch.long)
            with torch.no_grad():
                self.embeddings = policy.encode(graph)[self.candidate_nodes]

    def score_all(self, remaining):
        """Probability of every candidate at one common budget; does not decide."""
        fraction = remaining / self.budget if self.budget else 0.0
        with torch.no_grad():
            if self.graph is not None:
                logits = self.policy.score(self.embeddings, np.full(len(self.features), fraction, dtype=np.float32))
            else:
                features = self.features.copy()
                features[:, 6] = fraction
                logits = self.policy(torch.from_numpy(features))
            probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
        if probabilities.shape != (len(self.features),) or not np.isfinite(probabilities).all():
            raise ValueError('Policy produced invalid candidate probabilities')
        return probabilities

    def __call__(self, face_id, remaining):
        index = self.positions[face_id]
        self.features[index, 6] = remaining / self.budget if self.budget else 0.0
        with torch.no_grad():
            logits = (self.policy.score(self.embeddings[index:index + 1], [self.features[index, 6]])[0]
                      if self.graph is not None else self.policy(torch.from_numpy(self.features[index:index + 1]))[0])
            probability = torch.sigmoid(logits)
            action = bool(probability >= .5) if self.deterministic else bool(
                torch.rand((), generator=self.generator) < probability)
            log_probability = torch.nn.functional.logsigmoid(logits if action else -logits)
        self.actions[index] = action
        self.decided[index] = True
        self.probabilities[index] = float(probability)
        self.log_probabilities[index] = float(log_probability)
        self.remaining[index] = remaining
        return action


class RankedFaceDecisions(SequentialFaceDecisions):
    """Evaluation only: request the highest current probability until budget runs out."""
    def __init__(self, policy, candidates, features, budget, graph=None):
        super().__init__(policy, candidates, features, budget, deterministic=True, graph=graph)
        self.candidates = np.asarray(candidates, dtype=np.int64)
        self.initial_probabilities = self.score_all(budget)
        self._budget = None
        self._ordered = np.empty(0, dtype=np.int64)
        self._offset = 0
        self._scores = None
        self.decision_order = []

    def next_candidate(self, remaining):
        if remaining <= 0:
            raise ValueError('Cannot select a ranked candidate with zero budget')
        if remaining != self._budget:
            self._scores = self.score_all(remaining)
            indices = np.flatnonzero(~self.decided)
            self._ordered = indices[np.lexsort((self.candidates[indices], -self._scores[indices]))]
            self._budget, self._offset = remaining, 0
        while self._offset < len(self._ordered) and self.decided[self._ordered[self._offset]]:
            self._offset += 1
        if self._offset >= len(self._ordered):
            raise ValueError('Ranked candidate set is exhausted')
        index = self._ordered[self._offset]
        self._offset += 1
        return int(self.candidates[index])

    def __call__(self, face_id, remaining):
        index = self.positions[face_id]
        if self._budget != remaining or self.decided[index]:
            raise ValueError('Ranked decision does not match the current budget or candidate state')
        self.features[index, 6] = remaining / self.budget
        self.actions[index] = True
        self.decided[index] = True
        self.probabilities[index] = self._scores[index]
        self.remaining[index] = remaining
        self.decision_order.append(int(face_id))
        # Ranking is deterministic and is not a Bernoulli policy sample.
        return True


def visibility_counts(observations, face_count):
    sums = np.zeros(face_count, dtype=np.float64)
    pixels = np.zeros(face_count, dtype=np.int64)
    views = np.zeros(face_count, dtype=np.int64)
    for observation in observations:
        faces, losses = observation[:2]
        valid = faces >= 0
        sums += np.bincount(faces[valid], weights=losses[valid], minlength=face_count)
        pixels += np.bincount(faces[valid], minlength=face_count)
        views[np.unique(faces[valid])] += 1
    return sums, pixels, views


def normal_angles(vertices, faces, min_area):
    triangles = vertices[faces]
    crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(crosses, axis=1)
    valid = np.isfinite(lengths) & (lengths > 2 * min_area)
    normals = np.zeros_like(crosses, dtype=np.float64)
    normals[valid] = crosses[valid] / lengths[valid, None]
    edges = {}
    for index, face in enumerate(faces):
        for i in range(3):
            edge = tuple(sorted((int(face[i]), int(face[(i + 1) % 3]))))
            edges.setdefault(edge, []).append(index)
    angle_sum = np.zeros(len(faces), dtype=np.float64)
    neighbor_count = np.zeros(len(faces), dtype=np.int64)
    for incident in edges.values():
        if len(incident) != 2 or incident[0] == incident[1]:
            continue
        a, b = incident
        if not valid[a] or not valid[b]:
            continue
        angle = np.arccos(np.clip(np.dot(normals[a], normals[b]), -1., 1.))
        angle_sum[a] += angle
        angle_sum[b] += angle
        neighbor_count[a] += 1
        neighbor_count[b] += 1
    angles = np.zeros(len(faces), dtype=np.float64)
    np.divide(angle_sum, neighbor_count, out=angles, where=neighbor_count > 0)
    return lengths * 0.5, angles, neighbor_count > 0


def candidate_features(vertices, faces, sums, pixels, view_counts, num_views,
                       vertex_ema, vertex_seen, budget_fraction, fraction=0.5, min_area=1e-12):
    if not 0 < fraction <= 1 or not np.isfinite(min_area) or min_area < 0:
        raise ValueError('Candidate fraction must be in (0,1] and minimum area nonnegative')
    if len(vertex_ema) != len(vertices) or len(vertex_seen) != len(vertices):
        raise ValueError('Vertex gradient EMA length does not match the mesh')
    if num_views < 1 or not 0 <= budget_fraction <= 1:
        raise ValueError('Invalid view count or budget fraction')
    areas, angles, angle_valid = normal_angles(vertices, faces, min_area)
    valid = (pixels > 0) & np.isfinite(areas) & (areas > min_area) & np.isfinite(sums)
    ids = np.flatnonzero(valid)
    order = np.lexsort((ids, -areas[ids]))
    candidates = ids[order[:int(np.ceil(len(ids) * fraction))]]
    gradient_valid = (vertex_seen[faces] & np.isfinite(vertex_ema[faces]) & (vertex_ema[faces] >= 0)).all(axis=1)
    gradients = np.mean(vertex_ema[faces], axis=1)
    gradients[~gradient_valid] = 0.
    raw = np.stack((areas[candidates], sums[candidates] / pixels[candidates],
                    gradients[candidates], angles[candidates], pixels[candidates]), axis=1)
    scales = np.array([np.median(col[col > 0]) if np.any(col > 0) else 1.0 for col in raw.T])
    normalized = np.minimum(np.log1p(raw / scales), 5.0)
    features = np.column_stack((normalized[:, :4], view_counts[candidates] / num_views,
                                normalized[:, 4], np.full(len(candidates), budget_fraction),
                                ~gradient_valid[candidates], ~angle_valid[candidates])).astype(np.float32)
    total_error = float(sums[ids].sum())
    info = {'valid_faces': int(len(ids)), 'candidate_faces': int(len(candidates)),
            'candidate_rgb_error_sum': float(sums[candidates].sum()),
            'valid_rgb_error_sum': total_error,
            'candidate_rgb_error_coverage': float(sums[candidates].sum() / total_error) if total_error > 0 else None,
            'normalization_scales': scales.tolist(), 'feature_names': list(FEATURE_NAMES),
            'feature_definitions': FEATURE_DEFINITIONS, 'round_budget_fraction': float(budget_fraction),
            'gradient_missing': int((~gradient_valid[candidates]).sum()),
            'curvature_missing': int((~angle_valid[candidates]).sum())}
    return candidates, features, info


def remap_vertex_ema(ema, seen, event, new_vertex_count):
    pairs = [[i, i] for i in range(len(ema))]
    for edge in event['edge_events']:
        if edge['midpoint_vertex'] != len(pairs):
            raise ValueError('Unexpected midpoint vertex ordering')
        pairs.append(edge['edge'])
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(pairs) != new_vertex_count:
        raise ValueError('Vertex mapping length does not match the updated mesh')
    return ema[pairs].mean(axis=1), seen[pairs].all(axis=1), pairs


def update_vertex_ema(state, norms, beta, step_succeeded):
    if norms is None or not step_succeeded:
        return
    norms = np.asarray(norms, dtype=np.float64)
    if norms.shape != state['ema'].shape or not np.isfinite(norms).all():
        return
    state['ema'] = beta * state['ema'] + (1 - beta) * norms
    state['seen'][:] = True
