"""Sparse face-observation graphs for Stage 1; no graph library is required."""
import json

import numpy as np

from nerf.refinement_candidates import candidate_features, normal_angles, FEATURE_NAMES

GRAPH_FEATURE_NAMES = FEATURE_NAMES + ('rgb_missing',)
EDGE_FEATURE_NAMES = ('shared_edge_length', 'normal_angle_radians')


def graph_budget_state(graph, remaining_budget, round_budget):
    """Return current per-node budget feature and eligible mask without mutating the snapshot."""
    if round_budget < 0 or remaining_budget < 0 or remaining_budget > round_budget:
        raise ValueError('remaining budget must be between zero and round budget')
    features = graph['node_features'].copy()
    features[:, 6] = remaining_budget / round_budget if round_budget else 0.
    return features, graph['candidate_mask'] & (remaining_budget > 0)


def build_face_graph(vertices, faces, sums, pixels, view_counts, num_views, vertex_ema,
                     vertex_seen, remaining_budget, round_budget, fraction=.5, min_area=1e-12,
                     self_loops=False):
    """Build a sparse directed graph of geometrically valid faces.

    Edges with >2 incident valid faces are counted and excluded; self loops have
    invalid edge features. Indices are local to this one graph snapshot only.
    """
    vertices = np.asarray(vertices)
    faces = np.asarray(faces, dtype=np.int64)
    sums, pixels, view_counts = map(np.asarray, (sums, pixels, view_counts))
    vertex_ema, vertex_seen = np.asarray(vertex_ema), np.asarray(vertex_seen, dtype=bool)
    if faces.ndim != 2 or faces.shape[1] != 3 or vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError('vertices and faces must have shapes [V,3] and [F,3]')
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError('face vertex index out of range')
    if any(x.shape != (len(faces),) for x in (sums, pixels, view_counts)):
        raise ValueError('face observation arrays must match face count')
    if round_budget < 0 or remaining_budget < 0 or remaining_budget > round_budget:
        raise ValueError('remaining budget must be between zero and round budget')
    budget_fraction = remaining_budget / round_budget if round_budget else 0.
    candidates, candidate_values, info = candidate_features(
        vertices, faces, sums, pixels, view_counts, num_views, vertex_ema, vertex_seen,
        budget_fraction, fraction, min_area)
    areas, angles, angle_valid = normal_angles(vertices, faces, min_area)
    geometric = np.isfinite(areas) & (areas > min_area)
    ids = np.flatnonzero(geometric)
    face_to_node = np.full(len(faces), -1, dtype=np.int64)
    face_to_node[ids] = np.arange(len(ids), dtype=np.int64)
    candidate_mask = np.zeros(len(ids), dtype=bool)
    candidate_mask[face_to_node[candidates]] = True
    observed = (pixels[ids] > 0) & np.isfinite(sums[ids])
    gradient_valid = (vertex_seen[faces[ids]] & np.isfinite(vertex_ema[faces[ids]]) & (vertex_ema[faces[ids]] >= 0)).all(axis=1)
    gradients = np.where(gradient_valid, np.mean(vertex_ema[faces[ids]], axis=1), 0.)
    rgb = np.zeros(len(ids), dtype=np.float64)
    np.divide(sums[ids], pixels[ids], out=rgb, where=observed)
    scales = np.asarray(info['normalization_scales'])
    raw = np.column_stack((areas[ids], rgb, gradients, angles[ids], pixels[ids]))
    norm = np.minimum(np.log1p(np.maximum(raw, 0) / scales), 5.)
    features = np.column_stack((norm[:, :4], view_counts[ids] / num_views, norm[:, 4],
                                np.full(len(ids), budget_fraction), ~gradient_valid,
                                ~angle_valid[ids], ~observed)).astype(np.float32)
    # Preserve the exact current MLP input for all candidates.
    features[face_to_node[candidates], :len(FEATURE_NAMES)] = candidate_values
    raw_features = np.column_stack((areas[ids], np.where(observed, rgb, np.nan),
                                    np.where(gradient_valid, gradients, np.nan),
                                    np.where(angle_valid[ids], angles[ids], np.nan),
                                    view_counts[ids], pixels[ids], np.full(len(ids), budget_fraction)))
    local_faces = faces[ids]
    edges = np.sort(local_faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    owners = np.repeat(np.arange(len(ids), dtype=np.int64), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owners = edges[order], owners[order]
    if len(edges):
        starts = np.r_[0, np.flatnonzero(np.any(edges[1:] != edges[:-1], axis=1)) + 1]
        counts = np.diff(np.r_[starts, len(edges)])
    else:
        starts = counts = np.empty(0, dtype=np.int64)
    manifold = starts[counts == 2]
    left, right = owners[manifold], owners[manifold + 1]
    distinct = left != right
    manifold, left, right = manifold[distinct], left[distinct], right[distinct]
    tri = vertices[local_faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = cross / (2 * areas[ids, None])
    lengths = np.linalg.norm(vertices[edges[manifold, 0]] - vertices[edges[manifold, 1]], axis=1)
    edge_angles = np.arccos(np.clip(np.einsum('ij,ij->i', normals[left], normals[right]), -1., 1.))
    edge_index = np.stack((np.r_[left, right], np.r_[right, left]))
    edge_values = np.tile(np.column_stack((lengths, edge_angles)), (2, 1)).astype(np.float32)
    edge_valid = np.isfinite(edge_values)
    edge_values[~edge_valid] = 0.
    shared_vertices = np.tile(edges[manifold], (2, 1))
    if self_loops:
        nodes = np.arange(len(ids), dtype=np.int64)
        edge_index = np.concatenate((edge_index, np.stack((nodes, nodes))), axis=1)
        edge_values = np.concatenate((edge_values, np.zeros((len(ids), 2), dtype=np.float32)))
        edge_valid = np.concatenate((edge_valid, np.zeros((len(ids), 2), dtype=bool)))
        shared_vertices = np.concatenate((shared_vertices, np.full((len(ids), 2), -1, dtype=np.int64)))
    metadata = {'nodes': len(ids), 'directed_edges': int(edge_index.shape[1]),
                'undirected_adjacencies': len(left), 'boundary_edges': int(np.count_nonzero(counts == 1)),
                'nonmanifold_edges': int(np.count_nonzero(counts > 2)),
                'invalid_faces': int(len(faces) - len(ids)), 'self_loops': bool(self_loops),
                'nonmanifold_rule': 'exclude edges with more than two incident geometrically valid faces',
                'candidate_info': info, 'node_feature_names': list(GRAPH_FEATURE_NAMES),
                'raw_node_feature_names': ['area', 'rgb_mean', 'gradient_ema_mean', 'normal_angle_mean',
                                           'visible_views', 'visible_pixels', 'remaining_budget_fraction'],
                'edge_feature_names': list(EDGE_FEATURE_NAMES), 'edge_feature_units': ['scene_units', 'radians'],
                'edge_normalization': 'none; raw scene units and radians',
                'budget': {'remaining': int(remaining_budget), 'round': int(round_budget)},
                'policy': 'existing shared CandidatePolicy MLP; graph is observation only'}
    graph = {'node_to_face': ids, 'face_to_node': face_to_node, 'candidate_mask': candidate_mask,
             'decision_mask': candidate_mask & (remaining_budget > 0), 'node_features': features,
             'raw_node_features': raw_features, 'rgb_valid': observed, 'gradient_valid': gradient_valid,
             'curvature_valid': angle_valid[ids], 'edge_index': edge_index, 'edge_features': edge_values,
             'edge_feature_valid': edge_valid, 'edge_vertices': shared_vertices, 'metadata': metadata}
    return graph, candidates, candidate_values, info


def save_face_graph(path, graph, parent_face_id=None):
    arrays = {key: value for key, value in graph.items() if key != 'metadata'}
    arrays['metadata_json'] = np.asarray(json.dumps(graph['metadata'], allow_nan=False))
    if parent_face_id is not None:
        arrays['parent_face_id'] = np.asarray(parent_face_id, dtype=np.int64)
    np.savez_compressed(path, **arrays)
