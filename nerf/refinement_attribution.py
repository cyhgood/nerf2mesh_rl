"""Fixed pre-refinement face-region measurements for each split edge."""

import numpy as np


def _summary(ids, counts, before, after):
    ids = np.asarray(sorted(ids), dtype=np.int64)
    if not len(ids):
        return {'faces': 0, 'valid_faces': 0, 'valid_pixels': 0,
                'median_improvement': None, 'pixel_weighted_improvement': None,
                'positive_improvement_fraction': None}
    valid = (counts[ids] > 0) & np.isfinite(before[ids]) & np.isfinite(after[ids])
    chosen = ids[valid]
    if not len(chosen):
        return {'faces': len(ids), 'valid_faces': 0, 'valid_pixels': 0,
                'median_improvement': None, 'pixel_weighted_improvement': None,
                'positive_improvement_fraction': None}
    delta = before[chosen] - after[chosen]
    return {'faces': len(ids), 'valid_faces': len(chosen), 'valid_pixels': int(counts[chosen].sum()),
            'median_improvement': float(np.median(delta)),
            'pixel_weighted_improvement': float(np.average(delta, weights=counts[chosen])),
            'positive_improvement_fraction': float(np.mean(delta > 0))}


def edge_outcomes(event, face_count, rgb_counts, rgb_before, rgb_after, sil_counts, sil_before, sil_after,
                  grad_counts=None, grad_before=None, grad_after=None):
    requesters = {int(edge['requester_face']) for edge in event['edge_events']}
    affected = {int(face) for edge in event['edge_events'] for face in edge['affected_faces']}
    passive = set(map(int, event['passive_faces']))
    unchanged = set(range(face_count)) - affected

    def face_record(face):
        def metric(counts, before, after):
            valid = counts[face] > 0 and np.isfinite(before[face]) and np.isfinite(after[face])
            return {'valid': bool(valid), 'pixels': int(counts[face]),
                    'before': float(before[face]) if valid else None,
                    'after': float(after[face]) if valid else None,
                    'improvement': float(before[face] - after[face]) if valid else None}
        record = {'face_id': face, 'passively_split': face in passive,
                'rgb': metric(rgb_counts, rgb_before, rgb_after),
                'silhouette': metric(sil_counts, sil_before, sil_after)}
        if grad_counts is not None:
            record['gradient'] = metric(grad_counts, grad_before, grad_after)
        return record

    edges = []
    for edge in event['edge_events']:
        faces = set(map(int, edge['affected_faces']))
        entry = {'edge': edge['edge'], 'requester_face': int(edge['requester_face']),
                      'added_faces': int(edge['added_faces']),
                      'faces': [face_record(face) for face in sorted(faces)],
                      'combined_rgb': _summary(faces, rgb_counts, rgb_before, rgb_after),
                      'combined_silhouette': _summary(faces, sil_counts, sil_before, sil_after)}
        if grad_counts is not None:
            entry['combined_gradient'] = _summary(faces, grad_counts, grad_before, grad_after)
        edges.append(entry)
    groups = {}
    for name, ids in [('executed_requesters', requesters), ('passive_neighbors', passive),
                      ('unchanged_faces', unchanged)]:
        groups[name] = {'rgb': _summary(ids, rgb_counts, rgb_before, rgb_after),
                        'silhouette': _summary(ids, sil_counts, sil_before, sil_after)}
        if grad_counts is not None:
            groups[name]['gradient'] = _summary(ids, grad_counts, grad_before, grad_after)
    return {'edges': edges, 'groups': groups,
            'group_definition': 'unchanged_faces means every old face not affected by a split, including unobserved or rejected faces'}
