"""Selection-view, fixed pre-update face-region gradient diagnostics."""
import numpy as np

from nerf.refinement_texture import texture_face_sums, TEXTURE_DEFINITION


def _mean(values):
    return float(np.mean(values)) if len(values) else None


def _median(values):
    return float(np.median(values)) if len(values) else None


def gradient_diagnostics(before, after, face_count, visible_valid, hybrid_candidates,
                         executed, passive, parent_face_id, selection_views, evaluation_views):
    """Report pre-split errors and same-pixel improvements, keyed by old global face ID."""
    if len(before) != len(after) or len(before) != len(selection_views):
        raise ValueError('Gradient diagnostic views must have matching before/after observations')
    visible_valid = np.asarray(visible_valid, dtype=bool)
    hybrid_candidates = np.asarray(hybrid_candidates, dtype=bool)
    executed = np.asarray(executed, dtype=bool)
    passive = np.asarray(passive, dtype=bool)
    if any(a.shape != (face_count,) for a in (visible_valid, hybrid_candidates, executed, passive)):
        raise ValueError('Gradient diagnostic face masks have wrong shape')
    parent = np.asarray(parent_face_id if parent_face_id is not None else [], dtype=np.int64).reshape(-1)
    parent_valid = (parent >= 0) & (parent < face_count)
    child_counts = np.bincount(parent[parent_valid], minlength=face_count)
    has_child = child_counts > 0
    _, pre_error_sum, pre_pixels = texture_face_sums(before, face_count)
    pre_mean = np.full(face_count, np.nan)
    np.divide(pre_error_sum, pre_pixels, out=pre_mean, where=pre_pixels > 0)
    paired_pixels = np.zeros(face_count, np.int64)
    before_sum = np.zeros(face_count, np.float64)
    after_sum = np.zeros(face_count, np.float64)
    lost_pixels = np.zeros(face_count, np.int64)
    switched_pixels = np.zeros(face_count, np.int64)
    unmapped_pixels = np.zeros(face_count, np.int64)
    invalid_after_pixels = np.zeros(face_count, np.int64)
    for old, new in zip(before, after):
        if old[4] != new[4] or len(old[0]) != len(new[0]):
            raise ValueError('Gradient diagnostic views must have identical dimensions')
        old_faces, new_faces = old[0], new[0]
        old_error, old_covered = old[-3], old[-1]
        new_error, after_finite = new[-3], new[-2]
        start = old_covered & (old_faces >= 0) & (old_faces < face_count) & np.isfinite(old_error)
        after_ok = after_finite & np.isfinite(new_error)
        in_range = (new_faces >= 0) & (new_faces < len(parent))
        resolved = np.zeros(len(new_faces), dtype=bool)
        resolved[in_range] = parent_valid[new_faces[in_range]]
        missing_hit = new_faces < 0
        old_child = np.zeros(len(old_faces), dtype=bool)
        old_child[start] = has_child[old_faces[start]]
        invalid_mapping = start & (~old_child | (~missing_hit & ~resolved))
        lost = start & missing_hit
        switched = np.zeros(len(old_faces), dtype=bool)
        switch_mask = start & resolved
        switched[switch_mask] = parent[new_faces[switch_mask]] != old_faces[switch_mask]
        def add(target, mask, weights=None):
            target += np.bincount(old_faces[mask], weights=weights, minlength=face_count)
        add(lost_pixels, lost)
        add(switched_pixels, switched)
        add(unmapped_pixels, invalid_mapping)
        add(invalid_after_pixels, start & ~after_ok)
        # Lost coverage remains part of the fixed region when its RGB is finite;
        # malformed/missing lineage is excluded and reported separately.
        paired = start & after_ok & ~invalid_mapping
        add(paired_pixels, paired)
        add(before_sum, paired, old_error[paired])
        add(after_sum, paired, new_error[paired])
    paired_before = np.full(face_count, np.nan)
    paired_after = np.full(face_count, np.nan)
    np.divide(before_sum, paired_pixels, out=paired_before, where=paired_pixels > 0)
    np.divide(after_sum, paired_pixels, out=paired_after, where=paired_pixels > 0)
    improvement = paired_before - paired_after
    groups = {'visible_valid': visible_valid, 'hybrid_candidates': hybrid_candidates,
              'executed_requesters': executed, 'passive_neighbors': passive,
              'unchanged': ~(executed | passive)}
    total_error = float(pre_error_sum[visible_valid].sum())
    summary = {}
    for name, group in groups.items():
        eligible = group & visible_valid
        pre_valid = eligible & (pre_pixels > 0) & np.isfinite(pre_mean)
        pair_valid = eligible & (paired_pixels > 0) & np.isfinite(improvement)
        delta = improvement[pair_valid]
        count = int(paired_pixels[pair_valid].sum())
        error_sum = float(pre_error_sum[pre_valid].sum())
        summary[name] = {
            'faces_in_group': int(group.sum()), 'visible_valid_faces': int(eligible.sum()),
            'not_visible_or_invalid_geometry_faces': int((group & ~visible_valid).sum()),
            'pre_valid_faces': int(pre_valid.sum()), 'pre_valid_pixels': int(pre_pixels[pre_valid].sum()),
            'pre_no_valid_gradient_faces': int((eligible & ~pre_valid).sum()),
            'pre_face_error_mean': _mean(pre_mean[pre_valid]), 'pre_face_error_median': _median(pre_mean[pre_valid]),
            'pre_pixel_weighted_error': error_sum / int(pre_pixels[pre_valid].sum()) if pre_pixels[pre_valid].sum() else None,
            'pre_gradient_error_sum': error_sum,
            'pre_gradient_error_coverage': error_sum / total_error if total_error > 0 else None,
            'paired_valid_faces': int(pair_valid.sum()), 'paired_valid_pixels': count,
            'no_paired_gradient_faces': int((eligible & ~pair_valid).sum()),
            'no_child_mapping_faces': int((eligible & ~has_child).sum()),
            'faces_with_lost_coverage': int((eligible & (lost_pixels > 0)).sum()),
            'lost_coverage_pixels': int(lost_pixels[eligible].sum()),
            'faces_with_invalid_correspondence': int((eligible & (unmapped_pixels > 0)).sum()),
            'invalid_correspondence_pixels': int(unmapped_pixels[eligible].sum()),
            'invalid_after_pixels': int(invalid_after_pixels[eligible].sum()),
            'different_parent_pixels': int(switched_pixels[eligible].sum()),
            'mean_improvement': _mean(delta), 'median_improvement': _median(delta),
            'pixel_weighted_improvement': float(np.average(delta, weights=paired_pixels[pair_valid])) if count else None,
            'positive_improvement_fraction': float(np.mean(delta > 0)) if len(delta) else None,
            'paired_before_pixel_weighted_error': float(before_sum[pair_valid].sum() / count) if count else None,
            'paired_after_pixel_weighted_error': float(after_sum[pair_valid].sum() / count) if count else None}
    arrays = {'old_face_id': np.arange(face_count, dtype=np.int64), 'visible_valid': visible_valid,
              'hybrid_candidate': hybrid_candidates, 'executed_requester': executed, 'passive_neighbor': passive,
              'child_count': child_counts, 'pre_gradient_pixels': pre_pixels, 'pre_gradient_error_sum': pre_error_sum,
              'pre_gradient_error': pre_mean, 'paired_pixels': paired_pixels,
              'paired_before_error': paired_before, 'paired_after_error': paired_after, 'improvement': improvement,
              'lost_coverage_pixels': lost_pixels, 'different_parent_pixels': switched_pixels,
              'invalid_correspondence_pixels': unmapped_pixels, 'invalid_after_pixels': invalid_after_pixels}
    metadata = {'diagnostic_views': list(map(int, selection_views)), 'diagnostic_view_dataset': 'validation selection views',
                'independent_evaluation_views': list(map(int, evaluation_views)),
                'statistics_scope': 'per-branch pre-update mesh; no face-ID comparison across branches or rounds',
                'sobel_definition': TEXTURE_DEFINITION,
                'coverage_denominator': 'sum of pre-update per-pixel Sobel SmoothL1 over all visible valid old faces with valid 3x3 selection-view stencils',
                'coverage_denominator_gradient_error_sum': total_error,
                'coverage_denominator_pixels': int(pre_pixels[visible_valid].sum()),
                'invalid_child_parent_entries': int((~parent_valid).sum()),
                'fixed_region_rule': 'old face ID and old valid 3x3 mesh/foreground stencil; after error evaluated at same pixel even if coverage lost or new parent differs; invalid after RGB or malformed lineage excluded',
                'group_overlap': 'hybrid, executed, passive and unchanged are separate overlapping views; only executed/passive/unchanged partition old faces',
                'groups': summary}
    return metadata, arrays


def three_phase_gradient_diagnostics(before, after_split, after_optimization, face_count, visible_valid,
                                     hybrid_candidates, executed, passive, parent_face_id,
                                     selection_views, evaluation_views):
    """Compare three renders on the intersection of the original fixed region."""
    if len(before) != len(after_split) or len(before) != len(after_optimization):
        raise ValueError('Three-phase diagnostic views must match')
    args = (face_count, visible_valid, hybrid_candidates, executed, passive,
            parent_face_id, selection_views, evaluation_views)
    split_report, _ = gradient_diagnostics(before, after_split, *args)
    final_report, _ = gradient_diagnostics(before, after_optimization, *args)
    parent = np.asarray(parent_face_id if parent_face_id is not None else [], dtype=np.int64).reshape(-1)
    parent_valid = (parent >= 0) & (parent < face_count)
    has_child = np.bincount(parent[parent_valid], minlength=face_count) > 0
    common_before = []
    for old, split, final in zip(before, after_split, after_optimization):
        if old[4] != split[4] or old[4] != final[4]:
            raise ValueError('Three-phase diagnostic dimensions must match')
        old_faces = old[0]
        valid_old = (old_faces >= 0) & (old_faces < face_count)
        child = np.zeros(len(old_faces), dtype=bool)
        child[valid_old] = has_child[old_faces[valid_old]]
        common = old[-1] & valid_old & child & np.isfinite(old[-4]) & np.isfinite(old[-3])
        for current in (split, final):
            face_ids = current[0]
            in_range = (face_ids >= 0) & (face_ids < len(parent))
            valid_parent = np.zeros(len(face_ids), dtype=bool)
            valid_parent[in_range] = parent_valid[face_ids[in_range]]
            correspondence = (face_ids < 0) | valid_parent
            common &= current[-2] & np.isfinite(current[-3]) & correspondence
        filtered = list(old)
        filtered[-1] = common
        common_before.append(tuple(filtered))
    _, split_arrays = gradient_diagnostics(common_before, after_split, *args)
    _, final_arrays = gradient_diagnostics(common_before, after_optimization, *args)
    if not np.array_equal(split_arrays['paired_pixels'], final_arrays['paired_pixels']):
        raise RuntimeError('Three-phase fixed pixel sets differ')
    pixels = split_arrays['paired_pixels']
    before_error = split_arrays['paired_before_error']
    split_error = split_arrays['paired_after_error']
    final_error = final_arrays['paired_after_error']
    topology_improvement = before_error - split_error
    optimization_improvement = split_error - final_error
    groups = {'visible_valid': np.asarray(visible_valid, bool), 'hybrid_candidates': np.asarray(hybrid_candidates, bool),
              'executed_requesters': np.asarray(executed, bool), 'passive_neighbors': np.asarray(passive, bool),
              'unchanged': ~(np.asarray(executed, bool) | np.asarray(passive, bool))}
    summaries = {}
    for name, mask in groups.items():
        chosen = mask & np.asarray(visible_valid, bool) & (pixels > 0)
        count = int(pixels[chosen].sum())
        def average(values):
            return float(np.average(values[chosen], weights=pixels[chosen])) if count else None
        def change(values):
            selected = values[chosen]
            return {'mean_improvement': _mean(selected), 'median_improvement': _median(selected),
                    'pixel_weighted_improvement': average(values),
                    'positive_improvement_fraction': float(np.mean(selected > 0)) if len(selected) else None}
        summaries[name] = {'valid_faces': int(chosen.sum()), 'valid_pixels': count,
                           'before_error': average(before_error), 'after_split_error': average(split_error),
                           'after_optimization_error': average(final_error),
                           'topology_change': change(topology_improvement),
                           'optimization_change': change(optimization_improvement),
                           'total_change': change(before_error - final_error),
                           'after_split_coverage_and_mapping': {k: split_report['groups'][name][k] for k in (
                               'lost_coverage_pixels', 'faces_with_lost_coverage', 'invalid_correspondence_pixels',
                               'faces_with_invalid_correspondence', 'invalid_after_pixels', 'different_parent_pixels',
                               'no_child_mapping_faces', 'no_paired_gradient_faces')},
                           'after_optimization_coverage_and_mapping': {k: final_report['groups'][name][k] for k in (
                               'lost_coverage_pixels', 'faces_with_lost_coverage', 'invalid_correspondence_pixels',
                               'faces_with_invalid_correspondence', 'invalid_after_pixels', 'different_parent_pixels',
                               'no_child_mapping_faces', 'no_paired_gradient_faces')}}
    return {'pixel_set': 'intersection of pre-split fixed region with finite RGB and valid lineage at both later renders; lost coverage remains included',
            'groups': summaries}, {'three_phase_pixels': pixels, 'three_phase_before_error': before_error,
                                   'three_phase_after_split_error': split_error, 'three_phase_after_optimization_error': final_error,
                                   'topology_improvement': topology_improvement,
                                   'optimization_improvement': optimization_improvement}
