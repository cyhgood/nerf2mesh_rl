"""Read-only per-loss gradient comparison for sampled Stage 1 training steps."""
import random

import numpy as np
import torch


APPEARANCE_PREFIXES = ('encoder_color.', 'color_net.', 'specular_net.', 'encoder_dir.')
EPSILON = 1e-12


def parameter_groups(model, optimizer):
    optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
    groups = {'vertex_offsets': [], 'appearance': []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) not in optimized:
            continue
        local_name = name[7:] if name.startswith('module.') else name
        if local_name == 'vertices_offsets':
            groups['vertex_offsets'].append((name, parameter))
        elif local_name.startswith(APPEARANCE_PREFIXES) or local_name == 'individual_codes':
            groups['appearance'].append((name, parameter))
    return groups


def _gradient_for(loss, parameters, amp_scale):
    if loss is None:
        return None, 'loss_unavailable'
    if not torch.isfinite(loss.detach()).all().item():
        return None, 'nonfinite_loss'
    if not loss.requires_grad:
        return None, 'loss_has_no_gradient_graph'
    try:
        return torch.autograd.grad(loss * amp_scale, parameters, retain_graph=True, allow_unused=True), None
    except RuntimeError as error:
        return None, 'autograd_error: ' + str(error)


def _norm(grads, error, names, coefficient, epsilon, amp_scale):
    if error is not None:
        return {'valid': False, 'reason': error, 'missing_parameters': [], 'norm': None, 'scaled_norm': None}
    missing = [name for name, gradient in zip(names, grads) if gradient is None]
    if missing:
        return {'valid': False, 'reason': 'missing_parameter_gradient', 'missing_parameters': missing,
                'norm': None, 'scaled_norm': None}
    total = 0.
    for grad in grads:
        values = grad.detach().reshape(-1)
        for chunk in values.split(1048576):
            chunk = chunk.double() / amp_scale
            if not torch.isfinite(chunk).all().item():
                return {'valid': False, 'reason': 'nonfinite_gradient', 'missing_parameters': [], 'norm': None, 'scaled_norm': None}
            total += float(torch.dot(chunk, chunk).item())
    norm = float(np.sqrt(total))
    if not np.isfinite(norm):
        return {'valid': False, 'reason': 'nonfinite_norm', 'missing_parameters': [], 'norm': None, 'scaled_norm': None}
    if norm <= epsilon:
        return {'valid': False, 'reason': 'norm_too_small', 'missing_parameters': [], 'norm': None, 'scaled_norm': None}
    return {'valid': True, 'reason': None, 'missing_parameters': [], 'norm': norm,
            'scaled_norm': float(abs(coefficient) * norm)}


def analyze_loss_gradients(model, optimizer, rgb_loss, gradient_loss, gradient_pixels, gradient_coefficient, epsilon=EPSILON,
                           gradient_unavailable_reason=None, amp_scale=1.):
    """Temporarily scale autograd.grad and undo scaling without touching GradScaler."""
    if not np.isfinite(amp_scale) or amp_scale <= 0:
        raise ValueError('AMP gradient scale must be finite and positive')
    states = (random.getstate(), np.random.get_state(), torch.random.get_rng_state(),
              torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    try:
        groups = parameter_groups(model, optimizer)
        all_named = groups['vertex_offsets'] + groups['appearance']
        all_parameters = [value for _, value in all_named]
        if all_parameters:
            all_rgb, rgb_error = _gradient_for(rgb_loss, all_parameters, amp_scale)
            if gradient_unavailable_reason:
                all_grad, grad_error = None, gradient_unavailable_reason
            elif gradient_pixels == 0:
                all_grad, grad_error = None, 'no_valid_gradient_pixels'
            else:
                all_grad, grad_error = _gradient_for(gradient_loss, all_parameters, amp_scale)
        else:
            all_rgb = all_grad = None
            rgb_error = grad_error = 'no_optimized_parameters'
        output = {}
        offset = 0
        for name, named in groups.items():
            names = [key for key, _ in named]
            params = [value for _, value in named]
            metadata = [{'name': key, 'shape': list(value.shape), 'numel': value.numel()} for key, value in named]
            if not params:
                invalid = {'valid': False, 'reason': 'no_optimized_parameters', 'missing_parameters': [], 'norm': None, 'scaled_norm': None}
                output[name] = {'parameters': metadata, 'rgb': dict(invalid), 'image_gradient': dict(invalid),
                                'valid': False, 'reason': 'no_optimized_parameters', 'dot': None, 'cosine': None}
                continue
            rgb_grads = all_rgb[offset:offset + len(params)] if all_rgb is not None else None
            grad_grads = all_grad[offset:offset + len(params)] if all_grad is not None else None
            offset += len(params)
            rgb = _norm(rgb_grads, rgb_error, names, 1., epsilon, amp_scale)
            grad = _norm(grad_grads, grad_error, names, gradient_coefficient, epsilon, amp_scale)
            item = {'parameters': metadata, 'rgb': rgb, 'image_gradient': grad, 'valid': False,
                    'reason': None, 'dot': None, 'cosine': None}
            if not rgb['valid'] or not grad['valid']:
                item['reason'] = 'rgb: {}; image_gradient: {}'.format(rgb['reason'], grad['reason'])
            else:
                dot = 0.
                for left, right in zip(rgb_grads, grad_grads):
                    for a, b in zip(left.detach().reshape(-1).split(1048576), right.detach().reshape(-1).split(1048576)):
                        dot += float(torch.dot(a.double() / amp_scale, b.double() / amp_scale).item())
                cosine = dot / (rgb['norm'] * grad['norm'])
                if np.isfinite(dot) and np.isfinite(cosine):
                    item.update(valid=True, dot=dot, cosine=float(np.clip(cosine, -1., 1.)))
                else:
                    item['reason'] = 'nonfinite_dot_or_cosine'
            output[name] = item
        return output
    finally:
        random.setstate(states[0]); np.random.set_state(states[1]); torch.random.set_rng_state(states[2])
        if states[3] is not None:
            torch.cuda.set_rng_state_all(states[3])


def summarize(samples):
    result = {}
    for name in ('vertex_offsets', 'appearance'):
        values = [s['groups'][name]['cosine'] for s in samples if s['groups'][name]['valid']]
        result[name] = {'samples': len(samples), 'valid_count': len(values),
                        'median_cosine': float(np.median(values)) if values else None,
                        'negative_fraction': float(np.mean(np.asarray(values) < 0)) if values else None}
    return result
