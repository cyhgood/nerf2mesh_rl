"""Sobel observations and fixed pre-split face-region gradient statistics."""
import numpy as np
import torch
import torch.nn.functional as F

TEXTURE_FEATURE_VERSION = 'sobel_foreground_rgb_v1'
TEXTURE_DEFINITION = ('eval RGB color_space; fixed white compositing; Sobel / 8; per-pixel '
                      'sqrt(mean_RGB(gx^2+gy^2)); SmoothL1 mean over RGB and xy; '
                      'valid only if 3x3 target alpha >= .999 and RGB finite, '
                      'image boundary excluded; selection additionally requires 3x3 mesh coverage')


def _full_stencil(mask):
    value = F.conv2d(mask.float()[None, None], torch.ones((1, 1, 3, 3), device=mask.device), padding=1)
    return value[0, 0] == 9


def target_sobel_mask(target, alpha):
    """Fixed target-only stencil: never depends on a branch's rendering or mesh."""
    target = torch.as_tensor(target).float()
    alpha = torch.as_tensor(alpha, device=target.device).float()
    if target.ndim != 3 or target.shape[-1] != 3:
        raise ValueError('Sobel target requires an H x W x 3 image')
    alpha = alpha.reshape(target.shape[:2])
    valid = torch.isfinite(target).all(-1) & torch.isfinite(alpha) & (alpha >= .999)
    return _full_stencil(valid).detach().cpu().numpy().reshape(-1).copy()


def sobel_tensors(prediction, target, alpha, face_ids=None):
    """Differentiable Sobel maps and valid masks, in the caller's RGB color space."""
    prediction = torch.as_tensor(prediction).float()
    target = torch.as_tensor(target, device=prediction.device).float()
    alpha = torch.as_tensor(alpha, device=prediction.device).float()
    if prediction.ndim != 3 or prediction.shape[-1] != 3 or target.shape != prediction.shape:
        raise ValueError('Sobel requires matching H x W x 3 images')
    h, w = prediction.shape[:2]
    alpha = alpha.reshape(h, w)
    finite = torch.isfinite(prediction).all(-1) & torch.isfinite(target).all(-1) & torch.isfinite(alpha)
    foreground = finite & (alpha >= .999)
    coverage = foreground if face_ids is None else foreground & (torch.as_tensor(face_ids, device=prediction.device).reshape(h, w) >= 0)
    pred = torch.where(torch.isfinite(prediction), prediction, torch.zeros_like(prediction)).permute(2, 0, 1)[None]
    gt = torch.where(torch.isfinite(target), target, torch.zeros_like(target)).permute(2, 0, 1)[None]
    kernel = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=pred.device) / 8
    filters = torch.stack((kernel, kernel.T)).unsqueeze(1).repeat(3, 1, 1, 1)
    grad_pred = F.conv2d(F.pad(pred, (1, 1, 1, 1), mode='replicate'), filters, groups=3)
    grad_gt = F.conv2d(F.pad(gt, (1, 1, 1, 1), mode='replicate'), filters, groups=3)
    strength = grad_gt.reshape(1, 3, 2, h, w).square().sum(2).mean(1).sqrt()[0]
    error = F.smooth_l1_loss(grad_pred, grad_gt, reduction='none').mean(1)[0]
    return strength, error, _full_stencil(foreground), _full_stencil(coverage)


def sobel_observation(prediction, target, alpha, face_ids):
    return tuple(x.detach().cpu().numpy().reshape(-1).copy()
                 for x in sobel_tensors(prediction, target, alpha, face_ids))


def stage1_image_gradient_loss(prediction, target, alpha):
    """Mean six-channel Sobel SmoothL1 over finite opaque target stencils."""
    _, error, valid, _ = sobel_tensors(prediction, target, alpha)
    count = int(valid.sum().item())
    if count:
        return error[valid].mean(), count
    safe = torch.where(torch.isfinite(prediction), prediction, torch.zeros_like(prediction))
    return safe.reshape(-1)[:1].sum() * 0., 0


def stage1_texture_weighted_gradient_loss(prediction, target, alpha, gain=1., cap=4.):
    """Texture-emphasized loss with bounded, mean-one detached target weights."""
    if not np.isfinite(gain) or gain < 0 or not np.isfinite(cap) or cap < 1:
        raise ValueError('Texture weight gain must be nonnegative and cap at least one')
    strength, error, valid, _ = sobel_tensors(prediction, target, alpha)
    valid = valid & torch.isfinite(strength) & torch.isfinite(error)
    count = int(valid.sum().item())
    if not count:
        safe = torch.where(torch.isfinite(prediction), prediction, torch.zeros_like(prediction))
        zero = safe.reshape(-1)[:1].sum() * 0.
        return zero, zero, 0, {'scale': None, 'min': None, 'max': None, 'mean': None, 'raw_max': None}
    with torch.no_grad():
        values = strength.detach()[valid]
        positive = values[values > 0]
        scale = positive.median().clamp(min=1e-6) if len(positive) else values.new_tensor(1.)
        raw_weights = (1. + gain * values / scale).clamp(max=cap)
        weights = raw_weights / raw_weights.mean()
        stats = {'scale': float(scale.item()), 'min': float(weights.min().item()),
                 'max': float(weights.max().item()), 'mean': float(weights.mean().item()),
                 'raw_max': float(raw_weights.max().item())}
    return error[valid].mean(), (error[valid] * weights).mean(), count, stats


def texture_face_sums(observations, face_count):
    strength = np.zeros(face_count, dtype=np.float64)
    error = np.zeros(face_count, dtype=np.float64)
    counts = np.zeros(face_count, dtype=np.int64)
    for item in observations:
        face, t, e, _, covered = item[0], *item[-4:]
        valid = covered & (face >= 0) & np.isfinite(t) & np.isfinite(e)
        counts += np.bincount(face[valid], minlength=face_count)
        strength += np.bincount(face[valid], weights=t[valid], minlength=face_count)
        error += np.bincount(face[valid], weights=e[valid], minlength=face_count)
    return strength, error, counts


def fixed_gradient_regions(before, after, face_count):
    if len(before) != len(after):
        raise ValueError('Gradient reward requires the same before and after views')
    counts = np.zeros(face_count, dtype=np.int64)
    sums_before = np.zeros(face_count, dtype=np.float64)
    sums_after = np.zeros(face_count, dtype=np.float64)
    for old, new in zip(before, after):
        if len(old[0]) != len(new[0]):
            raise ValueError('Gradient reward requires the same image dimensions')
        face = old[0]
        old_error, covered = old[-3], old[-1]
        new_error, new_finite_fg = new[-3], new[-2]
        valid = covered & new_finite_fg & (face >= 0) & np.isfinite(old_error) & np.isfinite(new_error)
        counts += np.bincount(face[valid], minlength=face_count)
        sums_before += np.bincount(face[valid], weights=old_error[valid], minlength=face_count)
        sums_after += np.bincount(face[valid], weights=new_error[valid], minlength=face_count)
    first = np.full(face_count, np.nan)
    last = np.full(face_count, np.nan)
    np.divide(sums_before, counts, out=first, where=counts > 0)
    np.divide(sums_after, counts, out=last, where=counts > 0)
    return counts, first, last
