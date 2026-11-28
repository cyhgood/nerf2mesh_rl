"""Stage 1 silhouette supervision using explicit foreground targets."""

import torch


def stage1_foreground_alpha(images, data, shape, device):
    target = images[..., 3] if images.shape[-1] == 4 else None
    if target is None:
        for key in ('foreground_mask', 'mask', 'alpha'):
            if data.get(key) is not None:
                target = data[key]
                break
    if target is None:
        raise ValueError('Stage 1 image gradient loss requires RGBA images or an explicit foreground_mask, mask, or alpha')
    target = torch.as_tensor(target, device=device, dtype=torch.float32)
    if target.numel() != shape[0] * shape[1]:
        raise ValueError('Stage 1 image gradient foreground mask must contain one value per rendered pixel')
    return target.reshape(shape)


def stage1_silhouette_loss(alpha, images, data):
    """Mean absolute alpha error; reject missing or incompatible targets."""
    target = images[..., 3] if images.shape[-1] == 4 else None
    if target is None:
        for key in ('foreground_mask', 'mask', 'alpha'):
            if data.get(key) is not None:
                target = data[key]
                break
    if target is None:
        raise ValueError('Stage 1 silhouette loss requires RGBA images or an explicit foreground_mask, mask, or alpha')
    target = torch.as_tensor(target, device=alpha.device, dtype=torch.float32)
    if target.numel() != alpha.numel():
        raise ValueError('Stage 1 silhouette target must contain one value per rendered pixel')
    target = target.reshape(alpha.shape)
    if not torch.isfinite(target).all() or torch.any((target < 0) | (target > 1)):
        raise ValueError('Stage 1 silhouette target must contain finite values in [0, 1]')
    return (alpha.float() - target).abs().mean()
