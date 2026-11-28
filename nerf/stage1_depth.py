"""Explicit Stage 1 depth targets; PNGs are never decoded as depth."""
import hashlib
import json
import math
import os

import numpy as np
import torch


class Stage1DepthSource:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        with open(self.path, 'rb') as source:
            raw = source.read()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.metadata = json.loads(raw)
        m = self.metadata
        required = ('split', 'definition', 'source_unit', 'transformed_unit', 'scale', 'offset', 'invalid_value', 'frames')
        if any(k not in m for k in required):
            raise ValueError('Stage 1 depth manifest must declare ' + ', '.join(required))
        if m['split'] != 'train' or m['definition'] not in ('camera_z', 'ray_distance'):
            raise ValueError('Stage 1 depth manifest requires split=train and definition=camera_z or ray_distance')
        if not isinstance(m['source_unit'], str) or not m['source_unit'].strip() or m['transformed_unit'] != 'scene_units':
            raise ValueError('Depth source_unit must be named and transformed_unit must be scene_units')
        if not all(isinstance(m[k], (int, float)) and math.isfinite(m[k]) for k in ('scale', 'offset')) or m['scale'] <= 0:
            raise ValueError('Depth scale must be positive and scale/offset finite')
        invalid = m['invalid_value']
        if invalid != 'nan' and (not isinstance(invalid, (int, float)) or not math.isfinite(invalid)):
            raise ValueError('Depth invalid_value must be a finite number or "nan"')
        if not isinstance(m['frames'], dict) or not m['frames']:
            raise ValueError('Depth frames must map training view indices to .npy files')

    def describe(self):
        return {'manifest': self.path, 'manifest_sha256': self.sha256,
                **{key: self.metadata[key] for key in ('split', 'definition', 'source_unit', 'transformed_unit', 'scale', 'offset', 'invalid_value')}}

    def load(self, index, shape, device):
        name = self.metadata['frames'].get(str(int(index)))
        if not isinstance(name, str) or not name.lower().endswith('.npy'):
            raise ValueError('Depth manifest has no .npy for training view {}'.format(index))
        path = os.path.abspath(os.path.join(os.path.dirname(self.path), name))
        values = np.load(path, allow_pickle=False)
        if values.shape != tuple(shape) or values.dtype.kind != 'f':
            raise ValueError('Depth target {} must be a float .npy of shape {}'.format(path, tuple(shape)))
        valid = np.isfinite(values)
        invalid = self.metadata['invalid_value']
        if invalid != 'nan':
            valid &= values != invalid
        result = np.zeros(shape, dtype=np.float32)
        result[valid] = values[valid] * self.metadata['scale'] + self.metadata['offset']
        if not np.isfinite(result[valid]).all() or np.any(result[valid] <= 0):
            raise ValueError('Transformed valid depth values must be finite and positive')
        return (torch.as_tensor(result, device=device), torch.as_tensor(valid, device=device))


def stage1_depth_loss(prediction, hit, target, target_valid):
    """Mean L1 over target-valid raster hits; empty overlap produces no gradient."""
    if prediction.numel() != target.numel() or hit.numel() != target.numel():
        raise ValueError('Stage 1 depth prediction and target shapes do not match')
    prediction, hit = prediction.reshape(-1).float(), hit.reshape(-1).bool()
    target, target_valid = target.reshape(-1).float(), target_valid.reshape(-1).bool()
    valid = target_valid & hit & torch.isfinite(prediction) & (prediction > 0)
    if not valid.any():
        return prediction.sum() * 0, 0
    return (prediction[valid] - target[valid]).abs().mean(), int(valid.sum().item())
