# Optional Stage 1 depth supervision

Disabled by default. `--lambda_stage1_depth` is independent of the existing Stage 0 `--lambda_depth`. Set a positive value and `--stage1_depth_manifest PATH` to enable it. The initial implementation supports `--data_format nerf --train_split train` only. It never discovers or opens test-depth images. A missing training view is an error.

The manifest is a JSON file, explicitly prepared after determining the actual meaning of the data. Example (illustrative values, not a calibration of Lego data):

```json
{
  "split": "train",
  "definition": "camera_z",
  "source_unit": "centimeters",
  "transformed_unit": "scene_units",
  "scale": 0.01,
  "offset": 0.0,
  "invalid_value": 0,
  "frames": {"0": "depth/train_000.npy", "1": "depth/train_001.npy"}
}
```

`definition` is `camera_z` (positive forward, `-z` in the repository's camera coordinates) or `ray_distance` (Euclidean camera-center-to-surface distance). Source depths must be 2-D floating-point `.npy` arrays in rendered training-image resolution; no automatic resizing or PNG decoding occurs. Frame keys are **dataset training indices after filtering missing images and time filtering**, not necessarily original transform indices. Independently verify correspondence. Relative paths are resolved against the manifest directory. All requested training frames must be listed.

`invalid_value` is a finite raw sentinel or the string `nan`; nonfinite raw values are invalid regardless. The raw sentinel is masked **before** the affine conversion `scene_depth = raw * scale + offset`. `source_unit` documents the original quantity and `transformed_unit` must be `scene_units`, i.e. the same normalized scene coordinates as the loaded camera poses and mesh. Conversion, including dataset camera scaling, is the user's responsibility; unknown units are not calibrated automatically. Valid transformed values must be finite and positive. The manifest path, SHA-256 and interpretation are logged; multi-round metadata records them too.

Mesh depth comes from the rasterized interpolated world-space surface point, without alpha multiplication, projected onto camera forward or measured from camera center. With SSAA it uses nearest subsampling of both point depth and raster hit mask. The L1 loss averages only pixels having both a valid target and a mesh hit. An empty overlap contributes a zero gradient and reports zero valid pixels. This interface does not penalize missing mesh coverage; silhouette supervision is separate. `train/loss_stage1_depth` and `train/stage1_depth_valid_pixels` are TensorBoard metrics; multi-round records also contain per-step values.

## Stage 0 pseudo-depth investigation

Load a Stage 0 checkpoint and the **training** dataset using the same scale, intrinsics, resolution, camera poses, bounds and occupancy grid as training. Set `model.eval()`, use `torch.no_grad()`, obtain each pose from `train_data.poses[i:i+1]`, and construct full-frame rays with `get_rays(pose, train_data.intrinsics, H, W, N=-1)`; call `model.render(..., perturb=False)` with the same rendering options. `outputs['depth']` is `sum(w_k * t_k)`, and `outputs['weights_sum']` is `sum(w_k)` in both train and inference paths. With this repository's unnormalized camera rays `(x, y, -1)`, `t_k` is camera-forward z depth in scene units, not Euclidean ray length. For opaque-enough pixels, candidate conditional pseudo-depth is `depth / weights_sum`; for ray distance additionally multiply by `norm(rays_d)` when the camera rotation is orthonormal. Save raw float arrays and the weight sum or a separately chosen confidence mask, never the visualization PNG from `Trainer.test` (which min-max normalizes each frame). The exact opacity threshold must be selected and recorded; no universal threshold is implied. Inference uses an early-termination threshold and finite sampling, so conditional depth is still a weighted mean of potentially multiple surfaces, not guaranteed ground truth.

Before using pseudo-depth: inspect opacity and invalid coverage, compare train-view reconstructions and known geometry or ray/plane intersections, check consistent camera scale and axis convention, verify conditional vs raw depth near silhouettes and transparent/occluded surfaces, and compare a few independent views. Record checkpoint and grid state, rendering options, threshold and all transforms. Avoid interpreting the Lego test PNGs until their encoding, semantics and scale have been independently established.
