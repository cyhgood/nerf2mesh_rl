# Stage 1 loss gradient diagnostics

`--refine_loss_gradient_diagnostics` enables sampling only in the multi-branch
comparison. `--refine_loss_gradient_interval` (default 10) samples steps 1,
1+interval, ... in each branch/round, and `--refine_loss_gradient_max_samples`
(default 5) limits the number per round. The original output directory must be
new, so existing experiment results are not overwritten.

`loss_gradient_diagnostics_<branch>_round_<NNNN>.json` contains per-sample
training index, step, valid Sobel pixels, loss values, AMP scale and `groups`.
The branch JSON references the file and its `summary`. The cosine uses the
unweighted gradients; `scaled_norm` multiplies by the coefficient of the
corresponding loss (RGB coefficient 1). A missing gradient for any optimized
parameter in the group, a nonfinite loss/gradient, or a norm <= 1e-12 marks that
group invalid and gives a reason. In particular, zero lambda does not invalidate
the raw gradient, but yields zero scaled norm. A zero-pixel Sobel mask is invalid.

The appearance group comprises the optimizer parameters named `encoder_color.*`,
`color_net.*`, `specular_net.*`, `encoder_dir.*` and, when present and optimized,
`individual_codes`. Each report enumerates the actual names, shapes and element
counts. The current direction encoder is identity and generally has no parameters.
`encoder` and `sigma_net` are density-only and excluded. The vertex group contains
`vertices_offsets`. For this comparison entry point `ind_dim=0` is required.

RGB is the current training criterion, averaged over channels and pixels with its
actual training background. Image gradient is the existing ordinary (not texture
weighted) foreground Sobel loss on the current training image, with fixed white
background and the existing valid 3x3 stencil. It is computed even when its
coefficient is zero. Independent evaluation images are never used here. The two
additional `autograd.grad` calls use the current AMP loss scale and divide it out;
they neither accumulate into `.grad` nor advance the optimizer or scaler. RNG
states are restored. CUDA custom kernels can have numerical nondeterminism, so a
GPU end-to-end check is still needed for exact reproducibility there.
