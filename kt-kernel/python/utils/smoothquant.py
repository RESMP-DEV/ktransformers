"""INT8 artifact helpers for W8A8 conversion.

The SmoothQuant path is a straight W8A8 remap. The MR-GPTQ-inspired path keeps
the MR contract explicit: rotate weights offline with block Hadamards, store
group scales, and require the runtime to rotate activations before matmul.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SmoothQuantConfig:
    alpha: float = 0.5
    eps: float = 1e-12
    quant_max: int = 127


@dataclass(frozen=True)
class SmoothQuantParams:
    weight_q: torch.Tensor
    weight_scale: torch.Tensor
    smooth_scale: torch.Tensor


@dataclass(frozen=True)
class MRGPTQInt8Config:
    rotation_block_size: int = 128
    eps: float = 1e-12
    quant_max: int = 127
    scale_grid_steps: int = 16
    min_scale_ratio: float = 0.5
    static_activation_order: bool = True
    gptq_block_size: int = 128
    rel_damp: float = 1e-2


@dataclass(frozen=True)
class MRGPTQInt8Params:
    weight_q: torch.Tensor
    weight_scale: torch.Tensor
    activation_order: torch.Tensor
    hessian_diag: torch.Tensor
    rotation_block_size: int
    gptq_error_propagation: str


def _check_power_of_two(value: int, name: str) -> None:
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two, got {value}")


def normalized_hadamard(
    size: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a normalized Sylvester Hadamard matrix."""

    _check_power_of_two(size, "Hadamard size")
    h = torch.ones((1, 1), device=device, dtype=dtype)
    while h.shape[0] < size:
        h = torch.cat(
            (
                torch.cat((h, h), dim=1),
                torch.cat((h, -h), dim=1),
            ),
            dim=0,
        )
    return h / (float(size) ** 0.5)


def apply_block_hadamard(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Apply a normalized block-diagonal Hadamard transform over the last dim.

    Full ``block_size`` chunks are rotated. A non-divisible tail is copied
    unchanged, which keeps the transform lossless for odd model widths while
    making the tail policy explicit in the artifact manifest.
    """

    _check_power_of_two(block_size, "rotation_block_size")
    if x.shape[-1] < block_size:
        return x.clone()

    original_shape = x.shape
    x_2d = x.reshape(-1, original_shape[-1])
    full_width = (x_2d.shape[-1] // block_size) * block_size
    head = x_2d[:, :full_width]
    tail = x_2d[:, full_width:]
    blocks = head.reshape(-1, full_width // block_size, block_size)
    h = normalized_hadamard(block_size, device=x.device, dtype=x.dtype)
    rotated = torch.matmul(blocks, h).reshape(x_2d.shape[0], full_width)
    if tail.numel():
        rotated = torch.cat((rotated, tail.clone()), dim=-1)
    return rotated.reshape(original_shape)


def rotate_hessian_block_hadamard(hessian: torch.Tensor, block_size: int) -> torch.Tensor:
    """Rotate a Hessian/Gram matrix into the block-Hadamard activation basis."""

    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError(f"hessian must be square, got shape={tuple(hessian.shape)}")
    rotated_cols = apply_block_hadamard(hessian, block_size)
    rotated = apply_block_hadamard(rotated_cols.transpose(0, 1), block_size).transpose(0, 1)
    return rotated.contiguous()


def _fit_group_int8_scales(
    rotated_weight: torch.Tensor,
    hessian_diag: torch.Tensor,
    config: MRGPTQInt8Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_features, in_features = rotated_weight.shape
    block_size = int(config.rotation_block_size)
    group_count = (in_features + block_size - 1) // block_size
    qweight = torch.empty_like(rotated_weight, dtype=torch.int8)
    weight_scale = torch.empty((out_features, group_count), dtype=torch.float32)
    eps = float(config.eps)
    quant_max = float(config.quant_max)
    ratios = torch.ones(1, dtype=torch.float32, device=rotated_weight.device)
    if config.scale_grid_steps > 1:
        ratios = torch.linspace(
            1.0,
            max(float(config.min_scale_ratio), eps),
            steps=int(config.scale_grid_steps),
            dtype=torch.float32,
            device=rotated_weight.device,
        )

    for group_idx in range(group_count):
        start = group_idx * block_size
        end = min(start + block_size, in_features)
        segment = rotated_weight[:, start:end]
        max_abs = segment.abs().amax(dim=1).clamp_min(eps)
        base_scale = max_abs / quant_max
        best_scale = base_scale
        best_q = torch.round(segment / base_scale.reshape(-1, 1)).clamp(-quant_max, quant_max)

        if ratios.numel() > 1:
            act_weight = hessian_diag[start:end].to(device=segment.device).reshape(1, -1)
            best_error = torch.full((out_features,), float("inf"), dtype=torch.float32, device=segment.device)
            for ratio in ratios:
                candidate_scale = (base_scale * ratio).clamp_min(eps)
                candidate_q = torch.round(segment / candidate_scale.reshape(-1, 1)).clamp(-quant_max, quant_max)
                reconstructed = candidate_q * candidate_scale.reshape(-1, 1)
                error = ((reconstructed - segment).square() * act_weight).mean(dim=1)
                update = error < best_error
                best_error = torch.where(update, error, best_error)
                best_scale = torch.where(update, candidate_scale, best_scale)
                best_q = torch.where(update.reshape(-1, 1), candidate_q, best_q)

        qweight[:, start:end] = best_q.to(torch.int8)
        weight_scale[:, group_idx] = best_scale.cpu()

    return qweight.contiguous(), weight_scale.contiguous()


def _apply_gptq_error_propagation(
    rotated_weight: torch.Tensor,
    hessian: torch.Tensor,
    weight_scale: torch.Tensor,
    config: MRGPTQInt8Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_features, in_features = rotated_weight.shape
    if hessian.shape != (in_features, in_features):
        raise ValueError(f"hessian shape must be {(in_features, in_features)}, got {tuple(hessian.shape)}")

    device = rotated_weight.device
    hessian = hessian.to(device=device, dtype=torch.float32).clone()
    zero_cols = torch.nonzero(rotated_weight.eq(0).all(dim=0), as_tuple=False).reshape(-1)
    if zero_cols.numel():
        hessian[zero_cols, :] = 0
        hessian[:, zero_cols] = 0
        hessian[zero_cols, zero_cols] = 1

    damp = float(config.rel_damp) * torch.diag(hessian).mean().clamp_min(float(config.eps))
    hessian[range(in_features), range(in_features)] += damp
    if config.static_activation_order:
        perm = torch.argsort(torch.diag(hessian), descending=True)
    else:
        perm = torch.arange(in_features, device=device)
    perm_inv = torch.argsort(perm)
    hessian_perm = hessian[perm][:, perm]
    group_size = int(config.rotation_block_size)
    group_count = weight_scale.shape[1]
    group_idx = torch.arange(group_count, device=device).repeat_interleave(group_size)[:in_features][perm]

    try:
        hessian_inv = torch.linalg.inv(hessian_perm)
        hessian_inv_cho = torch.linalg.cholesky(hessian_inv, upper=True)
    except RuntimeError:
        hessian_inv_cho = torch.eye(in_features, device=device, dtype=torch.float32)

    w = rotated_weight[:, perm].clone()
    q_perm = torch.empty_like(w, dtype=torch.int8)
    scale_device = weight_scale.to(device=device, dtype=torch.float32)
    quant_max = float(config.quant_max)
    block_size = int(config.gptq_block_size)
    for c1 in range(0, in_features, block_size):
        c2 = min(c1 + block_size, in_features)
        ncols = c2 - c1
        w_blk = w[:, c1:c2].clone()
        errs = torch.zeros_like(w_blk)
        h_inv_blk = hessian_inv_cho[c1:c2, c1:c2]
        for i in range(ncols):
            w_col = w_blk[:, i]
            d = h_inv_blk[i, i].clamp_min(float(config.eps))
            scale = scale_device[:, group_idx[c1 + i]].clamp_min(float(config.eps))
            q_col = torch.round(w_col / scale).clamp(-quant_max, quant_max)
            w_q = q_col * scale
            w[:, c1 + i] = w_q
            q_perm[:, c1 + i] = q_col.to(torch.int8)
            err = (w_col - w_q) / d
            w_blk[:, i:].addr_(err, h_inv_blk[i, i:], alpha=-1)
            errs[:, i] = err
        w[:, c2:].addmm_(errs, hessian_inv_cho[c1:c2, c2:], alpha=-1)

    return q_perm[:, perm_inv].contiguous(), perm.cpu().to(torch.int32).contiguous()


def calibrate_weight_int8_smoothquant(
    weight: torch.Tensor,
    activation_amax: torch.Tensor,
    config: SmoothQuantConfig | None = None,
) -> SmoothQuantParams:
    """Quantize a 2D linear weight with per-input SmoothQuant scaling.

    The emitted tensors preserve ``x @ weight.T`` as:
    ``(x / smooth_scale) @ (weight_q * weight_scale).T``.
    """

    cfg = config or SmoothQuantConfig()
    if weight.ndim != 2:
        raise ValueError(f"SmoothQuant only supports 2D weights, got shape={tuple(weight.shape)}")

    weight_f = weight.detach().float()
    activation_amax_f = activation_amax.detach().float().reshape(-1)
    in_features = weight_f.shape[1]
    if activation_amax_f.numel() != in_features:
        raise ValueError(
            "activation_amax width must match weight input dimension: "
            f"{activation_amax_f.numel()} != {in_features}"
        )

    alpha = max(0.0, min(1.0, float(cfg.alpha)))
    eps = float(cfg.eps)
    act_scale = activation_amax_f.abs().clamp_min(eps)
    weight_scale_in = weight_f.abs().amax(dim=0).clamp_min(eps)
    smooth_scale = torch.pow(act_scale, alpha) / torch.pow(weight_scale_in, 1.0 - alpha)
    smooth_scale = smooth_scale.clamp_min(eps)

    smoothed_weight = weight_f * smooth_scale.reshape(1, -1)
    row_scale = smoothed_weight.abs().amax(dim=1).clamp_min(eps) / float(cfg.quant_max)
    weight_q = torch.round(smoothed_weight / row_scale.reshape(-1, 1)).clamp(
        -cfg.quant_max,
        cfg.quant_max,
    )

    return SmoothQuantParams(
        weight_q=weight_q.to(torch.int8).contiguous(),
        weight_scale=row_scale.to(torch.float32).contiguous(),
        smooth_scale=smooth_scale.to(torch.float32).contiguous(),
    )


def calibrate_weight_int8_mr_gptq(
    weight: torch.Tensor,
    activation_amax: torch.Tensor,
    config: MRGPTQInt8Config | None = None,
    hessian: torch.Tensor | None = None,
) -> MRGPTQInt8Params:
    """Quantize a 2D linear weight into a micro-rotated INT8 layout.

    This is the artifact side of an MR-GPTQ-style INT8 path. It applies the
    block Hadamard rotation offline and emits per-row/per-rotation-group INT8
    scales. If ``hessian`` is supplied, it runs GPTQ error propagation in the
    rotated activation basis; otherwise it falls back to amax-derived diagonal
    importance for static act-order metadata and weighted MSE scale fitting.
    """

    cfg = config or MRGPTQInt8Config()
    _check_power_of_two(cfg.rotation_block_size, "rotation_block_size")
    if weight.ndim != 2:
        raise ValueError(f"MR-GPTQ INT8 only supports 2D weights, got shape={tuple(weight.shape)}")

    weight_f = weight.detach().float()
    activation_amax_f = activation_amax.detach().float().reshape(-1).abs()
    in_features = weight_f.shape[1]
    if activation_amax_f.numel() != in_features:
        raise ValueError(
            "activation_amax width must match weight input dimension: "
            f"{activation_amax_f.numel()} != {in_features}"
        )

    block_size = int(cfg.rotation_block_size)
    eps = float(cfg.eps)
    rotated = apply_block_hadamard(weight_f, block_size)
    if hessian is not None:
        hessian_f = hessian.detach().float()
        if hessian_f.shape != (in_features, in_features):
            raise ValueError(f"hessian shape must be {(in_features, in_features)}, got {tuple(hessian_f.shape)}")
        hessian_rot = rotate_hessian_block_hadamard(hessian_f.to(rotated.device), block_size)
        hessian_diag = torch.diag(hessian_rot).clamp_min(eps).to(torch.float32).cpu()
    else:
        hessian_rot = None
        hessian_diag = activation_amax_f.square().clamp_min(eps).to(torch.float32)

    qweight, weight_scale = _fit_group_int8_scales(rotated, hessian_diag, cfg)

    if hessian_rot is not None:
        qweight, activation_order = _apply_gptq_error_propagation(rotated, hessian_rot, weight_scale, cfg)
        gptq_error_propagation = "applied"
    elif cfg.static_activation_order:
        activation_order = torch.argsort(hessian_diag, descending=True).to(torch.int32).contiguous()
        gptq_error_propagation = "not_applied_activation_amax_only"
    else:
        activation_order = torch.arange(in_features, dtype=torch.int32)
        gptq_error_propagation = "not_applied"

    return MRGPTQInt8Params(
        weight_q=qweight.contiguous(),
        weight_scale=weight_scale.contiguous(),
        activation_order=activation_order.cpu().contiguous(),
        hessian_diag=hessian_diag.cpu().contiguous(),
        rotation_block_size=block_size,
        gptq_error_propagation=gptq_error_propagation,
    )
