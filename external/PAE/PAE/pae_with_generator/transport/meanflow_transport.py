"""
B3 / global sample-level MeanFlow transport and loss for PAE LightningDiT.

This file adds a MeanFlow training path without changing the upstream Flow
Matching transport.  The production recipe implemented here is:

    bf16/autocast backbone forward/backward
    + fp32/no-autocast live-parameter JVP target path
    + target = (v - du).detach()
    + sample-level scalar (r,t), default 75% r=t and 25% r<t

The orientation follows the prior PDM-3-HT experiments: x_data at t=0, noise at
t=1, v = noise - x_data, z_t = (1-t) * x_data + t * noise.  Sampling from noise
to data then uses z_r = z_t - (t-r) * u_theta(z_t, r, t, y).
"""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.func import jvp

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel as _sdpa_kernel
except Exception:  # pragma: no cover - compatibility fallback for older torch
    SDPBackend = None
    _sdpa_kernel = None

from .utils import mean_flat


MIXED_BF16_BACKBONE_FP32_JVP = "bf16_backbone_fp32_jvp"


def _expand_time(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return t.view(t.shape[0], *([1] * (x.ndim - 1)))


def _tensor_norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.detach().float()).cpu())


def _rel(reference: torch.Tensor, candidate: torch.Tensor, eps: float = 1e-12) -> float:
    return _tensor_norm(reference - candidate) / (_tensor_norm(reference) + eps)


def _finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x.detach()).all().cpu())


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_mb(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))


def _math_sdpa_context(device_type: str):
    """Force math SDPA for forward-mode AD / FD compatibility.

    PyTorch efficient/flash SDPA kernels support ordinary backward, but not
    forward-mode AD as of the torch 2.9 build used in this environment.  The
    MeanFlow target path uses ``torch.func.jvp`` over the DiT, so it must run
    under math SDPA even when the global training config enables the faster
    default/flash kernels for the ordinary bf16 train forward/backward path.
    """
    if device_type != "cuda":
        return nullcontext()
    if _sdpa_kernel is not None and SDPBackend is not None:
        return _sdpa_kernel(SDPBackend.MATH)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False,
            enable_math=True,
            enable_mem_efficient=False,
            enable_cudnn=False,
        )
    return nullcontext()


def _slice_model_kwargs(model_kwargs: Dict[str, Any], index: torch.Tensor, batch: int) -> Dict[str, Any]:
    """Slice batch-aligned tensor kwargs for a sub-batch model call.

    LightningDiT kwargs currently contain ``y`` and ``force_drop_ids`` with a
    leading batch dimension.  Keep non-tensors and non-batch-aligned tensors
    unchanged so this remains compatible with scalar/static kwargs.
    """
    out: Dict[str, Any] = {}
    for key, value in model_kwargs.items():
        if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == int(batch):
            out[key] = value[index]
        else:
            out[key] = value
    return out


class B3MeanFlowTransport:
    """MeanFlow loss helper for Phase-0/B3 PAE DiT training."""

    def __init__(
        self,
        *,
        precision_recipe: str = MIXED_BF16_BACKBONE_FP32_JVP,
        equal_prob: float = 0.75,
        time_sampler: str = "ltg",
        t_low: float = 0.05,
        t_high: float = 0.95,
        lognorm_mu: float = -0.4,
        lognorm_sigma: float = 1.0,
        fd_eps: float = 1e-2,
        device_type: str = "cuda",
    ) -> None:
        if not (0.0 <= equal_prob <= 1.0):
            raise ValueError("equal_prob must be in [0, 1]")
        if not (0.0 <= t_low < t_high <= 1.0):
            raise ValueError("need 0 <= t_low < t_high <= 1")
        if time_sampler not in {"uniform", "ltg"}:
            raise ValueError("time_sampler must be uniform or ltg")
        if precision_recipe not in {MIXED_BF16_BACKBONE_FP32_JVP, "fp32", "bf16_autocast"}:
            raise ValueError(f"unsupported precision_recipe={precision_recipe!r}")
        self.precision_recipe = precision_recipe
        self.equal_prob = equal_prob
        self.time_sampler = time_sampler
        self.t_low = t_low
        self.t_high = t_high
        self.lognorm_mu = lognorm_mu
        self.lognorm_sigma = lognorm_sigma
        self.fd_eps = fd_eps
        self.device_type = device_type

    @property
    def backbone_forward_mode(self) -> str:
        if self.precision_recipe == MIXED_BF16_BACKBONE_FP32_JVP:
            return "bf16_autocast"
        return self.precision_recipe

    @property
    def jvp_target_mode(self) -> str:
        if self.precision_recipe == MIXED_BF16_BACKBONE_FP32_JVP:
            return "fp32"
        return self.precision_recipe

    @property
    def fd_audit_mode(self) -> str:
        return self.jvp_target_mode

    def sample_t(self, batch: int, device: torch.device) -> torch.Tensor:
        if self.time_sampler == "uniform":
            q = torch.rand(batch, device=device, dtype=torch.float32)
        else:
            logits = self.lognorm_mu + self.lognorm_sigma * torch.randn(batch, device=device, dtype=torch.float32)
            q = torch.sigmoid(logits)
        return self.t_low + (self.t_high - self.t_low) * q

    def sample_r_t(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.sample_t(batch, device)
        r_candidate = torch.rand(batch, device=device, dtype=torch.float32) * t
        eq_mask = torch.rand(batch, device=device) < self.equal_prob
        r = torch.where(eq_mask, t, r_candidate)
        return r, t, eq_mask

    def sample_path(self, x_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        x_data = x_data.float()
        noise = torch.randn_like(x_data)
        r, t, eq_mask = self.sample_r_t(x_data.shape[0], x_data.device)
        v = noise - x_data
        z_t = (1.0 - _expand_time(t, x_data)) * x_data + _expand_time(t, x_data) * noise
        return {"x_data": x_data, "noise": noise, "v": v, "z_t": z_t, "r": r, "t": t, "eq_mask": eq_mask}

    def call_model(
        self,
        model: nn.Module,
        z: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        model_kwargs: Dict[str, Any],
        mode: str,
    ) -> torch.Tensor:
        if mode == "fp32":
            # Explicitly disable autocast for the target/JVP/FD path.
            dev_type = z.device.type
            with torch.autocast(device_type=dev_type, enabled=False):
                return model(z.float(), r.float(), t.float(), **model_kwargs).float()
        if mode == "bf16_autocast":
            if z.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return model(z, r, t, **model_kwargs).float()
            return model(z.float(), r.float(), t.float(), **model_kwargs).float()
        raise ValueError(f"unknown mode={mode!r}")

    def full_jvp(
        self,
        model: nn.Module,
        z_t: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        v: torch.Tensor,
        model_kwargs: Dict[str, Any],
        mode: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        mode = self.jvp_target_mode if mode is None else mode
        delta = t - r
        z_dot = _expand_time(delta, v) * v
        r_dot = torch.zeros_like(r)
        t_dot = delta

        def fn(z: torch.Tensor, rr: torch.Tensor, tt: torch.Tensor) -> torch.Tensor:
            return self.call_model(model, z, rr, tt, model_kwargs, mode)

        with _math_sdpa_context(z_t.device.type):
            u, du = jvp(fn, (z_t.float(), r.float(), t.float()), (z_dot.float(), r_dot.float(), t_dot.float()))
        return u, du, (z_dot, r_dot, t_dot)

    def central_fd(
        self,
        model: nn.Module,
        primals: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tangents: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        model_kwargs: Dict[str, Any],
        eps: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        eps = self.fd_eps if eps is None else eps
        mode = self.fd_audit_mode if mode is None else mode
        z, r, t = primals
        z_dot, r_dot, t_dot = tangents
        with _math_sdpa_context(z.device.type), torch.no_grad():
            u_plus = self.call_model(model, z + eps * z_dot, r + eps * r_dot, t + eps * t_dot, model_kwargs, mode)
            u_minus = self.call_model(model, z - eps * z_dot, r - eps * r_dot, t - eps * t_dot, model_kwargs, mode)
        return (u_plus - u_minus) / (2.0 * eps)

    def training_losses(
        self,
        model: nn.Module,
        x_data: torch.Tensor,
        model_kwargs: Optional[Dict[str, Any]] = None,
        *,
        return_diagnostics: bool = True,
        log_memory: bool = True,
    ) -> Dict[str, Any]:
        """Compute MeanFlow loss for a latent batch.

        Returns a dict compatible with the upstream transport convention: the
        per-sample tensor is stored in ``terms['loss']``.  Diagnostics are stored
        in ``terms['diagnostics']`` and contain only detached/scalar telemetry.
        """
        if model_kwargs is None:
            model_kwargs = {}
        device = x_data.device
        path = self.sample_path(x_data)
        z_t, r, t, v, eq_mask = path["z_t"], path["r"], path["t"], path["v"], path["eq_mask"]

        diagnostics: Dict[str, Any] = {
            "precision_recipe": self.precision_recipe,
            "backbone_forward_mode": self.backbone_forward_mode,
            "jvp_target_mode": self.jvp_target_mode,
            "fd_audit_mode": self.fd_audit_mode,
            "jvp_param_source": "live",
            "ema_used_for_target_jvp": False,
            "target_detached": True,
            "r_t_sampling_granularity": "sample_level",
            "configured_equal_prob": self.equal_prob,
            "r_eq_t_count": int(eq_mask.detach().sum().cpu()),
            "r_eq_t_total": int(eq_mask.numel()),
            "realized_r_eq_t_fraction": float(eq_mask.float().mean().detach().cpu()),
        }

        if log_memory:
            _reset_peak(device)
            _cuda_sync(device)
        target_start = time.perf_counter()
        with torch.no_grad():
            # Exact fast path for the MeanFlow degenerate samples.  For r == t,
            # delta = t - r = 0, hence the JVP tangent is identically zero
            # (z_dot = r_dot = t_dot = 0) and du == 0 by linearity.  Therefore
            # target = v - du is exactly v.  The default sampler intentionally
            # makes most samples degenerate (equal_prob=0.75), so avoiding a
            # full DiT JVP for those rows is a large H100 throughput win without
            # changing the mathematical target.
            non_eq_mask = ~eq_mask
            non_eq_count = int(non_eq_mask.detach().sum().cpu())
            target = v.detach().clone()
            u_jvp: Optional[torch.Tensor]
            du: Optional[torch.Tensor]
            if non_eq_count > 0:
                sub_kwargs = _slice_model_kwargs(model_kwargs, non_eq_mask, int(v.shape[0]))
                u_jvp, du, _ = self.full_jvp(
                    model,
                    z_t[non_eq_mask],
                    r[non_eq_mask],
                    t[non_eq_mask],
                    v[non_eq_mask],
                    sub_kwargs,
                    self.jvp_target_mode,
                )
                target[non_eq_mask] = (v[non_eq_mask] - du).detach()
            else:
                u_jvp = None
                du = None
        _cuda_sync(device)
        diagnostics.update(
            {
                "target_jvp_sec": time.perf_counter() - target_start,
                "target_jvp_peak_memory_mb": _peak_mb(device) if log_memory else None,
                "target_jvp_effective_batch": non_eq_count,
                "target_jvp_skipped_equal_batch": int(eq_mask.numel()) - non_eq_count,
                "target_requires_grad": bool(target.requires_grad),
                "target_jvp_u_finite": True if u_jvp is None else _finite(u_jvp),
                "du_finite": True if du is None else _finite(du),
                "target_finite": _finite(target),
                "du_norm": 0.0 if du is None else _tensor_norm(du),
                "v_norm": _tensor_norm(v),
                "du_over_v_norm_ratio": (0.0 if du is None else _tensor_norm(du)) / (_tensor_norm(v) + 1e-12),
            }
        )
        if bool(eq_mask.any().detach().cpu()):
            diagnostics["actual_rt_samples_target_minus_v_max_abs"] = float((target[eq_mask] - v[eq_mask]).detach().float().abs().max().cpu())
        else:
            diagnostics["actual_rt_samples_target_minus_v_max_abs"] = None

        if log_memory:
            _reset_peak(device)
            _cuda_sync(device)
        forward_start = time.perf_counter()
        u = self.call_model(model, z_t, r, t, model_kwargs, self.backbone_forward_mode)
        _cuda_sync(device)
        diagnostics.update(
            {
                "backbone_forward_sec": time.perf_counter() - forward_start,
                "backbone_forward_peak_memory_mb": _peak_mb(device) if log_memory else None,
                "u_finite": _finite(u),
            }
        )
        if self.backbone_forward_mode == "bf16_autocast" and self.jvp_target_mode == "fp32":
            if u_jvp is not None:
                diagnostics["bf16_forward_vs_fp32_rel_err"] = _rel(u_jvp, u[non_eq_mask])
                diagnostics["fp32_reference_forward_norm"] = _tensor_norm(u_jvp)
            else:
                diagnostics["bf16_forward_vs_fp32_rel_err"] = None
                diagnostics["fp32_reference_forward_norm"] = None
        del u_jvp, du

        loss = mean_flat((u.float() - target.float()) ** 2)
        diagnostics["loss_finite"] = bool(torch.isfinite(loss.detach()).all().cpu())
        diagnostics["loss_mean"] = float(loss.detach().float().mean().cpu())
        return {"loss": loss, "pred": u, "target": target, "diagnostics": diagnostics if return_diagnostics else {}}

    def fd_audit(
        self,
        model: nn.Module,
        x_data: torch.Tensor,
        model_kwargs: Optional[Dict[str, Any]] = None,
        *,
        log_memory: bool = True,
    ) -> Dict[str, Any]:
        if model_kwargs is None:
            model_kwargs = {}
        device = x_data.device
        path = self.sample_path(x_data)
        z_t, r, t, v, eq_mask = path["z_t"], path["r"], path["t"], path["v"], path["eq_mask"]
        audit_forced_non_degenerate = False
        # FD correctness must not be accidentally masked by an all-r=t batch.
        # Keep the training sampler unchanged, but force one audit sample to r<t
        # if the Bernoulli draw produced only degenerate samples.
        if bool(eq_mask.all().detach().cpu()) and r.numel() > 0:
            r = r.clone()
            eq_mask = eq_mask.clone()
            r[0] = 0.5 * t[0]
            eq_mask[0] = False
            audit_forced_non_degenerate = True
        record: Dict[str, Any] = {
            "fd_eps": self.fd_eps,
            "fd_mode": self.fd_audit_mode,
            "jvp_mode": self.jvp_target_mode,
            "r_eq_t_count": int(eq_mask.detach().sum().cpu()),
            "r_eq_t_total": int(eq_mask.numel()),
            "realized_r_eq_t_fraction": float(eq_mask.float().mean().detach().cpu()),
            "audit_forced_non_degenerate": audit_forced_non_degenerate,
        }

        if log_memory:
            _reset_peak(device)
            _cuda_sync(device)
        start = time.perf_counter()
        with torch.no_grad():
            _u, du, tangents = self.full_jvp(model, z_t, r, t, v, model_kwargs, self.jvp_target_mode)
        _cuda_sync(device)
        record["jvp_sec"] = time.perf_counter() - start
        record["jvp_peak_memory_mb"] = _peak_mb(device) if log_memory else None
        record["u_finite"] = _finite(_u)
        record["du_finite"] = _finite(du)
        record["du_norm"] = _tensor_norm(du)

        if log_memory:
            _reset_peak(device)
            _cuda_sync(device)
        start = time.perf_counter()
        du_fd = self.central_fd(model, (z_t, r, t), tangents, model_kwargs, self.fd_eps, self.fd_audit_mode)
        _cuda_sync(device)
        record["fd_sec"] = time.perf_counter() - start
        record["fd_peak_memory_mb"] = _peak_mb(device) if log_memory else None
        record["fd_finite"] = _finite(du_fd)
        record["fd_norm"] = _tensor_norm(du_fd)
        record["fd_rel_err_full_jvp"] = _rel(du_fd, du)
        record["fd_abs_err_norm"] = _tensor_norm(du_fd - du)

        with torch.no_grad():
            r_eq = t.clone()
            _u_eq, du_eq, _ = self.full_jvp(model, z_t, r_eq, t, v, model_kwargs, self.jvp_target_mode)
            target_eq = (v - du_eq).detach()
        record["all_r_eq_t_degenerate_du_norm"] = _tensor_norm(du_eq)
        record["all_r_eq_t_degenerate_target_minus_v_max_abs"] = float((target_eq - v).detach().float().abs().max().cpu())
        record["all_r_eq_t_degenerate_finite"] = _finite(du_eq) and _finite(target_eq)
        record["target_detached"] = not bool(target_eq.requires_grad)
        record["no_nan_or_inf"] = all(
            bool(record.get(k, False)) for k in ("u_finite", "du_finite", "fd_finite", "all_r_eq_t_degenerate_finite")
        )
        return record


def create_meanflow_transport(**kwargs: Any) -> B3MeanFlowTransport:
    return B3MeanFlowTransport(**kwargs)
