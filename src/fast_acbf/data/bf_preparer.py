"""BFPreparer -- real-space vBF preparation before FFT caching."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from fast_acbf.data.bf_extractor import BFExtractor


_VALID_UPSCALE_METHODS = ("nearest", "bilinear", "zero_insert", "gaussian_kde", "lanczos_kde")
# Methods implemented via integer-stride zero-insertion; they require integer upscale.
_INTEGER_ONLY_UPSCALE_METHODS = ("zero_insert", "gaussian_kde", "lanczos_kde")
# KDE methods use kernel density estimation: a normalized kernel profile is placed
# at each native pixel (zero-inserted on the stride-N grid) and the profiles are summed.
_KDE_UPSCALE_METHODS = ("gaussian_kde", "lanczos_kde")
# gaussian_kde kernel width as a multiple of the upscale factor N (in upscaled pixels).
# sigma = N/2 makes each native pixel's Gaussian reach its neighbours at the grid stride.
_GAUSSIAN_SIGMA_RATIO = 0.5
# lanczos_kde default half-width a of the 1D profile sinc(x) * sinc(x/a) on [-a, a]
# (upscaled pixels); r = a is the 1D kernel radius.
_LANCZOS_DEFAULT_SIZE = 3
# Gaussian truncation radius as a multiple of sigma for both KDE kernels.
_SIGMA_TRUNC = 4.0
# Cap on KDE convolution temporaries per sub-batch of detector pixels (bytes);
# the KDE branch holds ~3 upscaled-size temporaries per pixel, so whole-stack
# calls are sub-batched to keep peak memory bounded.
_KDE_TEMP_CAP_BYTES = 256 * 1024**2


def _ceil_5smooth(n: int) -> int:
    """Smallest 5-smooth integer >= n. (Only prime factors 2, 3, 5.)"""
    candidate = n
    while True:
        x = candidate
        for p in (2, 3, 5):
            while x % p == 0:
                x //= p
        if x == 1:
            return candidate
        candidate += 1


def _compute_pad_for_axis(orig: int, min_pad: int) -> tuple[int, int, int]:
    """
    Find the smallest 5-smooth size >= orig + 2*min_pad.
    Returns (padded_size, pad_before, pad_after).
    """
    target = orig + 2 * min_pad
    padded = _ceil_5smooth(target)
    extra = padded - orig
    pad_before = extra // 2
    pad_after = extra - pad_before
    return padded, pad_before, pad_after


def _make_tukey_1d(total: int, pad_before: int, pad_after: int) -> np.ndarray:
    """Tukey window with flat unit central region and cosine pad rolloff."""
    w = np.ones(total, dtype=np.float32)
    if pad_before > 0:
        i = np.arange(pad_before, dtype=np.float32)
        w[:pad_before] = 0.5 * (1.0 - np.cos(np.pi * i / pad_before))
    if pad_after > 0:
        i = np.arange(pad_after, dtype=np.float32)
        w[total - pad_after:] = 0.5 * (1.0 - np.cos(np.pi * (pad_after - 1 - i) / pad_after))
    return w


def _make_tukey_2d(
    Ry: int, Rx: int,
    pad_top: int, pad_bottom: int,
    pad_left: int, pad_right: int,
) -> np.ndarray:
    """Outer product of per-axis Tukey windows. Shape (Ry, Rx), float32."""
    wy = _make_tukey_1d(Ry, pad_top, pad_bottom)
    wx = _make_tukey_1d(Rx, pad_left, pad_right)
    return np.outer(wy, wx).astype(np.float32)


def _upsampled_hat(N: int) -> np.ndarray:
    """1-D upsampled hat (linear B-spline) kernel on the upscaled grid.

    The triangle t[m] = 1 - |m|/N for |m| <= N (length 2N+1, sum N). Folded
    into the KDE profile because:

    * Its spectrum T̂(ω) = (1/N) sin²(πNω)/sin²(πω) is exactly 0 at every
      fold point ω = k/N (k != 0), where the zero-inserted native spectrum
      has its DC-replica spikes -- so the upsampled field has no period-N
      structure in q (no comb peaks, no dark lines).
    * N-shifted copies sum to exactly 1 (discrete partition of unity), so a
      constant field upsamples to a constant.

    Same kernel bilinear (F.interpolate) uses, which is why bilinear is
    already comb-free.
    """
    m = np.arange(-N, N + 1, dtype=np.float64)
    return 1.0 - np.abs(m) / N


def _make_gaussian_kernel_2d(sigma: float, upscale: int) -> np.ndarray:
    """Normalized separable 2D KDE upscaling kernel.

    A Gaussian of the given sigma (in upscaled pixels, truncated at 4 sigma)
    folded with the upsampled hat :func:`_upsampled_hat` so that the kernel
    spectrum has exact nulls at every fold point ω = k/N (k != 0), which
    removes the zero-insertion comb from the upsampled field (see
    _upsampled_hat). Shape (2r+1, 2r+1), float32, sums to 1.
    """
    N = int(upscale)
    r = max(1, int(np.ceil(_SIGMA_TRUNC * sigma)))
    g = np.exp(-0.5 * (np.arange(-r, r + 1, dtype=np.float64) / sigma) ** 2)
    g = np.convolve(g, _upsampled_hat(N), mode="full")
    k = np.outer(g, g)
    return (k / k.sum()).astype(np.float32)


def _make_lanczos_kernel_2d(size: int, upscale: int, sigma: float | None = None) -> np.ndarray:
    """Normalized separable 2D Lanczos upsampling kernel (optionally Gaussian-smoothed).

    The 1D profile on the upscaled grid is sinc(x/N) * sinc(x/(a*N)) on
    [-a*N, a*N] (N = upscale, a = size): the standard Lanczos lowpass that
    completes an N-fold zero-insertion, with cutoff at the baseband edge
    (pi/N). If sigma is given, the profile is first convolved with a Gaussian
    of that sigma (truncated at 4 sigma), yielding a single combined kernel
    rather than a second filter pass. Finally the profile is folded with the
    upsampled hat (see :func:`_upsampled_hat`): plain Lanczos nulls its
    spectrum at ω = ±k/N only for k ≠ 1, while the hat nulls at ALL nonzero
    k/N, so the combined kernel removes the upsampled-vBF comb at every fold
    point (support (a+1)N). The result sums to 1.
    """
    a = int(size)
    if a < 1:
        raise ValueError(f"lanczos_size must be a positive integer, got {size!r}.")
    N = int(upscale)
    r = a * N
    x = np.arange(-r, r + 1, dtype=np.float64)
    h = np.sinc(x / N) * np.sinc(x / r)
    if sigma is not None:
        if sigma <= 0:
            raise ValueError(f"upscale_sigma must be positive, got {sigma!r}.")
        rg = max(1, int(np.ceil(_SIGMA_TRUNC * sigma)))
        y = np.arange(-rg, rg + 1, dtype=np.float64)
        g = np.exp(-0.5 * (y / sigma) ** 2)
        h = np.convolve(h, g, mode="full")
        r += rg
    h = np.convolve(h, _upsampled_hat(N), mode="full")
    k = np.outer(h, h)
    return (k / k.sum()).astype(np.float32)


def _check_integer_upscale_compat(method: str, upscale: float) -> None:
    """Raise before any state mutation if an integer-only method gets a non-integer upscale."""
    if method in _INTEGER_ONLY_UPSCALE_METHODS and abs(upscale - round(upscale)) > 1e-6:
        raise ValueError(
            f"upscale_method='{method}' requires an integer upscale factor, "
            f"got upscale={upscale!r}. Use 'nearest' or 'bilinear' for fractional upscale."
        )


def _normalize_upscale_method(method: str) -> str:
    method = str(method).strip().lower()
    if method not in _VALID_UPSCALE_METHODS:
        raise ValueError(
            f"upscale_method must be one of {_VALID_UPSCALE_METHODS}, got {method!r}."
        )
    return method


def _prepared_shapes(
    raw_shape: tuple[int, int],
    upscale: float,
    pad_width: int | None,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int, int, int], bool]:
    """Return upscaled/padded shapes and pad metadata for a preparation state."""
    if upscale < 1.0:
        raise ValueError(f"upscale must be >= 1.0, got {upscale}.")
    Ry, Rx = raw_shape
    Ry_up = round(Ry * upscale)
    Rx_up = round(Rx * upscale)
    if Ry_up <= 0 or Rx_up <= 0:
        raise ValueError(f"upscale produces invalid shape {(Ry_up, Rx_up)}.")

    if pad_width is None:
        return (Ry_up, Rx_up), (Ry_up, Rx_up), (0, 0), (0, 0, 0, 0), False

    pad_width = int(pad_width)
    if pad_width < 0:
        raise ValueError(f"pad_width must be non-negative, got {pad_width}.")
    if pad_width == 0:
        return (Ry_up, Rx_up), (Ry_up, Rx_up), (0, 0), (0, 0, 0, 0), False

    effective_pad = round(pad_width * upscale)
    if effective_pad <= 0:
        return (Ry_up, Rx_up), (Ry_up, Rx_up), (0, 0), (0, 0, 0, 0), False
    if effective_pad >= min(Ry_up, Rx_up):
        raise ValueError(
            f"effective pad_width={effective_pad} must be < min(Ry={Ry_up}, Rx={Rx_up}) "
            "after upscaling."
        )

    Ry_p, pad_top, pad_bottom = _compute_pad_for_axis(Ry_up, effective_pad)
    Rx_p, pad_left, pad_right = _compute_pad_for_axis(Rx_up, effective_pad)
    if max(pad_top, pad_bottom) >= Ry_up or max(pad_left, pad_right) >= Rx_up:
        raise ValueError(
            f"Effective padding after 5-smooth rounding "
            f"(top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}) "
            f"equals or exceeds upscaled input dimension (Ry={Ry_up}, Rx={Rx_up}). "
            "Reduce pad_width."
        )
    return (
        (Ry_up, Rx_up),
        (Ry_p, Rx_p),
        (pad_top, pad_left),
        (pad_top, pad_bottom, pad_left, pad_right),
        True,
    )


class BFPreparer:
    """Wraps BFExtractor; applies real-space upscale, padding, and windowing."""

    def __init__(
        self,
        extractor: BFExtractor,
        upscale: float = 1.0,
        upscale_method: str = "zero_insert",
        pad_width: int | None = None,
        upscale_sigma: float | None = None,
        lanczos_size: int | None = None,
    ) -> None:
        method = _normalize_upscale_method(upscale_method)
        _check_integer_upscale_compat(method, float(upscale))
        if upscale_sigma is not None:
            upscale_sigma = float(upscale_sigma)
            if upscale_sigma <= 0:
                raise ValueError(f"upscale_sigma must be positive, got {upscale_sigma}.")
            if method not in _KDE_UPSCALE_METHODS:
                raise ValueError(
                    f"upscale_sigma is only used with upscale_method='gaussian_kde' or "
                    f"'lanczos_kde', got {method!r}."
                )
        if lanczos_size is not None:
            if int(lanczos_size) != lanczos_size:
                raise ValueError(f"lanczos_size must be an integer, got {lanczos_size!r}.")
            lanczos_size = int(lanczos_size)
            if lanczos_size < 1:
                raise ValueError(f"lanczos_size must be a positive integer, got {lanczos_size}.")
            if method != "lanczos_kde":
                raise ValueError(
                    f"lanczos_size is only used with upscale_method='lanczos_kde', "
                    f"got {method!r}."
                )
        self._extractor = extractor
        self.upscale = float(upscale)
        self.upscale_method = method
        self.upscale_sigma = upscale_sigma
        self.lanczos_size = lanczos_size
        self.pad_width = None if pad_width is None or int(pad_width) == 0 else int(pad_width)

        self.raw_shape = tuple(extractor.scan_shape)
        (
            self.upscaled_shape,
            self.padded_shape,
            self.pad_offsets,
            pads,
            self._has_pad,
        ) = _prepared_shapes(self.raw_shape, self.upscale, self.pad_width)
        self.pad_top, self.pad_bottom, self.pad_left, self.pad_right = pads
        self._window_np = (
            _make_tukey_2d(
                self.padded_shape[0],
                self.padded_shape[1],
                self.pad_top,
                self.pad_bottom,
                self.pad_left,
                self.pad_right,
            )
            if self._has_pad
            else None
        )
        if self.upscale_method in _KDE_UPSCALE_METHODS and self.upscaled_shape != self.raw_shape:
            N = round(self.upscale)
            if self.upscale_method == "gaussian_kde":
                sigma = self.upscale_sigma if self.upscale_sigma is not None else _GAUSSIAN_SIGMA_RATIO * N
                self._kde_kernel_np = _make_gaussian_kernel_2d(sigma, N)
            else:
                a = self.lanczos_size if self.lanczos_size is not None else _LANCZOS_DEFAULT_SIZE
                self._kde_kernel_np = _make_lanczos_kernel_2d(a, N, self.upscale_sigma)
            # 1D profile for the separable two-pass convolution: the 2D kernel is an
            # outer product k = g g^T / S, so a = k[mid, :] / sqrt(k[mid, mid])
            # satisfies a @ a^T == k (two 1D passes == the 2D conv), but the 1D
            # passes avoid the K^2 x image-size im2col workspace oneDNN allocates
            # for conv2d. a is scaled by N so its sum over N-shifts is exactly 1
            # per pass (the zero-inserted field is nonzero only at the stride-N
            # grid); without this, a flat field would be attenuated by 1/N^2.
            k = self._kde_kernel_np
            mid = k.shape[0] // 2
            self._kde_profile_np = (k[mid, :] / np.sqrt(k[mid, mid])) * N
            # Native pixels of reflect-extended border needed so the kernel reaches
            # edge pixels completely (see _upsample); at least 1 for odd kernels.
            self._kde_native_pad = max(1, int(np.ceil(self._kde_kernel_np.shape[-1] // 2 / N)))
            # Coverage pattern: the upsampled ones field, normalized to mean 1.
            # With the partition-of-unity profile the upsampled ones field is
            # exactly 1 everywhere (border included), so the division in
            # _upsample is a mathematical no-op, kept only as a float32-drift
            # guard. (Set to None first so the _upsample call skips it.)
            self._kde_ones_up_np = None
            ones_up = self._upsample(np.ones((1, *self.raw_shape), dtype=np.float32))[0]
            if ones_up.min() > 0:
                self._kde_ones_up_np = ones_up / ones_up.mean()
        else:
            self._kde_kernel_np = None
            self._kde_profile_np = None
            self._kde_native_pad = 0
            self._kde_ones_up_np = None

    # Duck-typed ImageFFT extractor interface
    @property
    def nb(self) -> int:
        return self._extractor.nb

    @property
    def scan_shape(self) -> tuple[int, int]:
        return self.padded_shape

    @property
    def strategy(self) -> str:
        return self._extractor.strategy

    @property
    def bf_iy(self):
        return self._extractor.bf_iy

    @property
    def bf_ix(self):
        return self._extractor.bf_ix

    def extract_chunk(self, b_start: int, b_end: int):
        return self._prepare(self._extractor.extract_chunk(b_start, b_end))

    def extract_all(self):
        return self._prepare(self._extractor.extract_all())

    def _prepare(self, vbf):
        if self.upscaled_shape != self.raw_shape:
            vbf = self._upsample(vbf)
        if self._has_pad:
            vbf = self._pad_and_window(vbf)
        return vbf

    def _upsample(self, vbf):
        is_tensor = isinstance(vbf, torch.Tensor)
        if is_tensor:
            src = vbf.float()
            device = src.device
        else:
            src = torch.from_numpy(np.asarray(vbf, dtype=np.float32))
            device = torch.device("cpu")

        inp = src.to(device).unsqueeze(1)  # (Nb, 1, Ry, Rx)
        if self.upscale_method == "zero_insert":
            # Insert values at the stride-N grid positions — trailing N-1 rows/cols empty.
            N = round(self.upscale)
            Ry_up, Rx_up = self.upscaled_shape
            out = torch.zeros(inp.shape[0], 1, Ry_up, Rx_up, dtype=inp.dtype, device=inp.device)
            out[:, :, ::N, ::N] = inp
        elif self.upscale_method in _KDE_UPSCALE_METHODS:
            # KDE: each native pixel contributes a normalized kernel profile; summing
            # them is a convolution of the zero-inserted image with the shared kernel
            # (symmetric kernel, so correlation == convolution), run as two 1D passes
            # (K x 1, then 1 x K) to avoid the K^2 im2col workspace of a 2D conv.
            #
            # The native data is reflect-extended by p pixels per side rather than
            # zero-padded (which would truncate edge profiles and taper the border),
            # so every kernel is fully supported; the output is cropped back.
            #
            # Pixels are processed in sub-batches so the per-call temporaries (~3
            # upscaled-size images) stay bounded regardless of stack size.
            #
            # Each result is divided by the cached coverage pattern (see __init__);
            # a no-op in exact arithmetic, guarding float32 drift only.
            N = round(self.upscale)
            p = self._kde_native_pad
            Ry, Rx = inp.shape[-2], inp.shape[-1]
            Ry_up, Rx_up = self.upscaled_shape
            mode = "reflect" if p < min(Ry, Rx) else "replicate"
            prof = torch.as_tensor(self._kde_profile_np, dtype=inp.dtype, device=inp.device)
            ones = (
                torch.as_tensor(self._kde_ones_up_np, dtype=inp.dtype, device=inp.device)
                if self._kde_ones_up_np is not None
                else None
            )
            r = prof.numel() // 2
            out = torch.empty(inp.shape[0], 1, Ry_up, Rx_up, dtype=inp.dtype, device=inp.device)
            Ry_ext, Rx_ext = (Ry + 2 * p) * N, (Rx + 2 * p) * N
            # Temp bytes per detector pixel: zero-inserted extended input + the two
            # 1D-pass outputs (each Ry_ext x Rx_ext) plus the final crop copy
            # (Ry_up x Rx_up), float32.
            per_px = (2 * Ry_ext * Rx_ext + Ry_up * Rx_up) * 4
            bs = max(1, _KDE_TEMP_CAP_BYTES // max(1, per_px))
            for b0 in range(0, inp.shape[0], bs):
                b1 = min(b0 + bs, inp.shape[0])
                seg = inp[b0:b1]
                ext = F.pad(seg, (p, p, p, p), mode=mode)  # (bs, 1, Ry+2p, Rx+2p)
                zi = torch.zeros(b1 - b0, 1, Ry_ext, Rx_ext, dtype=inp.dtype, device=inp.device)
                zi[:, :, ::N, ::N] = ext
                zi = F.conv2d(zi, prof.view(1, 1, prof.numel(), 1), padding=(r, 0))
                zi = F.conv2d(zi, prof.view(1, 1, 1, prof.numel()), padding=(0, r))
                out[b0:b1] = zi[..., p * N: p * N + Ry_up, p * N: p * N + Rx_up]
                if ones is not None:
                    # Divisive coverage normalization (mean-1 pattern); a no-op in
                    # exact arithmetic, guarding float32 accumulation drift.
                    out[b0:b1].div_(ones)
        elif self.upscale_method == "nearest":
            out = F.interpolate(inp, size=self.upscaled_shape, mode="nearest")
        else:
            out = F.interpolate(
                inp,
                size=self.upscaled_shape,
                mode="bilinear",
                align_corners=False,
            )
        out = out.squeeze(1).contiguous()
        if is_tensor:
            return out
        return out.cpu().numpy().astype(np.float32, copy=False)

    def make_ones_fft(self, device: str) -> torch.Tensor:
        """Return fft2 of a prepared ones vBF used to build the upscale reweighting map.

        Runs a single all-ones image through the same _prepare pipeline (zero-insertion,
        KDE convolution, reflect-padding, Tukey windowing) so the resulting FFT
        accounts for all spatial modulations. Returns shape (1, padded_Ry, padded_Rx)
        complex64 on `device`.
        """
        ones = np.ones((1, *self.raw_shape), dtype=np.float32)
        prepared = self._prepare(ones)  # (1, padded_Ry, padded_Rx)
        t = torch.from_numpy(np.asarray(prepared, dtype=np.float32)).to(device)
        return torch.fft.fft2(t)

    def _pad_and_window(self, vbf):
        """vBF shape: (B, Ry, Rx). Returns same array/tensor type."""
        if isinstance(vbf, torch.Tensor):
            padded = F.pad(
                vbf,
                (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom),
                mode="reflect",
            )
            w = torch.as_tensor(self._window_np, dtype=torch.float32, device=vbf.device)
            return padded * w

        padded = np.pad(
            vbf,
            ((0, 0), (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right)),
            mode="reflect",
        ).astype(np.float32)
        return padded * self._window_np


__all__ = [
    "BFPreparer",
    "_check_integer_upscale_compat",
    "_compute_pad_for_axis",
    "_make_gaussian_kernel_2d",
    "_make_lanczos_kernel_2d",
    "_prepared_shapes",
]
