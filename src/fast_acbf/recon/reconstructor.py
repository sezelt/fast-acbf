"""BFReconstructor — minimal BF-specific optimizable model.

Owns: ImageFFT, DetectorGeometry, ScanGeometry, AberrationState, CoordinateTransform,
and the reconstruction cache workspace. Nothing else.

Strict boundaries:
  - no raw dataset/source access
  - no plotting, no refinement orchestration
  - no get_probe / get_chi_surface / get_yx_shifts_*
  - no metadata application
  - no post-construction mutation of DetectorGeometry or ScanGeometry
"""

from __future__ import annotations

import torch

from fast_acbf.core.aberrations import AberrationState
from fast_acbf.core.acbf import reconstruct_acbf, reconstruct_acbf_complex_inversion
from fast_acbf.core.tcbf import reconstruct_tcbf
from fast_acbf.data.geometry import CoordinateTransform, ScanGeometry
from fast_acbf.data.geometry import DetectorGeometry
from fast_acbf.data.imagefft import ImageFFT
from fast_acbf.recon.cache import (
    ACBFGeometryCache,
    ACBFOpticsCache,
    TCBFCache,
    build_acbf_geometry_cache,
    build_acbf_optics_cache,
    build_tcbf_cache,
)

_VALID_BASIS_MODES = ('on_the_fly', 'precompute')
_VALID_FOV = ('original', 'full')


class _ConstFFTProvider:
    """Serves a single precomputed FFT for every detector pixel.

    Used to build the sparse-upscale reweighting map (zero_insert / gaussian_kde / lanczos_kde):
    run the same reconstruction operator over a constant (ones-derived) input to
    measure per-pixel coverage.
    """

    def __init__(self, fft_single: torch.Tensor) -> None:
        self._fft = fft_single  # (1, Ry, Rx) complex64

    def get_chunk(self, b_start: int, b_end: int) -> torch.Tensor:
        return self._fft.expand(b_end - b_start, -1, -1)


def _crop_to_original(
    image: torch.Tensor,
    pad_offsets: tuple[int, int],
    crop_shape: tuple[int, int],
) -> torch.Tensor:
    """Crop a prepared padded image back to its active unpadded prepared FOV."""
    pad_y, pad_x = pad_offsets
    Ry, Rx = crop_shape
    return image[pad_y:pad_y + Ry, pad_x:pad_x + Rx]


class BFReconstructor:
    """BF-specific optimizable model.

    Do not add user-facing conveniences here. Those live on BFSolver.
    """

    def __init__(
        self,
        imagefft: ImageFFT,
        detector_geom: DetectorGeometry,
        scan_geom: ScanGeometry,
        ab_state: AberrationState,
        coord_transform: CoordinateTransform,
        basis_mode: str = 'on_the_fly',
        eps: float = 1e-3,
        crop_shape: tuple[int, int] | None = None,
        pad_offsets: tuple[int, int] | None = None,
        fov: str = 'original',
    ) -> None:
        self.imagefft = imagefft
        self.detector_geom = detector_geom
        self.scan_geom = scan_geom
        self.ab_state = ab_state
        self.coord_transform = coord_transform
        self.eps = eps
        self._crop_shape = crop_shape
        self._pad_offsets = pad_offsets
        fov_str = str(fov)
        if fov_str not in _VALID_FOV:
            raise ValueError(f"fov must be one of {_VALID_FOV}, got {fov_str!r}.")
        self._fov = fov_str
        self._tcbf_cache: dict[tuple, TCBFCache] = {}
        self._acbf_cache: dict[tuple, tuple[ACBFGeometryCache, ACBFOpticsCache | None]] = {}
        self.basis_mode = basis_mode  # validated via property setter (accesses _acbf_cache)

    @property
    def basis_mode(self) -> str:
        return self._basis_mode

    @basis_mode.setter
    def basis_mode(self, value: str) -> None:
        value = str(value).strip().lower()
        if value not in _VALID_BASIS_MODES:
            raise ValueError(f"basis_mode must be one of {_VALID_BASIS_MODES}, got {value!r}.")
        self._basis_mode = value
        # Invalidate precomputed optics when switching modes so they rebuild correctly.
        for key in self._acbf_cache:
            geometry, _ = self._acbf_cache[key]
            self._acbf_cache[key] = (geometry, None)

    @property
    def device(self) -> str:
        return self.imagefft.device

    @property
    def rotation_deg(self) -> float:
        return self.coord_transform.rotation_deg

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Full reset — clears both basis caches and the ImageFFT cache."""
        self._tcbf_cache = {}
        self._acbf_cache = {}
        self.imagefft.clear()

    def clear_basis_cache(self) -> None:
        """Clear orientation-dependent basis caches; ImageFFT cache is preserved."""
        self._tcbf_cache = {}
        self._acbf_cache = {}

    def replace_prepared_data(
        self,
        imagefft: ImageFFT,
        scan_geom: ScanGeometry,
        *,
        crop_shape: tuple[int, int] | None,
        pad_offsets: tuple[int, int] | None,
    ) -> None:
        """Replace prepared FFT provider and scan geometry, then clear derived caches."""
        self.imagefft = imagefft
        self.scan_geom = scan_geom
        self._crop_shape = crop_shape
        self._pad_offsets = pad_offsets
        self.clear_basis_cache()

    def set_coord_transform(self, ct: CoordinateTransform, clear_basis: bool = True) -> None:
        """Replace the coordinate transform. Clears basis cache by default."""
        self.coord_transform = ct
        if clear_basis:
            self.clear_basis_cache()

    # ------------------------------------------------------------------
    # Internal frame helpers — delegate to CoordinateTransform
    # ------------------------------------------------------------------

    def _frame_cache_key(self) -> tuple:
        ct = self.coord_transform
        return (ct.rotation_deg, ct.flipud, ct.fliplr, ct.transpose)

    def _get_scan_frame_coeffs(self) -> torch.Tensor:
        return self.ab_state.to_scan_frame(self.coord_transform.rotation_deg)

    def _get_transformed_centers(self, in_scan_frame: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        det = self.detector_geom
        return self.coord_transform.apply_to_centers(det.kY_centers, det.kX_centers, in_scan_frame)

    def _get_transformed_grids(self, in_scan_frame: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        det = self.detector_geom
        return self.coord_transform.apply_to_grids(det.kY_grid, det.kX_grid, in_scan_frame)

    # ------------------------------------------------------------------
    # Lazy-build cache accessors
    # ------------------------------------------------------------------

    def _get_tcbf_cache(self, chunk_size: int = 64) -> TCBFCache:
        key = (chunk_size, *self._frame_cache_key())
        if key not in self._tcbf_cache:
            kX_full, kY_full = self._get_transformed_centers()
            self._tcbf_cache[key] = build_tcbf_cache(
                kX_full, kY_full,
                self.ab_state.order_keys, self.detector_geom.wavelength, chunk_size,
            )
        return self._tcbf_cache[key]

    def _get_acbf_cache(
        self, rolloff: float = 0, chunk_size: int = 64,
    ) -> tuple[ACBFGeometryCache, ACBFOpticsCache | None]:
        """Return (geometry, optics_or_None).

        Geometry is mode-independent. Optics is built on demand only in
        'precompute' mode; the consumer-facing return is gated on basis_mode.
        """
        key = (rolloff, chunk_size, *self._frame_cache_key())
        if key not in self._acbf_cache:
            kX_full, kY_full = self._get_transformed_centers()
            det = self.detector_geom
            geometry = build_acbf_geometry_cache(
                kX_full, kY_full,
                self.ab_state.order_keys, det.max_alpha, det.wavelength,
                self.ab_state.max_order, rolloff, chunk_size,
            )
            self._acbf_cache[key] = (geometry, None)

        geometry, optics = self._acbf_cache[key]
        if self.basis_mode == 'precompute':
            if optics is None:
                sg = self.scan_geom
                optics = build_acbf_optics_cache(geometry, sg.qx_grid, sg.qy_grid)
                self._acbf_cache[key] = (geometry, optics)
            return geometry, optics
        return geometry, None

    # ------------------------------------------------------------------
    # Reconstruction
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_acbf_algorithm(acbf_algorithm) -> str:
        if acbf_algorithm is None:
            return 'phase_only'
        return str(acbf_algorithm).strip().lower().replace('-', '_')

    def _get_recon_grids(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (qx_grid, qy_grid) for the active prepared scan grid."""
        sg = self.scan_geom
        return sg.qx_grid, sg.qy_grid

    def _reconstruct_impl(self, mode: str = 'tcBF', **kwargs) -> torch.Tensor:
        fov = str(kwargs.get('fov', self._fov))
        if fov not in _VALID_FOV:
            raise ValueError(f"fov must be one of {_VALID_FOV}, got {fov!r}.")

        qx_grid, qy_grid = self._get_recon_grids()
        coeffs = self._get_scan_frame_coeffs()
        mode_key = mode.lower()

        if mode_key == 'tcbf':
            cache = self._get_tcbf_cache(chunk_size=kwargs.get('chunk_size', 64))
            result = reconstruct_tcbf(
                self.imagefft, qx_grid, qy_grid, cache, coeffs, self.device,
            )

        elif mode_key == 'acbf':
            extractor = self.imagefft.extractor
            if (
                hasattr(extractor, 'upscale_method')
                and extractor.upscale_method == 'zero_insert'
                and extractor.upscale != 1.0
            ):
                raise ValueError(
                    "upscale_method='zero_insert' is not supported for acBF reconstruction "
                    "because the phase-alignment transfer function is not sparse-grid compatible. "
                    "Use upscale_method='gaussian_kde', 'lanczos_kde', 'nearest', or 'bilinear' instead."
                )
            rolloff = kwargs.get('rolloff', 0)
            chunk_size = kwargs.get('chunk_size', 64)
            acbf_algorithm = self._normalize_acbf_algorithm(kwargs.get('acbf_algorithm'))
            geometry, optics = self._get_acbf_cache(rolloff=rolloff, chunk_size=chunk_size)

            if acbf_algorithm == 'phase_only':
                result = reconstruct_acbf(
                    self.imagefft, qx_grid, qy_grid, geometry, optics,
                    coeffs, self.eps, self.device,
                )
            elif acbf_algorithm == 'complex_inversion':
                result = reconstruct_acbf_complex_inversion(
                    self.imagefft, qx_grid, qy_grid, geometry, optics,
                    coeffs, self.device,
                    regularization=kwargs.get('regularization', 1e-3),
                    support_threshold=kwargs.get('support_threshold', 1e-6),
                )
            else:
                raise ValueError(
                    f"Unsupported acBF algorithm {acbf_algorithm!r}. "
                    "Choose 'phase_only' or 'complex_inversion'."
                )

        else:
            raise ValueError(f"Unsupported mode {mode!r}. Choose 'tcBF' or 'acBF'.")

        # Sparse-upscale reweighting — normalize for non-uniform sub-pixel coverage (tcBF only).
        # Applied to zero_insert, gaussian_kde, and lanczos_kde; the weight map is ~1 in the
        # interior and <1 at the tapered edges.
        if mode_key == 'tcbf':
            extractor = self.imagefft.extractor
            if (
                hasattr(extractor, 'upscale_method')
                and extractor.upscale_method in ('zero_insert', 'gaussian_kde', 'lanczos_kde')
                and extractor.upscale != 1.0
            ):
                ones_provider = _ConstFFTProvider(extractor.make_ones_fft(self.device))
                weight_map = reconstruct_tcbf(
                    ones_provider, qx_grid, qy_grid, cache, coeffs, self.device,
                )
                peak = weight_map.abs().max().clamp(min=1e-9)
                result = result / weight_map.clamp(min=1e-3 * peak)

        # fov crop — only when padding was used
        if self._pad_offsets is not None and fov == 'original':
            result = _crop_to_original(result, self._pad_offsets, self._crop_shape)

        return result

    def reconstruct(self, mode: str = 'tcBF', requires_grad: bool = False, **kwargs) -> torch.Tensor:
        """Run reconstruction. No-grad by default; opt-in for AD optimization paths."""
        if requires_grad:
            return self._reconstruct_impl(mode=mode, **kwargs)
        with torch.no_grad():
            return self._reconstruct_impl(mode=mode, **kwargs)
