"""Pipeline policy resolution for Dataset4D -> BFExtractor -> ImageFFT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from fast_acbf.data.bf_extractor import BFExtractor
from fast_acbf.data.bf_preparer import BFPreparer, _check_integer_upscale_compat, _prepared_shapes
from fast_acbf.data.imagefft import ImageFFT

if TYPE_CHECKING:
    from fast_acbf.data.dataset4d import Dataset4D
    from fast_acbf.data.geometry import DetectorGeometry


_VALID_PIPELINES = ('speed', 'balanced', 'memory')
_VALID_STORAGE = ('auto', 'device', 'host', 'none')
_VALID_FILL = ('auto', 'precompute', 'lazy', 'on_the_fly')
_VALID_EXTRACTOR = (
    'auto',
    'device_mask',
    'host_mask',
    'disk_per_pixel',
    'disk_slab',
    'disk_scan_row',
)


@dataclass(frozen=True)
class PipelineResolution:
    pipeline: str
    imagefft_storage: str
    imagefft_fill: str
    extractor_strategy: str
    raw_bytes: int
    vbf_bytes: int
    imagefft_bytes: int


class PipelineManager:
    """Resolve pipeline policy and build the ImageFFT object consumed by recon."""

    def __init__(
        self,
        dataset: Dataset4D,
        detector_geom: DetectorGeometry,
        *,
        device: str,
        pipeline: str = 'balanced',
        imagefft_storage: str = 'auto',
        imagefft_fill: str = 'auto',
        extractor_strategy: str = 'auto',
        fft_batch_size: int = 64,
        vram_margin: float = 0.60,
        ram_margin: float = 0.80,
        pad_width: int | None = None,
        upscale: float = 1.0,
        upscale_method: str = 'zero_insert',
        upscale_sigma: float | None = None,
        lanczos_size: int | None = None,
    ) -> None:
        self.dataset = dataset
        self.detector_geom = detector_geom
        self.device = device
        self.pipeline = self._validate('pipeline', pipeline, _VALID_PIPELINES)
        self.imagefft_storage_request = self._validate(
            'imagefft_storage', imagefft_storage, _VALID_STORAGE
        )
        self.imagefft_fill_request = self._validate('imagefft_fill', imagefft_fill, _VALID_FILL)
        self.extractor_strategy_request = self._validate(
            'extractor_strategy', extractor_strategy, _VALID_EXTRACTOR
        )
        self.fft_batch_size = int(fft_batch_size)
        self.vram_margin = float(vram_margin)
        self.ram_margin = float(ram_margin)
        self.upscale = 1.0
        self.upscale_method = 'bilinear'
        self.upscale_sigma = None
        self.lanczos_size = None
        self.pad_width = None
        self._effective_scan_shape = dataset.scan_shape
        self.set_preparation(
            upscale=upscale,
            upscale_method=upscale_method,
            pad_width=pad_width,
            upscale_sigma=upscale_sigma,
            lanczos_size=lanczos_size,
        )

        self.resolution = self._resolve()
        self.extractor: BFExtractor | None = None
        self.preparer: BFPreparer | None = None
        self.imagefft: ImageFFT | None = None

    def set_preparation(
        self,
        *,
        upscale: float,
        upscale_method: str,
        pad_width: int | None,
        upscale_sigma: float | None = None,
        lanczos_size: int | None = None,
    ) -> None:
        if pad_width is not None and int(pad_width) < 0:
            raise ValueError(f"pad_width must be non-negative, got {pad_width}.")
        _check_integer_upscale_compat(str(upscale_method).strip().lower(), float(upscale))
        self.upscale = float(upscale)
        self.upscale_method = str(upscale_method).strip().lower()
        self.upscale_sigma = None if upscale_sigma is None else float(upscale_sigma)
        self.lanczos_size = None if lanczos_size is None else int(lanczos_size)
        self.pad_width = int(pad_width) if (pad_width is not None and int(pad_width) > 0) else None
        _, padded_shape, _, _, _ = _prepared_shapes(
            self.dataset.scan_shape, self.upscale, self.pad_width
        )
        self._effective_scan_shape = padded_shape

    @staticmethod
    def _validate(name: str, value: str, valid: tuple[str, ...]) -> str:
        value = str(value).strip().lower()
        if value not in valid:
            raise ValueError(f"{name} must be one of {valid}, got {value!r}.")
        return value

    @property
    def nb(self) -> int:
        return int(self.detector_geom.n_bf_pixels)

    @property
    def vbf_bytes(self) -> int:
        Ry, Rx = self._effective_scan_shape or self.dataset.scan_shape
        return int(self.nb * Ry * Rx * 4)

    @property
    def imagefft_bytes(self) -> int:
        return int(self.vbf_bytes * 2)  # complex64 = 2 x float32 bytes

    def build_full_pipeline(self) -> tuple[BFPreparer, ImageFFT]:
        self.extractor = BFExtractor(
            self.dataset,
            self.detector_geom,
            device=self.device,
            strategy=self.resolution.extractor_strategy,
        )
        self.preparer = BFPreparer(
            self.extractor,
            upscale=self.upscale,
            upscale_method=self.upscale_method,
            pad_width=self.pad_width,
            upscale_sigma=self.upscale_sigma,
            lanczos_size=self.lanczos_size,
        )
        self.imagefft = ImageFFT(
            self.preparer,
            device=self.device,
            storage=self.resolution.imagefft_storage,
            fill=self.resolution.imagefft_fill,
            batch_size=self.fft_batch_size,
        )
        return self.preparer, self.imagefft

    def rebuild_prepared_pipeline(
        self,
        *,
        upscale: float,
        upscale_method: str,
        pad_width: int | None,
        upscale_sigma: float | None = None,
        lanczos_size: int | None = None,
    ) -> tuple[BFPreparer, ImageFFT]:
        self.set_preparation(
            upscale=upscale,
            upscale_method=upscale_method,
            pad_width=pad_width,
            upscale_sigma=upscale_sigma,
            lanczos_size=lanczos_size,
        )
        self.resolution = self._resolve()
        return self.build_full_pipeline()

    def _resolve(self) -> PipelineResolution:
        raw_bytes = self.dataset.nbytes_float32
        vbf_bytes = self.vbf_bytes
        imagefft_bytes = self.imagefft_bytes

        storage = self._resolve_storage(raw_bytes, vbf_bytes, imagefft_bytes)
        fill = self._resolve_fill(storage)
        extractor_strategy = self._resolve_extractor_strategy(
            storage, fill, raw_bytes, vbf_bytes, imagefft_bytes
        )
        self._validate_resolved_combination(
            storage, fill, extractor_strategy, raw_bytes, vbf_bytes, imagefft_bytes
        )

        return PipelineResolution(
            pipeline=self.pipeline,
            imagefft_storage=storage,
            imagefft_fill=fill,
            extractor_strategy=extractor_strategy,
            raw_bytes=raw_bytes,
            vbf_bytes=vbf_bytes,
            imagefft_bytes=imagefft_bytes,
        )

    def _resolve_storage(self, raw_bytes: int, vbf_bytes: int, imagefft_bytes: int) -> str:
        if self.imagefft_storage_request != 'auto':
            requested = self.imagefft_storage_request
            if requested == 'host':
                available = self._available_ram()
                if available is not None and imagefft_bytes > int(available * self.ram_margin):
                    raise RuntimeError(
                        f"imagefft_storage='host' requires {imagefft_bytes / 2**30:.2f} GiB for "
                        f"the FFT cache but only {available / 2**30:.2f} GiB RAM is available. "
                        "Use imagefft_storage='none' to recompute on the fly, or reduce the scan "
                        "area or aperture to shrink the cache."
                    )
            elif requested == 'device':
                free_vram = self._free_vram()
                if free_vram is not None and imagefft_bytes > int(free_vram * self.vram_margin):
                    raise RuntimeError(
                        f"imagefft_storage='device' requires {imagefft_bytes / 2**30:.2f} GiB for "
                        f"the FFT cache but only {free_vram / 2**30:.2f} GiB VRAM is free. "
                        "Use imagefft_storage='host' or 'none'."
                    )
            return requested

        if self.pipeline == 'memory':
            return 'none'

        if self._device_imagefft_fits(imagefft_bytes):
            return 'device'
        if self._host_imagefft_fits(imagefft_bytes):
            return 'host'
        return 'none'

    def _resolve_fill(self, storage: str) -> str:
        if storage == 'none':
            if self.imagefft_fill_request not in ('auto', 'on_the_fly'):
                raise ValueError("imagefft_storage='none' requires imagefft_fill='on_the_fly'.")
            return 'on_the_fly'

        if self.imagefft_fill_request != 'auto':
            if self.imagefft_fill_request == 'on_the_fly':
                raise ValueError("imagefft_fill='on_the_fly' requires imagefft_storage='none'.")
            return self.imagefft_fill_request

        return 'precompute'

    def _resolve_extractor_strategy(
        self,
        storage: str,
        fill: str,
        raw_bytes: int,
        vbf_bytes: int,
        imagefft_bytes: int,
    ) -> str:
        if self.extractor_strategy_request != 'auto':
            return self.extractor_strategy_request

        if (
            storage != 'none'
            and fill == 'precompute'
            and self._can_use_device_mask(raw_bytes, vbf_bytes, imagefft_bytes)
        ):
            return 'device_mask'

        if not self.dataset.is_lazy:
            return 'host_mask'

        host_extra = imagefft_bytes if storage == 'host' else 0
        if (
            self.pipeline != 'memory'
            and storage != 'none'
            and self._host_raw_fits(raw_bytes, extra_bytes=host_extra)
        ):
            return 'host_mask'

        chunks = self.dataset.backend_chunks
        if chunks is None:
            return 'disk_scan_row' if fill == 'precompute' else 'disk_slab'
        c0, c1, c2, c3 = chunks
        if c2 == 1 and c3 == 1:
            return 'disk_per_pixel'
        if c0 == 1 and c1 == 1:
            return 'disk_scan_row'
        return 'disk_per_pixel'

    def _validate_resolved_combination(
        self,
        storage: str,
        fill: str,
        extractor_strategy: str,
        raw_bytes: int,
        vbf_bytes: int,
        imagefft_bytes: int,
    ) -> None:
        if extractor_strategy == 'device_mask':
            self._validate_device_mask(storage, fill, raw_bytes, vbf_bytes, imagefft_bytes)
        if extractor_strategy == 'host_mask' and self.dataset.is_lazy:
            host_extra = imagefft_bytes if storage == 'host' else 0
            if not self._host_raw_fits(raw_bytes, extra_bytes=host_extra):
                raise RuntimeError(
                    f"extractor_strategy='host_mask' requires {raw_bytes / 2**30:.2f} GiB "
                    f"for raw host data"
                    f"{' plus ImageFFT cache' if host_extra else ''}, but only "
                    f"{(self._available_ram() or 0) / 2**30:.2f} GiB RAM is available. "
                    "Use a disk extraction strategy or reduce the scan area."
                )

    def _validate_device_mask(
        self,
        storage: str,
        fill: str,
        raw_bytes: int,
        vbf_bytes: int,
        imagefft_bytes: int,
    ) -> None:
        if storage == 'none' or fill != 'precompute':
            raise ValueError(
                "extractor_strategy='device_mask' requires persistent ImageFFT storage "
                "and imagefft_fill='precompute'. Use extractor_strategy='host_mask' or a "
                "disk strategy for lazy/on-the-fly execution."
            )
        if self.dataset.is_lazy and not self._host_raw_fits(raw_bytes):
            raise RuntimeError(
                f"extractor_strategy='device_mask' must materialize the lazy raw source first, "
                f"requiring {raw_bytes / 2**30:.2f} GiB host RAM, but only "
                f"{(self._available_ram() or 0) / 2**30:.2f} GiB is available. "
                "Use a disk extraction strategy or materialize a smaller ROI."
            )
        dev = torch.device(self.device)
        if dev.type != 'cuda':
            return
        free_vram = self._free_vram()
        if free_vram is None:
            return
        needed = raw_bytes + vbf_bytes + imagefft_bytes
        if needed > int(free_vram * self.vram_margin):
            raise RuntimeError(
                f"extractor_strategy='device_mask' requires about {needed / 2**30:.2f} GiB "
                f"VRAM for raw data, vBF images, and ImageFFT, but only "
                f"{free_vram / 2**30:.2f} GiB is free. Use extractor_strategy='host_mask' "
                "or a disk strategy, or reduce the scan area/aperture."
            )

    def _can_use_device_mask(self, raw_bytes: int, vbf_bytes: int, imagefft_bytes: int) -> bool:
        if self.pipeline == 'memory':
            return False
        dev = torch.device(self.device)
        if dev.type != 'cuda':
            return False
        if self.dataset.is_lazy and self.pipeline != 'speed':
            return False
        if self.dataset.is_lazy and not self._host_raw_fits(raw_bytes):
            return False
        free_vram = self._free_vram()
        if free_vram is None:
            return False
        needed = raw_bytes + vbf_bytes + imagefft_bytes
        return needed <= int(free_vram * self.vram_margin)

    def _device_imagefft_fits(self, imagefft_bytes: int) -> bool:
        dev = torch.device(self.device)
        if dev.type != 'cuda':
            return False
        free_vram = self._free_vram()
        return free_vram is not None and imagefft_bytes <= int(free_vram * self.vram_margin)

    def _host_imagefft_fits(self, imagefft_bytes: int) -> bool:
        available = self._available_ram()
        return available is not None and imagefft_bytes <= int(available * self.ram_margin)

    def _host_raw_fits(self, raw_bytes: int, *, extra_bytes: int = 0) -> bool:
        available = self._available_ram()
        needed = raw_bytes + extra_bytes
        return available is not None and needed <= int(available * self.ram_margin)

    def _free_vram(self) -> int | None:
        dev = torch.device(self.device)
        if dev.type != 'cuda':
            return None
        free, _ = torch.cuda.mem_get_info(dev)
        return int(free)

    @staticmethod
    def _available_ram() -> int | None:
        try:
            import psutil
        except ImportError:
            return None
        return int(psutil.virtual_memory().available)


__all__ = ["PipelineManager", "PipelineResolution"]
