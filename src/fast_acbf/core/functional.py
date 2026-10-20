"""Differentiable physics primitives — no state, no side effects. Portable to PtyRAD."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ptyrad.core.functional import fftshift2, ifftshift2, torch_phasor


def imshift_with_batch(imgs, shifts, grid, batch_size=None):
    """
    Generates a batch of shifted images from an input stack of images.

    This function shifts complex/real-valued input images by applying phase shifts
    in the Fourier domain, achieving subpixel shifts in both x and y directions.
    Optionally applies a CTF correction in the same operation.

    Args:
        imgs (torch.Tensor): The input images to be shifted.
                             Shape must be (Nb, ..., Ny, Nx).
        shifts (torch.Tensor): The shifts to be applied. Shape is (Nb, 2),
                               where each slice is (shift_y, shift_x) in pixels.
        grid (torch.Tensor): The k-space grid used for computing the shifts.
                             Shape=(2, Ny, Nx), spanning [-0.5, 0.5).
        batch_size (int, optional): The number of images to process at once to avoid OOM.
                                    If None, processes the entire batch simultaneously.

    Returns:
        torch.Tensor: The shifted (and optionally CTF-corrected) images.
                      Shape remains (Nb, ..., Ny, Nx).
    """
    assert imgs.shape[-2:] == grid.shape[-2:], \
        f"Incompatible dimensions: imgs {imgs.shape[-2:]} vs grid {grid.shape[-2:]}"
    assert imgs.shape[0] == shifts.shape[0], \
        f"Batch size mismatch: imgs has {imgs.shape[0]}, shifts has {shifts.shape[0]}"

    Nb = imgs.shape[0]
    ndim = imgs.ndim

    batch_size = batch_size or Nb

    shift_expand_shape = [-1] + [1] * (ndim - 1)
    grid_expand_shape = [1] * (ndim - 2) + list(grid.shape[-2:])

    ky = grid[0].view(*grid_expand_shape)
    kx = grid[1].view(*grid_expand_shape)

    shifted_imgs_list = []

    for i in range(0, Nb, batch_size):
        end = min(i + batch_size, Nb)

        imgs_b = imgs[i:end]
        shifts_b = shifts[i:end]

        shift_y = shifts_b[:, 0].view(*shift_expand_shape)
        shift_x = shifts_b[:, 1].view(*shift_expand_shape)

        phase = -2 * torch.pi * (shift_x * kx + shift_y * ky)
        w = torch_phasor(phase)

        img_fft = torch.fft.fft2(imgs_b)
        shifted_img_b = torch.fft.ifft2(img_fft * w)

        shifted_imgs_list.append(shifted_img_b)

    return torch.cat(shifted_imgs_list, dim=0)


def make_soft_aperture_torch(alpha: torch.Tensor, max_alpha_mrad: float, rolloff_mrad: float) -> torch.Tensor:
    """
    Creates a differentiable circular aperture mask with a cosine edge taper.

    Args:
        alpha: Tensor of scattering angles in radians.
        max_alpha_mrad: The semiangle cutoff in milliradians.
        rolloff_mrad: The width of the soft edge transition in milliradians.
    """
    cutoff = max_alpha_mrad / 1000.0

    if rolloff_mrad <= 0.0:
        return (alpha <= cutoff).to(torch.float32)

    rolloff = rolloff_mrad / 1000.0

    transition = 0.5 * (1.0 + torch.cos(torch.pi * (alpha - cutoff + rolloff) / rolloff))
    mask = torch.where(alpha > cutoff, torch.zeros_like(alpha), transition)
    mask = torch.where(alpha < (cutoff - rolloff), torch.ones_like(alpha), mask)

    return mask


def make_probe_from_chi(chi, mask):
    assert chi.shape[-2:] == mask.shape[-2:]

    psi = torch_phasor(-1 * chi)
    probe = mask * psi
    probe = fftshift2(torch.fft.ifft2(ifftshift2(probe)))
    probe = probe / torch.sqrt(torch.sum((torch.abs(probe)) ** 2))
    return probe


def generate_shift_basis(order_keys: list, kX: torch.Tensor, kY: torch.Tensor, wavelength: float):
    """Generates the unweighted (C=1) analytic shift basis vectors for dx and dy."""
    alphaX = kX * wavelength
    alphaY = kY * wavelength
    alpha_sq = alphaX**2 + alphaY**2

    alpha_sq_safe = alpha_sq + 1e-12

    max_m = max([m for (n, m) in order_keys]) if order_keys else 0

    X, Y = {}, {}
    X[0] = torch.ones_like(alphaX)
    Y[0] = torch.zeros_like(alphaX)

    for m in range(max_m):
        X[m+1] = X[m] * alphaX - Y[m] * alphaY
        Y[m+1] = X[m] * alphaY + Y[m] * alphaX

    basis_dx_list = []
    basis_dy_list = []

    for (n, m) in order_keys:
        p = (n + 1 - m) / 2.0

        if n + 1 - m > 0:
            rad_deriv_x = (n + 1 - m) * alphaX * (alpha_sq_safe ** (p - 1.0))
            rad_deriv_y = (n + 1 - m) * alphaY * (alpha_sq_safe ** (p - 1.0))
        else:
            rad_deriv_x = torch.zeros_like(alphaX)
            rad_deriv_y = torch.zeros_like(alphaX)

        rad_base = alpha_sq ** p

        if m == 0:
            dx = (rad_deriv_x * X[m]) / (n + 1)
            dy = (rad_deriv_y * X[m]) / (n + 1)
            basis_dx_list.append(dx)
            basis_dy_list.append(dy)

        else:
            part1_x_a = rad_deriv_x * X[m]
            part1_y_a = rad_deriv_y * X[m]
            part2_x_a = rad_base * (m * X[m-1])
            part2_y_a = rad_base * (m * -Y[m-1])

            basis_dx_list.append((part1_x_a + part2_x_a) / (n + 1))
            basis_dy_list.append((part1_y_a + part2_y_a) / (n + 1))

            part1_x_b = rad_deriv_x * Y[m]
            part1_y_b = rad_deriv_y * Y[m]
            part2_x_b = rad_base * (m * Y[m-1])
            part2_y_b = rad_base * (m * X[m-1])

            basis_dx_list.append((part1_x_b + part2_x_b) / (n + 1))
            basis_dy_list.append((part1_y_b + part2_y_b) / (n + 1))

    return torch.stack(basis_dx_list, dim=0), torch.stack(basis_dy_list, dim=0)


def generate_aberration_basis(max_order: int, order_keys: list, kX: torch.Tensor, kY: torch.Tensor, wavelength: float):
    """Generates the unweighted (C=1) Cartesian polynomial basis tensors."""
    alphaX = kX * wavelength
    alphaY = kY * wavelength
    alpha_sq = alphaX**2 + alphaY**2

    X, Y = {}, {}
    X[0] = torch.ones_like(alpha_sq)
    Y[0] = torch.zeros_like(alpha_sq)

    for m in range(max_order + 1):
        X[m+1] = X[m] * alphaX - Y[m] * alphaY
        Y[m+1] = X[m] * alphaY + Y[m] * alphaX

    multiplier = (2 * torch.pi / wavelength)
    basis_list = []

    for (n, m) in order_keys:
        power_rad = (n + 1 - m) / 2.0
        term_radial = alpha_sq ** power_rad

        if m == 0:
            basis_list.append(term_radial * X[m] / (n + 1) * multiplier)
        else:
            basis_list.append(term_radial * X[m] / (n + 1) * multiplier)  # a
            basis_list.append(term_radial * Y[m] / (n + 1) * multiplier)  # b

    return torch.stack(basis_list, dim=0)


# ----------------------------------------------------------------------
# Upscale support estimates
# ----------------------------------------------------------------------


def probe_patch_radius(defocus: float, max_alpha_mrad: float) -> float:
    """Radius (Angstroms) of the object region imaged by one diffraction pattern.

    In linearized BF each pattern is the FT of
    ``O(x) |P(x - r_i)|² e^{iφ_P(x - r_i)}``, i.e. the object is weighted by
    the probe intensity — a shadow image of the aperture patch — centered on
    the scan point ``r_i``. Defocused by Δz, stationary phase shows the probe
    is the geometric shadow of the aperture, a disk of radius ``α |Δz|`` with
    Fresnel fringes of width ``sqrt(λ |Δz|)`` at its edge (neglected here).

    The in-focus Airy contribution (``0.61 λ / α``) is deliberately omitted:
    it is the diffraction-limited patch and does not grow with defocus, so
    including it would only set a floor on the estimate.

    Args:
        defocus: Defocus distance in Angstroms (signed; |·| is taken).
        max_alpha_mrad: Convergence semi-angle in mrad.

    Returns:
        Patch radius in Angstroms (zero at exactly Δz = 0).
    """
    return (max_alpha_mrad / 1000.0) * abs(defocus)


def estimate_max_upscale(
    defocus: float,
    scan_step_size: float,
    max_alpha_mrad: float,
) -> float:
    """Estimate the maximum upsampling supported by the scan data.

    The enhanced resolution of tcBF/acBF relies on every diffraction pattern
    imaging an object patch of radius :func:`probe_patch_radius` centered on
    its scan point. On a square scan lattice with step ``δ`` the farthest
    object point from any scan point is a cell corner at ``δ/√2``, so the
    patches overlap everywhere in the image iff ``a_probe ≥ δ/√2``, giving
    the maximum supported upsampling

        N_max = √2 · a_probe / δ

    (the 1-D analogue is ``a_probe / δ``).

    Args:
        defocus: Measured defocus distance in Angstroms
            (e.g. ``AberrationState.get_physical('C_1_0')``).
        scan_step_size: Raw scan step in Angstroms per pixel.
        max_alpha_mrad: Convergence semi-angle in mrad.

    Returns:
        The 2-D coverage ratio as a float (e.g. ``8.8``); floor it for the
        largest integer upscale that is fully supported.
    """
    a_probe = probe_patch_radius(defocus, max_alpha_mrad)
    return math.sqrt(2.0) * a_probe / scan_step_size
