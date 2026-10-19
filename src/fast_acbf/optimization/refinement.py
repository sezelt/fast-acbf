"""Refinement routines — line search and AD-based optimization.

All functions accept a solver-like object via duck typing (no BFSolver import).
Required solver interface:
    .ab_state          — AberrationState with set_physical / get_physical
    .reconstruct(mode, requires_grad=False, **kwargs) -> Tensor
    .set_rotation_deg(deg, clear_basis=False) — handles cache invalidation internally
    .set_flips(flipud, fliplr, transpose)     — sets flip flags and clears basis cache
    .clear_basis_cache()         — clears orientation-dependent caches, preserves FFT cache
    .clear_cache()               — full reset (both basis and FFT caches)
    .coord_transform             — dict with flipud/fliplr/transpose/rotation_deg (read-only view)
    .reconstructed_image         — writable attribute for caching last result
    .last_c10_stack_axis         — writable attribute
    .tolerance_factors           — dict {n: Tn_in_ang} for defocus range auto-scaling

State safety: always mutate solver state through public setters (set_rotation_deg,
set_flips), never direct dict-item writes, so cache invalidation stays consistent.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from scipy.optimize import minimize_scalar

from fast_acbf.optimization.metrics import QualityMetrics


def _validate_scan_roi(scan_roi, dataset_shape):
    """Normalize a scan ROI tuple against the leading scan dimensions."""
    if scan_roi is None:
        return None
    if len(scan_roi) != 4:
        raise ValueError("scan_roi must be a 4-tuple (y0, y1, x0, x1).")

    y0, y1, x0, x1 = (int(v) for v in scan_roi)
    ry, rx = dataset_shape[:2]
    if not (0 <= y0 < y1 <= ry and 0 <= x0 < x1 <= rx):
        raise ValueError(
            f"scan_roi must satisfy 0 <= y0 < y1 <= {ry} and "
            f"0 <= x0 < x1 <= {rx}, got {(y0, y1, x0, x1)}."
        )
    return y0, y1, x0, x1


def _copy_aberrations(src_solver, dst_solver):
    """Copy physical detector-frame aberration values between compatible solvers."""
    with torch.no_grad():
        for key in dst_solver.ab_state.coeffs:
            dst_solver.ab_state.set_physical(key, src_solver.ab_state.get_physical(key))


def _build_roi_solver(solver, scan_roi):
    """Create a temporary solver on a scan-space crop with matching physics state."""
    from fast_acbf.solver import BFSolver

    y0, y1, x0, x1 = _validate_scan_roi(scan_roi, solver._dataset.scan_shape)
    roi_dataset = solver._dataset.crop_scan_roi(y0, y1, x0, x1)
    return BFSolver(
        dataset=roi_dataset,
        max_alpha=solver.max_alpha,
        scan_step_size=solver.raw_scan_step_size,
        dk=solver.dk,
        wavelength=solver.wavelength,
        max_order=solver.max_order,
        aberrations=solver.ab_state.get_cartesian_dict(),
        device=solver.device,
        coord_transform=solver.coord_transform,  # returns a dict
        eps=solver.eps,
        pipeline=solver.pipeline,
        imagefft_storage=solver.imagefft_storage,
        imagefft_fill=solver.imagefft_fill,
        extractor_strategy=solver.extractor_strategy,
        basis_mode=solver.basis_mode,
        pad_width=solver.pad_width,
        fov=solver.fov,
        upscale=solver.upscale,
        upscale_method=solver.upscale_method,
        upscale_sigma=solver.upscale_sigma,
        lanczos_size=solver.lanczos_size,
    )


# ---------------------------------------------------------------------------
# Internal sweep helper (mirrors BFSolver._sweep_c10_stack logic)
# ---------------------------------------------------------------------------

def _sweep_c10(solver, c10_axis: torch.Tensor, mode: str, **kwargs):
    """Evaluate reconstruction over a C10 axis; restore original C10 on exit."""
    original_c10 = solver.ab_state.get_physical('C_1_0')
    stack_images = []
    try:
        with torch.no_grad():
            for c10 in c10_axis:
                solver.ab_state.set_physical('C_1_0', c10)
                img = solver.reconstruct(mode=mode, **kwargs)
                stack_images.append(img)
    finally:
        solver.ab_state.set_physical('C_1_0', original_c10)

    c10_axis = c10_axis.detach().clone()
    stack = torch.stack(stack_images, dim=0)
    solver.last_c10_stack_axis = c10_axis
    return c10_axis, stack


def _refine_defocus_brent(
    solver,
    *,
    search_range: tuple,
    metric: str,
    metric_kwargs: dict,
    blur: bool,
    blur_kernel_size: int,
    blur_sigma: float,
    plot_search: bool,
    mode: str,
    xatol: float,
    **kwargs,
) -> None:
    """Bounded Brent search for C10, adaptively sampling instead of a fixed grid.

    Only reliable when search_range is already narrow/unimodal (see
    refine_defocus's docstring). Restores C10 to its pre-call value if the
    search itself raises, then always commits the best point found.
    """
    print(f"Starting defocus line search (brent): bounded to "
          f"[{search_range[0]:.1f}, {search_range[1]:.1f}] Ang")

    metric_eval_kwargs = {
        "metric": metric,
        "blur": blur,
        "blur_kernel_size": blur_kernel_size,
        "blur_sigma": blur_sigma,
    }
    metric_eval_kwargs.update(metric_kwargs)

    original_c10 = solver.ab_state.get_physical('C_1_0')
    evaluated_c10: list[float] = []
    evaluated_scores: list[float] = []
    best_score = -np.inf
    best_image = None

    def _score_at(c10: float) -> float:
        nonlocal best_score, best_image
        with torch.no_grad():
            solver.ab_state.set_physical('C_1_0', float(c10))
            img = solver.reconstruct(mode=mode, **kwargs)
            score = QualityMetrics.evaluate(img, **metric_eval_kwargs).item()
        evaluated_c10.append(float(c10))
        evaluated_scores.append(score)
        if score > best_score:
            best_score = score
            best_image = img.detach().clone()
        return score

    try:
        result = minimize_scalar(
            lambda c10: -_score_at(c10),
            bounds=search_range,
            method='bounded',
            options={'xatol': xatol},
        )
        optimal_c10 = float(result.x)
        _score_at(optimal_c10)  # ensure solver state / cached image land exactly on it
    finally:
        solver.ab_state.set_physical('C_1_0', original_c10)

    print(f"Optimal C10 found at {optimal_c10:.2f} Ang (brent, "
          f"{len(evaluated_c10)} evaluations)")

    with torch.no_grad():
        solver.ab_state.set_physical('C_1_0', optimal_c10)
    solver.reconstructed_image = best_image

    if plot_search:
        from fast_acbf.vis.plotting import plot_defocus_line_search
        order = np.argsort(evaluated_c10)
        plot_defocus_line_search(
            c10_axis_np=np.asarray(evaluated_c10)[order],
            quality_scores=np.asarray(evaluated_scores)[order],
            optimal_c10=optimal_c10,
            search_range=search_range,
            metric=metric,
            method='brent',
            mode=mode,
            fit_coeffs=None,
            fit_type=None,
        )


# ---------------------------------------------------------------------------
# Public refinement functions
# ---------------------------------------------------------------------------

def refine_defocus(
    solver,
    *,
    search_range: tuple | None = None,
    num_points: int = 5,
    metric: str = 'sobel',
    metric_kwargs: dict | None = None,
    method: str = 'max',
    blur: bool = True,
    blur_kernel_size: int = 5,
    blur_sigma: float = 1,
    plot_search: bool = True,
    mode: str = 'tcBF',
    search_halfwidth: float | None = None,
    defocus_range_tolerance_factor: float = 24.0,
    xatol: float = 0.5,
    **kwargs,
) -> None:
    """
    Line search for defocus (C10), brute-force (grid) or adaptive (Brent).

    'max'/'fit_parabola' sweep num_points evenly-spaced C10 values, score each
    slice with QualityMetrics, then pick the max (or fit a parabola to the
    sampled scores). 'brent' instead runs SciPy's bounded Brent search, which
    adaptively concentrates evaluations near the optimum instead of spending a
    fixed budget uniformly — typically far fewer reconstructions for equal or
    better precision, but only reliable when search_range is already narrow
    enough that the score-vs-defocus curve is close to unimodal (e.g. once
    focus_sign has constrained the range to one side of through-focus). A wide
    or sign-unconstrained range is more likely to have secondary local peaks
    from residual aberrations, where 'max'/'fit_parabola' are safer since they
    can't get stuck in the wrong one.

    Updates solver.ab_state in-place and caches the best reconstruction in
    solver.reconstructed_image.

    Args:
        solver:           Solver-like object (see module docstring).
        search_range:     Literal (min_c10, max_c10) bounds in Angstroms. If None,
                          uses search_halfwidth or defocus_range_tolerance_factor.
        num_points:       Number of C10 values to sample. Only used for
                          method='max'/'fit_parabola'.
        metric:           Focus metric for QualityMetrics.evaluate.
        metric_kwargs:    Extra keyword arguments for QualityMetrics.evaluate.
        method:           'max' (default), 'fit_parabola', or 'brent'.
        blur:             Pre-blur images before scoring.
        blur_kernel_size: Kernel size for Gaussian blur.
        blur_sigma:       Sigma for Gaussian blur.
        plot_search:      Show matplotlib line-search summary.
        mode:             Reconstruction mode ('tcBF' or 'acBF').
        search_halfwidth: Current-centered half-width in Angstroms. Mutually
                          exclusive with search_range.
        defocus_range_tolerance_factor:
                          Multiplier on the 1st-order Kirkland tolerance T₁
                          for the auto defocus range when neither search_range
                          nor search_halfwidth is provided.
        xatol:            Absolute C10 convergence tolerance in Angstroms, only
                          used for method='brent'.
    """
    mode = mode.lower()
    metric_kwargs = dict(metric_kwargs or {})
    method = method.lower()
    if method not in ('max', 'fit_parabola', 'brent'):
        raise ValueError(f"Unsupported method: {method!r}. Choose 'max', 'fit_parabola', or 'brent'.")
    if search_range is not None and search_halfwidth is not None:
        raise ValueError("Provide either search_range or search_halfwidth, not both.")

    if search_range is None:
        c10 = solver.ab_state.get_physical('C_1_0')
        if search_halfwidth is None:
            T1 = solver.tolerance_factors[1]
            half = defocus_range_tolerance_factor * T1
            range_source = f"{defocus_range_tolerance_factor:.0f}×T₁"
        else:
            half = float(search_halfwidth)
            range_source = f"{half:.1f} Å half-width"
        search_range = (c10 - half, c10 + half)
        print(f"Auto defocus search_range: ({search_range[0]:.1f}, {search_range[1]:.1f}) Å "
              f"(C10={c10:.1f} ± {half:.1f} Å from {range_source})")

    min_def, max_def = min(search_range), max(search_range)
    search_range = (min_def, max_def)

    if method == 'brent':
        _refine_defocus_brent(
            solver,
            search_range=search_range,
            metric=metric,
            metric_kwargs=metric_kwargs,
            blur=blur,
            blur_kernel_size=blur_kernel_size,
            blur_sigma=blur_sigma,
            plot_search=plot_search,
            mode=mode,
            xatol=xatol,
            **kwargs,
        )
        return

    print(f"Starting defocus line search: {num_points} points between {search_range[0]} and {search_range[1]} Ang")

    device = next(iter(solver.ab_state.coeffs.values())).device
    c10_axis = torch.linspace(search_range[0], search_range[1], num_points,
                               dtype=torch.float32, device=device)

    c10_axis, scan_stack = _sweep_c10(solver, c10_axis, mode=mode, **kwargs)
    metric_eval_kwargs = {
        "metric": metric,
        "blur": blur,
        "blur_kernel_size": blur_kernel_size,
        "blur_sigma": blur_sigma,
    }
    metric_eval_kwargs.update(metric_kwargs)
    quality_scores = QualityMetrics.evaluate(scan_stack, **metric_eval_kwargs).detach().cpu().numpy()
    c10_axis_np = c10_axis.detach().cpu().numpy()

    if method == 'fit_parabola':
        fit_coeffs = np.polyfit(c10_axis_np, quality_scores, 2)
        a, b, _ = fit_coeffs
        if a < 0:
            optimal_c10 = -b / (2 * a)
            optimal_c10 = np.clip(optimal_c10, search_range[0], search_range[1])
            fit_type = "Parabolic vertex"
            optimal_index = int(np.argmin(np.abs(c10_axis_np - optimal_c10)))
        else:
            optimal_index = int(np.argmax(quality_scores))
            optimal_c10 = c10_axis_np[optimal_index]
            fit_type = "Discrete max (fit inverted)"
    else:  # method == 'max', the only other value reachable here
        optimal_index = int(np.argmax(quality_scores))
        optimal_c10 = c10_axis_np[optimal_index]
        fit_coeffs = None

    print(f"Optimal C10 found at {optimal_c10:.2f} Ang ({method})")

    with torch.no_grad():
        solver.ab_state.set_physical('C_1_0', float(optimal_c10))
    solver.reconstructed_image = scan_stack[optimal_index]

    if plot_search:
        from fast_acbf.vis.plotting import plot_defocus_line_search
        plot_defocus_line_search(
            c10_axis_np=c10_axis_np,
            quality_scores=quality_scores,
            optimal_c10=optimal_c10,
            search_range=search_range,
            metric=metric,
            method=method,
            mode=mode,
            fit_coeffs=fit_coeffs if method == 'fit_parabola' else None,
            fit_type=fit_type if method == 'fit_parabola' else None,
        )


def refine_aberrations(
    solver,
    lr: float = 1,
    lr_scales=None,
    iters: int = 50,
    metric: str = 'sobel',
    metric_kwargs: dict | None = None,
    plot_recon_every_n_iter=None,
    save_dir=None,
    mode: str = 'tcBF',
    scan_roi=None,
    **kwargs,
) -> None:
    """
    Gradient-based refinement of all aberration coefficients via Adam.

    Minimizes -QualityMetrics(reconstruction) with per-order learning rate scaling.
    Updates solver.ab_state in-place.

    Args:
        solver:                  Solver-like object.
        lr:                      Base learning rate.
        lr_scales:               Per-order LR multipliers, length max_order.
        iters:                   Number of gradient steps.
        metric:                  Focus metric to maximize.
        metric_kwargs:           Extra keyword arguments for QualityMetrics.evaluate.
        plot_recon_every_n_iter: Show reconstruction every N iterations if set.
        save_dir:                Directory to save per-iteration figures if set.
        mode:                    Reconstruction mode.
        scan_roi:                Optional (y0, y1, x0, x1) scan crop for AD refinement.
    """
    mode = mode.lower()
    metric_kwargs = dict(metric_kwargs or {})

    if scan_roi is not None:
        roi_solver = _build_roi_solver(solver, scan_roi)
        refine_aberrations(
            roi_solver,
            lr=lr,
            lr_scales=lr_scales,
            iters=iters,
            metric=metric,
            metric_kwargs=metric_kwargs,
            plot_recon_every_n_iter=plot_recon_every_n_iter,
            save_dir=save_dir,
            mode=mode,
            scan_roi=None,
            **kwargs,
        )
        _copy_aberrations(roi_solver, solver)
        roi_solver.clear_cache()
        del roi_solver
        if torch.device(solver.device).type == 'cuda':
            torch.cuda.empty_cache()

        solver.clear_basis_cache()
        solver.reconstructed_image = solver.reconstruct(
            mode=mode, requires_grad=False, **kwargs,
        ).detach()
        return

    max_order = solver.ab_state.max_order

    if lr_scales is None:
        lr_scales = [1.0] * max_order
    lr_scales = list(lr_scales)
    if len(lr_scales) != max_order:
        raise ValueError(f"lr_scales must have length max_order={max_order}, got {len(lr_scales)}")

    param_groups = []
    for idx, n in enumerate(range(1, max_order + 1)):
        order_params = [
            solver.ab_state.coeffs[key]
            for key in solver.ab_state.coeffs
            if int(key.split('_')[1]) == n
        ]
        param_groups.append({'params': order_params, 'lr': lr * lr_scales[idx]})

    optimizer = torch.optim.Adam(param_groups, lr=lr)

    for i in range(iters):
        optimizer.zero_grad()
        summed_img = solver.reconstruct(mode=mode, requires_grad=True, **kwargs)
        solver.reconstructed_image = summed_img.detach()
        loss = -1 * QualityMetrics.evaluate(summed_img, metric=metric, **metric_kwargs)
        loss.backward()
        optimizer.step()

        if plot_recon_every_n_iter is not None and i % plot_recon_every_n_iter == 0:
            with torch.no_grad():
                title_str = f'Iter {i}, Loss (-{metric}) : {loss.item():.4g}'
                ab_dict = solver.get_aberrations_dict(layout='flat')
                desc_str = ", ".join(f"{ab}: {val:.2f}" for ab, val in ab_dict.items())

                if save_dir is not None:
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = f'{save_dir}/figure_recon_iter_{str(i).zfill(3)}.png'
                else:
                    save_path = None

                solver.plot_reconstruction(
                    title_str=title_str, desc_str=desc_str,
                    save_path=save_path, mode=mode, **kwargs,
                )


def refine_scan_rotation(
    solver,
    *,
    search_range: tuple | None = None,
    num_points: int = 9,
    metric: str = 'sobel',
    metric_kwargs: dict | None = None,
    plot_search: bool = True,
    mode: str = 'tcBF',
    search_halfwidth: float | None = None,
    method: str = 'grid',
    xatol: float = 0.05,
    **kwargs,
) -> None:
    """
    Search for the optimal scan rotation angle, brute-force (grid) or adaptive (Brent).

    Scores reconstructions over search_range and sets the optimal rotation via
    solver.set_rotation_deg() (which triggers cache invalidation automatically).

    Args:
        solver:       Solver-like object.
        search_range: Literal (min_deg, max_deg) rotation bounds to search.
                      Defaults to (-45°, +45°).
        num_points:   Number of angles to sample. Only used for method='grid'.
        metric:       Focus metric for QualityMetrics.evaluate.
        metric_kwargs:
                      Extra keyword arguments for QualityMetrics.evaluate.
        plot_search:  Show matplotlib line-search summary.
        mode:         Reconstruction mode.
        search_halfwidth:
                      Current-centered rotation half-width in degrees. Mutually
                      exclusive with search_range.
        method:       'grid' (default): brute-force evenly-spaced sweep over
                      num_points angles. Robust to a possibly-multimodal range
                      (e.g. the wide default ±45° window) since it samples the
                      whole interval rather than assuming a single peak.
                      'brent': bounded Brent search (SciPy's method='bounded').
                      Adaptively concentrates evaluations near the optimum —
                      typically far fewer reconstructions than 'grid' for equal
                      or better precision — but only reliable on an already
                      narrow, near-unimodal window, e.g. a tight post-grid-search
                      refinement step.
        xatol:        Absolute angular convergence tolerance in degrees, only
                      used for method='brent'.
    """
    mode = mode.lower()
    metric_kwargs = dict(metric_kwargs or {})
    method = method.lower()
    if method not in ('grid', 'brent'):
        raise ValueError(f"Unsupported method: {method!r}. Choose 'grid' or 'brent'.")
    if search_range is not None and search_halfwidth is not None:
        raise ValueError("Provide either search_range or search_halfwidth, not both.")

    if search_range is None:
        if search_halfwidth is None:
            search_range = (-45.0, 45.0)
        else:
            half = float(search_halfwidth)
            search_range = (solver.rotation_deg - half, solver.rotation_deg + half)

    min_rot, max_rot = min(search_range), max(search_range)

    original_rotation = solver.rotation_deg
    evaluated_angles: list[float] = []
    evaluated_scores: list[float] = []
    best_score = -np.inf
    best_image = None

    def _score_at(angle_deg: float) -> float:
        nonlocal best_score, best_image
        # clear_basis=True: each angle produces a distinct cache key; without
        # clearing, entries accumulate in _basis_cache simultaneously. Angles
        # are never revisited within one search, so only one entry is needed
        # at a time.
        with torch.no_grad():
            solver.set_rotation_deg(float(angle_deg), clear_basis=True)
            img = solver.reconstruct(mode=mode, **kwargs)
            score = QualityMetrics.evaluate(img, metric=metric, **metric_kwargs).item()
        evaluated_angles.append(float(angle_deg))
        evaluated_scores.append(score)
        if score > best_score:
            best_score = score
            best_image = img.detach().clone()
        return score

    try:
        if method == 'grid':
            print(f"Starting rotation line search (grid): {num_points} points between "
                  f"{min_rot:.1f} and {max_rot:.1f} deg")
            for angle in np.linspace(min_rot, max_rot, num_points):
                _score_at(angle)
            optimal_rotation = evaluated_angles[int(np.argmax(evaluated_scores))]
        else:
            print(f"Starting rotation line search (brent): bounded to "
                  f"[{min_rot:.1f}, {max_rot:.1f}] deg")
            result = minimize_scalar(
                lambda angle: -_score_at(angle),
                bounds=(min_rot, max_rot),
                method='bounded',
                options={'xatol': xatol},
            )
            optimal_rotation = float(result.x)
            _score_at(optimal_rotation)  # ensure solver state / cached image land exactly on it
    finally:
        solver.set_rotation_deg(original_rotation)

    print(f"Optimal rotation found at {optimal_rotation:.2f} deg "
          f"({len(evaluated_angles)} evaluations, method={method})")
    solver.set_rotation_deg(optimal_rotation, clear_basis=True)
    solver.reconstructed_image = best_image

    if plot_search:
        from fast_acbf.vis.plotting import plot_rotation_line_search
        order = np.argsort(evaluated_angles)
        plot_rotation_line_search(
            angles_deg=np.asarray(evaluated_angles)[order],
            quality_scores=np.asarray(evaluated_scores)[order],
            optimal_rotation=optimal_rotation,
            metric=metric,
            mode=mode,
        )


def refine_flips(
    solver,
    metric: str = 'sobel',
    metric_kwargs: dict | None = None,
    plot_search: bool = True,
    mode: str = 'tcBF',
    **kwargs,
) -> dict:
    """
    Exhaustive search over all 8 flip/transpose combinations.

    Tests all combinations of (flipud, fliplr, transpose) using the current
    rotation_deg, then sets the best combination via solver.coord_transform
    and re-triggers cache invalidation.

    Args:
        solver: Solver-like object.
        metric: Focus metric for QualityMetrics.evaluate.
        metric_kwargs: Extra keyword arguments for QualityMetrics.evaluate.
        plot_search: Show 2×4 reconstruction panel summary.
        mode:   Reconstruction mode.

    Returns:
        Dict mapping each (flipud, fliplr, transpose) combo to its score,
        with 'best' key indicating the winning combination.
    """
    mode = mode.lower()
    metric_kwargs = dict(metric_kwargs or {})
    combos = [
        (flipud, fliplr, transpose)
        for flipud    in (False, True)
        for fliplr    in (False, True)
        for transpose in (False, True)
    ]

    original = {
        'flipud':    solver.coord_transform.get('flipud',    False),
        'fliplr':    solver.coord_transform.get('fliplr',    False),
        'transpose': solver.coord_transform.get('transpose', False),
    }

    results = {}
    plot_images = []
    best_score = -np.inf
    best_image = None
    try:
        with torch.no_grad():
            for (flipud, fliplr, transpose) in combos:
                solver.set_flips(flipud, fliplr, transpose)
                img = solver.reconstruct(mode=mode, **kwargs)
                score = QualityMetrics.evaluate(img, metric=metric, **metric_kwargs).item()
                results[(flipud, fliplr, transpose)] = score
                if score > best_score:
                    best_score = score
                    best_image = img.detach().clone()
                if plot_search:
                    plot_images.append(img.detach().cpu().numpy())
    finally:
        solver.set_flips(**original)

    best_combo = max(results, key=results.__getitem__)
    results['best'] = best_combo

    print(f"Best flip combination: flipud={best_combo[0]}, fliplr={best_combo[1]}, "
          f"transpose={best_combo[2]}  (score={results[best_combo]:.4g})")

    solver.set_flips(*best_combo)
    solver.reconstructed_image = best_image

    if plot_search:
        from fast_acbf.vis.plotting import plot_flips_grid_search
        plot_flips_grid_search(
            images=plot_images,
            scores={combo: results[combo] for combo in combos},
            best_combo=best_combo,
            metric=metric,
            mode=mode,
        )

    return results


# ---------------------------------------------------------------------------
# O(2) orientation helpers — used by refine_all_params
# ---------------------------------------------------------------------------

# D4 lookup table: maps (is_flipped, q) -> (flipud, fliplr, transpose).
#
# Parameterization: the full scan→detector transform is decomposed as
#   "apply (flipud, fliplr, transpose) at rotation_deg = residual_angle"
# such that it equals the O(2) state
#   "(transpose=is_flipped only, flipud=fliplr=False) at rotation_deg = opt_angle_deg"
#
# is_flipped=False entries are the four pure CCW rotations (0°, 90°, 180°, 270°).
# is_flipped=True entries are the four reflections (transpose composed with CCW rotations).
# Derived from _get_transformed_bf_coordinates and verified numerically.
_D4_TABLE = {
    (False, 0): (False, False, False),  # identity
    (False, 1): (True,  False, True),   # CCW 90°
    (False, 2): (True,  True,  False),  # CCW 180°
    (False, 3): (False, True,  True),   # CCW 270°
    (True,  0): (False, False, True),   # transpose
    (True,  1): (False, True,  False),  # fliplr
    (True,  2): (True,  True,  True),   # flipud + fliplr + transpose
    (True,  3): (True,  False, False),  # flipud
}


def map_to_ptyrad_state(if_transposed: bool, opt_angle_deg: float) -> dict:
    """
    Maps an optimized O(2) state (chirality + continuous rotation) to
    PtyRAD-compatible D4 flags + residual rotation.

    O(2) parameterization uses only 2 chiralities rather than 8 flip combos:
      - if_transposed=False: (flipud=F, fliplr=F, transpose=F) + rotation opt_angle_deg
      - if_transposed=True:  (flipud=F, fliplr=F, transpose=T) + rotation opt_angle_deg

    Returns dict with keys: flipud, fliplr, transpose (bool), rotation_deg (float).
    Residual rotation is opt_angle_deg mod 90°, in roughly [-45°, +45°].
    """
    q = int(round(opt_angle_deg / 90.0)) % 4
    residual_angle = opt_angle_deg - q * 90.0
    flipud, fliplr, transpose = _D4_TABLE[(bool(if_transposed), q)]
    return {
        'flipud': flipud,
        'fliplr': fliplr,
        'transpose': transpose,
        'rotation_deg': residual_angle,
    }


def _orientation_grid_search(
    solver,
    defocus_range: tuple,
    rotation_num_points: int,
    defocus_num_points: int,
    metric: str,
    metric_kwargs: dict | None,
    mode: str,
    **kwargs,
) -> None:
    """
    Joint grid search over 2 chiralities × rotation_num_points angles × defocus_num_points
    defocus values. Applies the best (chirality, angle, C_1_0) state to solver, decomposed
    to PtyRAD D4 flags via map_to_ptyrad_state.

    Cache discipline: chirality changes call clear_basis_cache() (2 total); each rotation angle
    calls set_rotation_deg(..., clear_basis=True) so only one basis cache entry exists at
    a time. Defocus sweeps reuse the same cache entry since C_1_0 is not part of the key.
    """
    metric_kwargs = dict(metric_kwargs or {})
    angles = np.linspace(0.0, 360.0, rotation_num_points, endpoint=False)
    c10_values = np.linspace(defocus_range[0], defocus_range[1], defocus_num_points)
    total = 2 * rotation_num_points * defocus_num_points
    print(f"Starting orientation+defocus grid search: {total} evaluations "
          f"(2 chiralities × {rotation_num_points} angles × {defocus_num_points} defocus points)")

    original_c10 = solver.ab_state.get_physical('C_1_0')
    # Non-rotationally-symmetric terms (m≠0) interact with rotation: C12 can compensate
    # for orientation errors, creating deep local minima. Zero them during the grid search.
    non_sym_keys = [k for k in solver.ab_state.coeffs if k.endswith('_a') or k.endswith('_b')]
    original_non_sym = {k: solver.ab_state.get_physical(k) for k in non_sym_keys}

    best_score = -np.inf
    best_if_transposed = False
    best_angle = 0.0
    best_c10 = float(c10_values[0])

    try:
        with torch.no_grad():
            for k in non_sym_keys:
                solver.ab_state.set_physical(k, 0.0)

            for if_transposed in (False, True):
                # One cache clear per chirality — the expensive operation
                solver.set_flips(False, False, if_transposed)
                solver.set_rotation_deg(0.0)

                for angle in angles:
                    # clear_basis=True: sequential search never revisits old angles,
                    # so accumulating per-angle basis cache entries only wastes memory.
                    solver.set_rotation_deg(float(angle), clear_basis=True)
                    for c10 in c10_values:
                        solver.ab_state.set_physical('C_1_0', float(c10))
                        img = solver.reconstruct(mode=mode, **kwargs)
                        score = QualityMetrics.evaluate(img, metric=metric, **metric_kwargs).item()
                        if score > best_score:
                            best_score = score
                            best_if_transposed = if_transposed
                            best_angle = float(angle)
                            best_c10 = float(c10)
    finally:
        solver.ab_state.set_physical('C_1_0', original_c10)
        for k, v in original_non_sym.items():
            solver.ab_state.set_physical(k, v)
        solver.clear_basis_cache()

    ptyrad_state = map_to_ptyrad_state(best_if_transposed, best_angle)
    solver.set_flips(ptyrad_state['flipud'], ptyrad_state['fliplr'], ptyrad_state['transpose'])
    solver.set_rotation_deg(ptyrad_state['rotation_deg'])
    solver.ab_state.set_physical('C_1_0', best_c10)

    print(f"Best: if_transposed={best_if_transposed}, angle={best_angle:.1f}°, C10={best_c10:.2f}Å "
          f"(score={best_score:.4g})")
    print(f"  → flipud={ptyrad_state['flipud']}, fliplr={ptyrad_state['fliplr']}, "
          f"transpose={ptyrad_state['transpose']}, rotation_deg={ptyrad_state['rotation_deg']:.1f}°")


# ---------------------------------------------------------------------------
# High-level orchestrator
# ---------------------------------------------------------------------------

def refine_all_params(
    solver,
    targets=('orientation_defocus', 'coarse_aberrations', 'fine_rotation', 'fine_aberrations'),
    metric: str = 'sobel',
    metric_kwargs: dict | None = None,
    mode: str = 'tcBF',
    defocus_range=None,
    defocus_range_tolerance_factor: float = 24.0,
    rotation_num_points: int = 36,
    defocus_num_points: int = 11,
    fine_rotation_halfwidth: float = 5.0,
    fine_rotation_num_points: int = 11,
    fine_rotation_xatol: float = 0.05,
    aberration_lr: float = 1.0,
    aberration_iters: int = 50,
    refinement_scan_roi=None,
    **kwargs,
) -> None:
    """
    Coarse-to-fine parameter orchestration.

    Executes a subset of the four pipeline steps, selected by `targets`:

      'orientation_defocus'         — Joint 2-chirality × rotation × defocus grid search (Step 1).
      'coarse_aberrations'  — Adam optimisation restricted to 1st + 2nd order (Step 2).
      'fine_rotation'       — Tight ±fine_rotation_halfwidth° bounded Brent search (Step 3).
      'fine_aberrations'    — Full-order Adam optimisation (Step 4).

    Step 3 uses refine_scan_rotation(method='brent') rather than a grid: Step 1
    has already bracketed the correct rotation basin, so the ±fine_rotation_halfwidth°
    window is expected to be near-unimodal, which is exactly where an adaptive
    bounded search converges in far fewer evaluations than a fixed-point sweep.
    Brent has no notion of "how many points" — it adaptively decides how many
    evaluations to spend based on fine_rotation_xatol (the convergence
    tolerance) and how smooth the score-vs-angle curve looks locally, so that
    parameter (not fine_rotation_num_points) is what now controls Step 3's
    precision/effort tradeoff.

    Args:
        solver:                         Solver-like object.
        targets:                        Ordered sequence of step names to run.
        metric:                         Focus metric shared across all steps.
        metric_kwargs:                  Extra keyword arguments for QualityMetrics.evaluate.
        mode:                           Reconstruction mode ('tcBF' or 'acBF').
        defocus_range:                  (min_c10, max_c10) in Å for Step 1. If None,
                                        auto-computed as current_C10 ± defocus_range_tolerance_factor × T₁.
        defocus_range_tolerance_factor: Multiplier on the 1st-order Kirkland tolerance T₁
                                        for the auto defocus range. Default 24 (= ±6π phase).
        rotation_num_points:            Rotation angles sampled in [0°, 360°) for Step 1.
        defocus_num_points:             Defocus samples in defocus_range for Step 1.
        fine_rotation_halfwidth:        Half-width in degrees for Step 3 search range.
        fine_rotation_num_points:       Unused — Step 3 is now an adaptive Brent search,
                                        not a fixed-point sweep. Kept for signature
                                        back-compat.
        fine_rotation_xatol:            Absolute angular convergence tolerance in degrees
                                        for Step 3's Brent search.
        aberration_lr:                  Base learning rate for Adam steps.
        aberration_iters:               Gradient steps per Adam call.
        refinement_scan_roi:            Optional scan ROI for AD aberration steps.
    """
    mode = mode.lower()
    targets = tuple(targets)
    metric_kwargs = dict(metric_kwargs or {})

    if 'orientation_defocus' in targets and defocus_range is None:
        c10 = solver.ab_state.get_physical('C_1_0')
        T1 = solver.tolerance_factors[1]
        half = defocus_range_tolerance_factor * T1
        defocus_range = (c10 - half, c10 + half)
        print(f"Auto defocus_range: ({defocus_range[0]:.1f}, {defocus_range[1]:.1f}) Å "
              f"(C10={c10:.1f} ± {half:.1f} Å = ±{defocus_range_tolerance_factor:.0f}×T₁)")

    if 'orientation_defocus' in targets:
        _orientation_grid_search(
            solver,
            defocus_range=defocus_range,
            rotation_num_points=rotation_num_points,
            defocus_num_points=defocus_num_points,
            metric=metric,
            metric_kwargs=metric_kwargs,
            mode=mode,
            **kwargs,
        )

    if 'coarse_aberrations' in targets:
        max_order = solver.ab_state.max_order
        coarse_lr_scales = [1.0] * min(2, max_order) + [0.0] * max(0, max_order - 2)
        refine_aberrations(
            solver,
            lr=aberration_lr,
            lr_scales=coarse_lr_scales,
            iters=aberration_iters,
            metric=metric,
            metric_kwargs=metric_kwargs,
            mode=mode,
            scan_roi=refinement_scan_roi,
            **kwargs,
        )

    if 'fine_rotation' in targets:
        refine_scan_rotation(
            solver,
            search_halfwidth=fine_rotation_halfwidth,
            method='brent',
            xatol=fine_rotation_xatol,
            metric=metric,
            metric_kwargs=metric_kwargs,
            plot_search=False,
            mode=mode,
            **kwargs,
        )

    if 'fine_aberrations' in targets:
        refine_aberrations(
            solver,
            lr=aberration_lr,
            iters=aberration_iters,
            metric=metric,
            metric_kwargs=metric_kwargs,
            mode=mode,
            scan_roi=refinement_scan_roi,
            **kwargs,
        )
    
    # Save the final image based on the retrieved params
    solver.reconstructed_image = solver.reconstruct(
        mode=mode, requires_grad=False, **kwargs,
    ).detach()
