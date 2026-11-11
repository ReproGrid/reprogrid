"""Simultaneous dq impedance estimation from coherent MLBS/IRS records."""

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import max_len_seq


def perturbations(protocol):
    journal = protocol["journal"]
    design = protocol["independent_benchmark"]
    samples_per_chip = round(1 / (journal["acquisition_step_s"] * journal["generation_hz"]))
    if not np.isclose(samples_per_chip * journal["acquisition_step_s"] * journal["generation_hz"], 1.):
        raise ValueError("A chip must contain an integer number of samples.")
    order = journal["mlbs_order"]
    mlbs = 2 * max_len_seq(order, state=np.ones(order, dtype=int), taps=design["mlbs_taps"])[0] - 1
    chips = np.column_stack((np.tile(mlbs, 2), np.concatenate((mlbs, -mlbs))))
    return np.repeat(chips, samples_per_chip, axis=0).astype(float)


def spectral_leakage(perturbation):
    power = abs(np.fft.rfft(perturbation, axis=0))**2
    leakage = power[1::2, 0].sum() / power[:, 0].sum()
    leakage_irs = power[::2, 1].sum() / power[:, 1].sum()
    return {"mlbs_on_odd_bins": float(leakage), "irs_on_even_bins": float(leakage_irs)}


def voltage_usable(voltage, floor):
    return abs(voltage) > floor * np.linalg.norm(voltage, axis=-1, keepdims=True)


def aligned_row_impedance(i, v, bins, floor, condition_limit):
    """Eqs. (10)–(11), with the IRS ratios interpolated on even MLBS bins."""
    neighbors = bins[:, None] + np.array([-1, 1])
    at_mlbs = voltage_usable(v[bins], floor)
    at_irs = voltage_usable(v[neighbors], floor).all(axis=1)
    z = np.full((len(bins), 2, 2), np.nan + 1j * np.nan)
    condition = np.full((len(bins), 2), np.inf)
    for row in range(2):
        candidates = np.flatnonzero(at_mlbs[:, row] & at_irs[:, row])
        mlbs_bins = bins[candidates]
        irs_bins = neighbors[candidates]
        mlbs_ratios = i[mlbs_bins] / v[mlbs_bins, row, None]
        irs_ratios = (i[irs_bins] / v[irs_bins, row, None]).mean(axis=1)
        ratios = np.stack((mlbs_ratios, irs_ratios), axis=1)
        condition[candidates, row] = np.linalg.cond(ratios)
        good = condition[candidates, row] < condition_limit
        # Each row of Z satisfies (I_d/V_row, I_q/V_row) dot Z_row = 1
        # for both excitations, i.e. the two determinants in Eq. (10).
        z[candidates[good], row] = np.linalg.solve(ratios[good], np.ones((int(good.sum()), 2, 1)))[..., 0]
    return z, condition


def exact_common_bin_impedance(i, v):
    """Eq. (9) with independent drive columns on identical frequencies."""
    return np.linalg.solve(i.swapaxes(-1, -2), v.swapaxes(-1, -2)).swapaxes(-1, -2)


def from_records(current, voltage, bins, protocol):
    model = protocol["independent_benchmark"]
    return aligned_row_impedance(np.fft.rfft(current, axis=0), np.fft.rfft(voltage, axis=0), bins,
                                 model["voltage_floor_relative_to_bin_vector"],
                                 model["ratio_condition_limit"])


def complex_average(estimates):
    # Geometric magnitude and circular phase average. Express phase relative
    # to the first frame before averaging to avoid the +/- pi discontinuity.
    valid = np.isfinite(estimates).all(axis=0) & (abs(estimates) > 0.).all(axis=0)
    out = np.full(estimates.shape[1:], np.nan + 1j * np.nan)
    anchor = estimates[0, valid]
    magnitude = np.exp(np.log(abs(estimates[:, valid])).mean(axis=0))
    phase = np.angle(anchor) + np.angle(estimates[:, valid] / anchor).mean(axis=0)
    out[valid] = magnitude * np.exp(1j * phase)
    return out


def median_magnitude_phase(z, width):
    """Median of magnitude and unwrapped phase within each valid bin run."""
    filtered = z.copy()
    for row, col in np.ndindex(2, 2):
        values = z[:, row, col]
        finite = np.isfinite(values)
        boundaries = np.flatnonzero(np.diff(np.r_[False, finite, False]))
        for start, stop in boundaries.reshape(-1, 2):
            segment = values[start:stop]
            phase = median_filter(np.unwrap(np.angle(segment)), size=width, mode="nearest")
            filtered[start:stop, row, col] = median_filter(abs(segment), size=width, mode="nearest") * np.exp(1j * phase)
    return filtered


def loop_eigenvalues(loop):
    """Eq. (2), with continuity-based ordering for plotting and crossings."""
    a, b, c, d = loop.reshape(-1, 4).T
    discriminant = np.sqrt((a - d)**2 + 4 * b * c)
    branches = np.column_stack(((a + d + discriminant) / 2, (a + d - discriminant) / 2))
    for k in range(1, len(branches)):
        old, current = branches[k - 1], branches[k]
        if abs(old - current[::-1]).sum() < abs(old - current).sum():
            branches[k] = current[::-1]
    return branches


def band_metrics(loop, frequency):
    usable = np.isfinite(loop).all(axis=(1, 2))
    if not usable.any():
        return {"minimum_distance_to_minus_one": None, "nearest_frequency_hz": None,
                "unit_magnitude_crossings": [], "valid_loop_bins": 0}
    distance = np.full((len(loop), 2), np.nan)
    crossings = []
    boundaries = np.flatnonzero(np.diff(np.r_[False, usable, False]))
    for start, stop in boundaries.reshape(-1, 2):
        branches = loop_eigenvalues(loop[start:stop])
        distance[start:stop] = abs(1 + branches)
        offset = abs(branches) - 1
        for k, column in np.argwhere(offset[:-1] * offset[1:] < 0):
            fraction = -offset[k, column] / (offset[k + 1, column] - offset[k, column])
            eigenvalue = (1 - fraction) * branches[k, column] + fraction * branches[k + 1, column]
            crossings.append({
                "frequency_hz": float((1 - fraction) * frequency[start + k] + fraction * frequency[start + k + 1]),
                "absolute_angular_distance_from_minus_one_deg": float(abs(np.angle(-eigenvalue, deg=True))),
            })
    row, _ = np.unravel_index(np.nanargmin(distance), distance.shape)
    return {"minimum_distance_to_minus_one": float(np.nanmin(distance)),
            "nearest_frequency_hz": float(frequency[row]), "unit_magnitude_crossings": crossings,
            "valid_loop_bins": int(usable.sum())}


def error_statistics(estimate, reference):
    usable = np.isfinite(estimate).all(axis=(1, 2)) & np.isfinite(reference).all(axis=(1, 2))
    error = np.full(len(estimate), np.nan)
    error[usable] = np.linalg.norm((estimate - reference)[usable], axis=(1, 2)) / np.linalg.norm(reference[usable], axis=(1, 2))
    valid = error[usable]
    if len(valid):
        med, p95, worst = np.quantile(valid, [.5, .95, 1.])
        metrics = {"median": float(med), "p95": float(p95), "max": float(worst)}
    else:
        metrics = {"median": None, "p95": None, "max": None}
    return error, {**metrics, "valid_bins": int(usable.sum()), "total_bins": len(usable)}
