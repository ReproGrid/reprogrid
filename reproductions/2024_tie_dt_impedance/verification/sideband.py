"""Independent scalar sideband solution of the balanced three-phase circuit."""

import numpy as np


def sideband_reference(circuit, frequency_hz):
    s = 2j * np.pi * frequency_hz
    plant_z, grid_y = [], []
    for sign in (1, -1):
        sideband = s + 1j * sign * circuit.omega
        grid_series_z = circuit.grid_r + circuit.grid_l * sideband
        grid_y.append(1 / grid_series_z + circuit.grid_c * sideband)

        # Unknowns are inductor current, node voltage divided by s, and
        # the two integral states. The PI makes voltage O(s) near DC.
        # Solve KVL, KCL and the PI constraints in these scaled variables.
        kp_i = circuit.current_kp[0, 0] + 1j * sign * circuit.current_kp[1, 0]
        kp_v = circuit.voltage_kp[0, 0] + 1j * sign * circuit.voltage_kp[1, 0]
        ki_i, ki_v = circuit.current_ki, circuit.voltage_ki
        matrix = np.zeros((len(s), 4, 4), dtype=complex)
        matrix[:, 0, 0] = circuit.filter_r + circuit.filter_l * sideband + kp_i
        matrix[:, 0, 1] = (1 + kp_i * kp_v) * s
        matrix[:, 0, 2] = -kp_i * ki_v
        matrix[:, 0, 3] = -ki_i
        matrix[:, 1, 0] = -1.
        matrix[:, 1, 1] = circuit.filter_c * sideband * s
        matrix[:, 2, 1] = 1.
        matrix[:, 2, 2] = 1.
        matrix[:, 3, 0] = 1.
        matrix[:, 3, 1] = kp_v * s
        matrix[:, 3, 2] = -ki_v
        matrix[:, 3, 3] = s
        current = np.broadcast_to(np.array([[0.], [1.], [0.], [0.]]), (len(s), 4, 1))
        plant_z.append(s * np.linalg.solve(matrix, current)[:, 1, 0])

    # A three-wire abc branch has two circular modes. Reconstruct dq by
    # expressing a d/q excitation in those modes and combining the responses.
    basis = np.array([[1, 1], [-1j, 1j]]) / np.sqrt(2)
    z = basis @ (np.column_stack(plant_z)[:, :, None] * np.eye(2)) @ basis.conj().T
    y = basis @ (np.column_stack(grid_y)[:, :, None] * np.eye(2)) @ basis.conj().T
    return z, y


def balanced_abc_round_trip(values, times, fundamental_hz):
    angles = 2 * np.pi * fundamental_hz * times[:, None] + np.array([0., -2 * np.pi / 3, 2 * np.pi / 3])
    transform = np.stack((np.cos(angles), -np.sin(angles)), axis=-1)
    abc = (transform @ values[..., None])[..., 0]
    back = 2 / 3 * (transform.swapaxes(-1, -2) @ abc[..., None])[..., 0]
    error = np.max(abs(back - values)) / max(1., np.max(abs(values)))
    imbalance = np.max(abs(abc.sum(axis=1))) / max(1., np.max(abs(abc)))
    return abc, float(error), float(imbalance)
