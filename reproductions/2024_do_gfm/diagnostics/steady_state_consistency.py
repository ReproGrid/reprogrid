"""Evaluate the Fig. 5(b) and 6(b) plateau means using nominal circuit KVL."""

from hashlib import sha256
import json
from math import sqrt
from pathlib import Path


CASE = Path(__file__).resolve().parents[1]


def evaluate(plateau, reactance, resistance, setpoint, power_base_ratio):
    """Use the grid voltage as the phasor reference and export-current sign."""
    means = plateau["last_50_ms_mean"]
    p = means["p_g_pu"]
    q = means["q_g_pu"]
    grid_voltage = means["u_g_pu"]
    command = means["u_c_ref_pu"]
    current = complex(p, -q) / grid_voltage
    impedance = complex(resistance, reactance)
    terminal_voltage = abs(grid_voltage + impedance * current)
    quasi_static_voltage = abs(grid_voltage + 1j * reactance * current)
    q_at_setpoint = (sqrt((setpoint * grid_voltage)**2 - (reactance * p)**2)
                     - grid_voltage**2) / reactance
    base_sensitivity = abs(grid_voltage + impedance * current * power_base_ratio) - terminal_voltage

    return {
        "commanded_active_power_pu": plateau["to_pu"],
        "last_50_ms_mean": means,
        "terminal_voltage_from_pq_pu": terminal_voltage,
        "command_minus_terminal_voltage_from_pq_pu": command - terminal_voltage,
        "quasi_static_voltage_from_pq_pu": quasi_static_voltage,
        "low_angle_reactive_power_at_voltage_setpoint_pu": q_at_setpoint,
        "observed_minus_low_angle_reactive_power_pu": q - q_at_setpoint,
        "change_in_terminal_voltage_if_pq_scaled_by_nameplate_to_space_vector_base_ratio_pu": base_sensitivity,
    }


def main():
    protocol_bytes = (CASE / "protocol.json").read_bytes()
    metrics_bytes = (CASE / "results/metrics.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    metrics = json.loads(metrics_bytes)
    reactance = protocol["weak_grid"]["total_inductance_pu"]
    resistance = protocol["simulation_assumptions"]["plant_series_resistance_pu"]
    setpoint = protocol["weak_grid"]["converter_voltage_setpoint_pu"]
    output = {
        "source_doi": protocol["paper"]["doi"],
        "source_sha256": {
            "protocol.json": sha256(protocol_bytes).hexdigest(),
            "results/metrics.json": sha256(metrics_bytes).hexdigest(),
        },
        "figures": ["5(b)", "6(b)"],
        "window_s": protocol["numerics"]["plateau_window_s"],
        "reactance_pu": reactance,
        "resistance_pu": resistance,
        "quasi_static_voltage_setpoint_pu": setpoint,
        "definitions": {
            "current": "I = (P - jQ) / Ug from paper Eqs. (4) and (5), using real positive Ug",
            "terminal_voltage": "abs(Ug + (R+jX)*I), neglecting the current derivative on each plateau",
            "quasi_static_voltage": "abs(Ug + jX*I) from paper Eq. (2)",
            "low_angle_reactive_power": "(sqrt((Vc_setpoint*Ug)^2 - (X*P)^2) - Ug^2)/X",
        },
        "interpretation": (
            "The voltage residual compares the plotted command with the steady-state terminal fundamental inferred from power. "
            "It is conditional on the nominal series-impedance model and on treating the command as a realized voltage. "
            "The published traces do not supply the realized converter-terminal fundamental or enough independent hardware data to assign the residual to a cause."
        ),
        "runs": {},
    }
    for controller, runs in metrics["runs"].items():
        power_base_ratio = protocol["table_i"]["nameplate_va"] / runs["base"]["p"]
        output["runs"][controller] = {
            label: [evaluate(step, reactance, resistance, setpoint, power_base_ratio)
                    for step in runs[key]]
            for label, key in (("simulation", "switched_steps"), ("published", "paper_steps"))
        }
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
