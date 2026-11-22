"""DC microgrid branch equations, equilibrium, and nonlinear time integration."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root


@dataclass(frozen=True)
class Components:
    source_r_ohm: float
    load_r_ohm: float
    source_l_h: float
    load_l_h: float
    bus_c_f: float
    load_c_f: float
    reference_v: float
    line_r_ohm: float


class Network:
    """Use amperes and volts with states ordered [i_s, i_l, v_b, v_l]."""

    def __init__(self, topology, components, droop_ohm):
        self.c = components
        self.n_bus = topology["buses"]
        self.source_buses = np.array(topology["sources"], dtype=int)
        self.load_buses = np.array(topology["loads"], dtype=int)
        self.n_loads = len(self.load_buses)
        n_sources = len(self.source_buses)
        load_stop = n_sources + self.n_loads
        bus_stop = load_stop + self.n_bus
        self.n = bus_stop + self.n_loads
        self.sources = slice(0, n_sources)
        self.loads = slice(n_sources, load_stop)
        self.buses = slice(load_stop, bus_stop)
        self.load_v = slice(bus_stop, self.n)
        self.diagonal = np.arange(bus_stop, self.n)
        self.r_source = self.c.source_r_ohm + droop_ohm

        self.edges = np.array(topology["edges"], dtype=int) - 1
        self.incidence = np.zeros((self.n_bus, len(self.edges)))
        for index, (upstream, downstream) in enumerate(self.edges):
            self.incidence[upstream, index] = 1
            self.incidence[downstream, index] = -1
        self.laplacian = self.incidence @ self.incidence.T / self.c.line_r_ohm
        self.S = np.eye(self.n_bus)[:, self.source_buses - 1]
        self.T = np.eye(self.n_bus)[:, self.load_buses - 1]

        self.A = self._base_matrix()
        self.b = np.zeros(self.n)
        self.b[self.sources] = self.c.reference_v / self.c.source_l_h

        conductance = self.laplacian + self.S @ self.S.T / self.r_source
        self.bus_impedance = np.linalg.solve(conductance, self.T)
        self.open_circuit_bus = np.linalg.solve(conductance, self.S.sum(axis=1) * self.c.reference_v / self.r_source)
        self.load_impedance = self.T.T @ self.bus_impedance + self.c.load_r_ohm * np.eye(self.n_loads)

    def _base_matrix(self):
        c = self.c
        s, l, b, v = self.sources, self.loads, self.buses, self.load_v
        A = np.zeros((self.n, self.n))
        source_identity = np.eye(len(self.source_buses))
        load_identity = np.eye(self.n_loads)
        A[s, s] = -self.r_source / c.source_l_h * source_identity
        A[s, b] = -self.S.T / c.source_l_h
        A[l, l] = -c.load_r_ohm / c.load_l_h * load_identity
        A[l, b] = self.T.T / c.load_l_h
        A[l, v] = -load_identity / c.load_l_h
        A[b, s] = self.S / c.bus_c_f
        A[b, l] = -self.T / c.bus_c_f
        A[b, b] = -self.laplacian / c.bus_c_f
        A[v, l] = load_identity / c.load_c_f
        return A

    def equilibrium(self, power):
        """Solve the high-voltage branch using the open-circuit voltage as the seed."""
        power = np.broadcast_to(power, (self.n_loads,))
        open_load = self.T.T @ self.open_circuit_bus

        def residual(voltage):
            return voltage + self.load_impedance @ (power / voltage) - open_load

        def voltage_jacobian(voltage):
            return np.eye(self.n_loads) - self.load_impedance * (power / voltage**2)

        solution = root(residual, open_load, jac=voltage_jacobian, tol=1e-11)
        voltage = solution.x
        if min(voltage) <= 0 or np.max(np.abs(residual(voltage))) > 1e-8:
            raise RuntimeError(f"No high-voltage equilibrium found for {power} W: {solution.message}")
        load_current = power / voltage
        bus = self.open_circuit_bus - self.bus_impedance @ load_current
        source_current = (self.c.reference_v - self.S.T @ bus) / self.r_source
        return np.r_[source_current, load_current, bus, voltage]

    def delta(self, state, power):
        return power / (self.c.load_c_f * state[self.load_v] ** 2)

    def jacobian(self, delta):
        J = self.A.copy()
        J[self.diagonal, self.diagonal] += delta
        return J

    def rhs(self, state, power):
        derivative = self.A @ state + self.b
        derivative[self.load_v] -= power / (self.c.load_c_f * state[self.load_v])
        return derivative

    def branch_rhs(self, state, power):
        """Assemble Eqs. (1)–(3) directly, using individual branch currents."""
        c = self.c
        source, load = state[self.sources], state[self.loads]
        bus, voltage = state[self.buses], state[self.load_v]
        source_rate = (c.reference_v - self.r_source * source - bus[self.source_buses - 1]) / c.source_l_h
        load_rate = (bus[self.load_buses - 1] - c.load_r_ohm * load - voltage) / c.load_l_h
        node_current = np.zeros(self.n_bus)
        node_current[self.source_buses - 1] += source
        node_current[self.load_buses - 1] -= load
        for left, right in self.edges:
            current = (bus[left] - bus[right]) / c.line_r_ohm
            node_current[left] -= current
            node_current[right] += current
        return np.r_[source_rate, load_rate, node_current / c.bus_c_f, (load - power / voltage) / c.load_c_f]

    def power_imbalance_w(self, state, power):
        line_drop = self.incidence.T @ state[self.buses]
        source, load = state[self.sources], state[self.loads]
        dissipation = (self.r_source * (source @ source) + self.c.load_r_ohm * (load @ load)
                       + (line_drop @ line_drop) / self.c.line_r_ohm)
        return float(self.c.reference_v * np.sum(source) - dissipation - np.sum(np.broadcast_to(power, (self.n_loads,))))

    def modal_eigenvalues(self, delta):
        """Laplacian modes for the homogeneous network with a pair at each bus."""
        if len(self.source_buses) != self.n_bus or self.n_loads != self.n_bus:
            raise ValueError("Modal decomposition requires a source and a CPL at each bus")
        c = self.c
        return np.array([np.linalg.eigvals([
            [-self.r_source / c.source_l_h, 0, -1 / c.source_l_h, 0],
            [0, -c.load_r_ohm / c.load_l_h, 1 / c.load_l_h, -1 / c.load_l_h],
            [1 / c.bus_c_f, -1 / c.bus_c_f, -mode / c.bus_c_f, 0],
            [0, 1 / c.load_c_f, 0, delta],
        ]) for mode in np.linalg.eigvalsh(self.laplacian)])


def graph_symmetries(network, topology):
    """Bus permutations in the dihedral group of Fig. 3 or leaf swap of Fig. 6."""
    maps = []
    if "cycle" in topology:
        cycle = topology["cycle"]
        leaves = {int(bus): leaf for bus, leaf in topology["pendant_buses"].items()}
        for sign in (-1, 1):
            for shift in (0, 2, 4):
                bus_map = {bus: cycle[(sign * j + shift) % len(cycle)] for j, bus in enumerate(cycle)}
                bus_map.update({leaf: leaves[bus_map[bus]] for bus, leaf in leaves.items()})
                maps.append(bus_map)
    elif "symmetric_leaves" in topology:
        a, b = topology["symmetric_leaves"]
        maps.append({a: b, b: a})

    buses = np.arange(1, network.n_bus + 1)
    blocks = (network.source_buses, network.load_buses, buses, network.load_buses)
    permutations = []
    for bus_map in maps:
        permutation = []
        offset = 0
        for block in blocks:
            indices = {bus: i + offset for i, bus in enumerate(block)}
            permutation.extend(indices[bus_map.get(bus, bus)] for bus in block)
            offset += len(block)
        permutations.append(np.array(permutation))
    return permutations


def simulate(network, profile, numerics, method="Radau"):
    """Restart the solver at each step and keep all circuit states continuous."""
    events = profile["first_step_s"] + profile["step_period_s"] * np.arange(profile["steps"])
    limits = np.r_[profile["start_s"], events, profile["stop_s"]]
    powers = 1000 * (profile["first_kw"] + profile["increment_kw"] * np.arange(len(limits) - 1))
    state = network.equilibrium(powers[0])
    samples, trajectories, loads, segments = [], [], [], []

    def cutoff(_, x):
        return np.min(x[network.load_v]) - numerics["integration_voltage_stop_v"]

    cutoff.terminal = True
    cutoff.direction = -1
    for start, stop, watts in zip(limits[:-1], limits[1:], powers, strict=True):
        solution = solve_ivp(
            lambda _, x: network.rhs(x, watts), (start, stop), state, method=method,
            jac=lambda _, x: network.jacobian(network.delta(x, watts)),
            rtol=numerics["integration_rtol"], atol=numerics["integration_atol"],
            dense_output=True, events=cutoff,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        actual_stop = solution.t[-1]
        times = np.r_[np.arange(start, actual_stop - 1e-10, numerics["trajectory_sample_s"]), actual_stop]
        values = solution.sol(times).T
        segments.append({"start_s": float(start), "stop_s": float(actual_stop), "watts_per_load": int(watts),
                         "cutoff": solution.status == 1})
        samples.append(times)
        trajectories.append(values)
        loads.append(np.full(len(times), watts))
        state = solution.y[:, -1]
        if solution.status == 1:
            break
    return np.concatenate(samples), np.concatenate(trajectories), np.concatenate(loads), segments
