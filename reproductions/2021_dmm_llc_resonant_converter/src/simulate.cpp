#include "reference.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

struct Settings {
    Circuit c;
    int frequency;
    Tick step, end, tail, stride, gate_step;
    double tie, max_i, max_v;
    Tick ramp_start{}, ramp_end{};
    int ramp_start_hz{}, ramp_end_hz{};
    std::array<Tick,2> figure_centers{};

    double cycles(Tick t) const {
        double initial = double(t) / ps_per_s;
        if (t <= ramp_start) return ramp_start_hz * initial;
        double duration = double(ramp_end - ramp_start) / ps_per_s;
        double active = double(std::min(t, ramp_end) - ramp_start) / ps_per_s;
        return ramp_start_hz * (double(ramp_start) / ps_per_s + active)
               + (ramp_end_hz - ramp_start_hz) * active * active / (2 * duration)
               + ramp_end_hz * double(std::max(Tick{0}, t - ramp_end)) / ps_per_s;
    }

    int gate(Tick t) const {
        if (gate_step) t = (t / gate_step) * gate_step;
        if (!frequency) {
            if (t <= ramp_start) return 2 * ((t * ramp_start_hz) % ps_per_s) < ps_per_s ? 9 : 6;
            double phase = cycles(t);
            return phase - std::floor(phase) < 0.5 ? 9 : 6;
        }
        return 2 * ((t * frequency) % ps_per_s) < ps_per_s ? 9 : 6;
    }
};

struct Checks {
    double current{}, voltage{}, diode{}, kcl{}, energy{}, history{};
    double max_outside_tie_v{};
    std::int64_t attempts{}, mode_mismatches{}, stage_mismatches{};
    std::array<std::int64_t,16> counts{};
};

double max_abs(const Vec4& a) {
    double max_value = 0;
    for (double x : a) max_value = std::max(max_value, std::abs(x));
    return max_value;
}

bool finite(const Vec4& a) {
    return std::all_of(a.begin(), a.end(), [](double x){ return std::isfinite(x); });
}

void sample(std::ofstream& file, Tick t, const Result& result, int gate) {
    const auto& x = result.x;
    const std::array<double,9> row{double(t), x[IR], x[IM], x[VCR], x[VO], result.v_x, result.v_y,
                                    double(result.mask), double(gate)};
    file.write(reinterpret_cast<const char*>(row.data()), sizeof(double) * row.size());
}

void assess(Checks& out, const Settings& s, double dt, const Vec4& old, const Result& a, const Result& b,
            double drive) {
    const Circuit& c = s.c;
    for (int i = 0; i < 2; ++i) out.current = std::max(out.current, std::abs(a.x[i] - b.x[i]));
    for (int i = 2; i < 4; ++i) out.voltage = std::max(out.voltage, std::abs(a.x[i] - b.x[i]));
    auto vd = a.diode_v();
    auto vd_ref = b.diode_v();
    double diode_loss = 0;
    double edge_a = INFINITY, edge_b = INFINITY;
    std::array<double,4> diode_i{};
    for (int k = 0; k < 4; ++k) {
        double voltage = vd[k];
        diode_i[k] = c.diode_g(a.mask, k) * voltage;
        diode_loss += diode_i[k] * voltage;
        out.diode = std::max(out.diode, ((a.mask >> k) & 1) ? -voltage : voltage);
        edge_a = std::min(edge_a, std::abs(voltage));
        edge_b = std::min(edge_b, std::abs(vd_ref[k]));
    }
    if (a.mask != b.mask) {
        ++out.mode_mismatches;
        out.max_outside_tie_v = std::max(out.max_outside_tie_v, std::max(edge_a, edge_b));
    }
    double rect_in = c.turns * (a.x[IR] - a.x[IM]);
    double cap_i = c.co / dt * (a.x[VO] - old[VO]);
    double load_i = a.x[VO] / c.load;
    double kcl = std::max({std::abs(rect_in - diode_i[0] + diode_i[1]),
                          std::abs(-rect_in - diode_i[2] + diode_i[3]),
                          std::abs(diode_i[0] + diode_i[2] - cap_i - load_i)});
    double scale = std::max({1.0, std::abs(rect_in), std::abs(load_i), std::abs(cap_i), max_abs(diode_i)});
    out.kcl = std::max(out.kcl, kcl / scale);

    const Vec4 weights{c.lr, c.lm, c.cr, c.co};
    double energy_delta = 0, be_loss = 0, total_magnitude = 0;
    for (int i = 0; i < 4; ++i) {
        double change = a.x[i] - old[i];
        double de = weights[i] * change * (a.x[i] + old[i]) / 2;
        energy_delta += de;
        be_loss += weights[i] * change * change / 2;
        total_magnitude += std::abs(de);
    }
    double supplied = dt * drive * a.x[IR];
    double resistor_loss = dt * (c.bridge_r() * a.x[IR] * a.x[IR] + diode_loss + a.x[VO] * load_i);
    double denom = std::max(c.vin * dt, total_magnitude + be_loss + resistor_loss + std::abs(supplied));
    out.energy = std::max(out.energy, std::abs(energy_delta + be_loss + resistor_loss - supplied) / denom);
}

Checks probe_quadrants(const Settings& s, const Dmm& dmm, const NodalReference& ref) {
    // Exercise both commuting pairs, blocking, the negative-voltage all-on region,
    // and the two shared boundaries of each conducting pair.
    Checks checks;
    for (int gate : {6, 9}) {
        double drive = s.c.drive(gate);
        for (double h1 : {-100., -1., 1., 100.}) {
            double m = std::abs(h1);
            for (double h2 : {2 * dmm.m1 * m, dmm.m1 * m, 0.5 * dmm.m1 * m, 0.,
                              -0.5 * dmm.m2 * m, -dmm.m2 * m, -2 * dmm.m2 * m}) {
                Vec4 x{0., dmm.gamma * drive - h1 / s.c.turns, 0., h2 * dmm.dt / s.c.co};
                auto a = dmm.solve(x, drive);
                ++checks.counts[a.mask];
                auto b = ref.solve(x, drive, 0, checks.attempts);
                assess(checks, s, dmm.dt, x, a, b, drive);
            }
        }
    }
    return checks;
}

void checks_json(const Checks& c) {
    std::cout << "{\"max_current_delta_a\":" << c.current << ",\"max_voltage_delta_v\":" << c.voltage
              << ",\"max_diode_sign_violation_v\":" << c.diode << ",\"max_kcl_relative\":" << c.kcl
              << ",\"max_be_energy_relative\":" << c.energy << ",\"max_single_stage_history_rel\":" << c.history
              << ",\"max_mode_mismatch_boundary_distance_v\":" << c.max_outside_tie_v
              << ",\"reference_matrix_solves\":" << c.attempts << ",\"mode_mismatches\":" << c.mode_mismatches
              << ",\"single_stage_mode_mismatches\":" << c.stage_mismatches << ",\"mask_counts\":[";
    for (int i = 0; i < 16; ++i) std::cout << (i ? "," : "") << c.counts[i];
    std::cout << "]}";
}

int main(int argc, char** argv) {
    try {
        if (argc != 22 && argc != 28) throw std::runtime_error("Invalid simulator argument count");
        const std::string method = argv[1];
        const std::string path = argv[2];
        Settings s{{std::stod(argv[3]), std::stod(argv[4]), std::stod(argv[5]), std::stod(argv[6]),
                    std::stod(argv[7]), std::stod(argv[8]), std::stod(argv[9]), std::stod(argv[10]),
                    std::stod(argv[11]), std::stod(argv[12])},
                   std::stoi(argv[13]), std::stoll(argv[14]), std::stoll(argv[15]), std::stoll(argv[16]),
                   std::stoll(argv[17]), std::stoll(argv[18]), std::stod(argv[19]), std::stod(argv[20]),
                   std::stod(argv[21])};
        if (argc == 28) {
            s.ramp_start = std::stoll(argv[22]);
            s.ramp_end = std::stoll(argv[23]);
            s.ramp_start_hz = std::stoi(argv[24]);
            s.ramp_end_hz = std::stoi(argv[25]);
            s.figure_centers = {std::stoll(argv[26]), std::stoll(argv[27])};
        }
        if ((!s.frequency && (s.ramp_end <= s.ramp_start || s.ramp_start_hz <= 0 || s.ramp_end_hz <= 0)) ||
            s.step <= 0 || s.stride <= 0 || s.end <= 0 || s.gate_step < 0) {
            throw std::runtime_error("Invalid gate schedule or step");
        }
        const bool verified = method == "verified";
        const bool full_fe = method == "forward_full";
        const bool fe = method == "forward_pairs" || full_fe;
        const bool pairs = method == "backward_pairs";
        if (!(verified || fe || pairs || method == "dmm")) throw std::runtime_error("Unknown solver method");
        const auto& c = s.c;
        const double dt = double(s.step) / ps_per_s;
        Dmm dmm(c, dt);
        NodalReference reference(c, dt, s.tie);
        ForwardEuler euler{c, dt, full_fe};
        std::ofstream wave(path, std::ios::binary), comparison;
        if (verified) comparison.open(path + ".reference", std::ios::binary);
        if (!wave || (verified && !comparison)) throw std::runtime_error("Cannot open trajectory output");

        Result a = euler.point({0, 0, 0, c.vo0}), b = a;
        Vec4 two_steps_ago{};
        int last_gate = s.gate(0);
        double last_drive = c.drive(last_gate);
        Tick last = 0;
        std::string status = "completed";
        Checks checks;
        sample(wave, 0, a, last_gate);
        if (verified) sample(comparison, 0, b, last_gate);
        for (Tick tick = s.step; tick <= s.end; tick += s.step) {
            int gate = s.gate(tick);
            double drive = c.drive(gate);
            Result next = fe ? euler.solve(a, last_drive) : dmm.solve(a.x, drive, pairs);
            if (verified) {
                Result b_next = reference.solve(b.x, drive, b.mask, checks.attempts);
                assess(checks, s, dt, a.x, next, b_next, drive);
                if (tick > s.step) {
                    auto direct = dmm.currents(a.x, drive);
                    auto delayed = dmm.single_stage_currents(two_steps_ago, last_drive, a.mask, drive);
                    for (int i = 0; i < 2; ++i) {
                        checks.history = std::max(checks.history, std::abs(direct[i] - delayed[i]) /
                                                   std::max({1.0, std::abs(direct[0]), std::abs(direct[1])}));
                    }
                    if (dmm.select(delayed) != next.mask) ++checks.stage_mismatches;
                }
                b = b_next;
            }
            two_steps_ago = a.x;
            a = next;
            last_drive = drive;
            last = tick;
            ++checks.counts[a.mask];
            if (!finite(a.x) || std::abs(a.x[IR]) > s.max_i || std::abs(a.x[IM]) > s.max_i ||
                std::abs(a.x[VCR]) > s.max_v * c.vin || std::abs(a.x[VO]) > s.max_v * c.vin) {
                status = "numerical_cutoff";
            }
            bool zoom = !s.frequency && (std::abs(tick - s.figure_centers[0]) <= 5'000'000 ||
                                         std::abs(tick - s.figure_centers[1]) <= 5'000'000);
            if (tick >= s.tail || zoom || tick % s.stride < s.step || status != "completed" || tick + s.step > s.end) {
                sample(wave, tick, a, gate);
                if (verified) sample(comparison, tick, b, gate);
            }
            if (status != "completed") break;
        }
        if (!wave || (verified && !comparison)) throw std::runtime_error("Incomplete output write");
        std::cout << std::setprecision(17) << "{\"status\":\"" << status << "\",\"last_ps\":" << last
                  << ",\"steps\":" << last / s.step << ",\"m1\":" << dmm.m1 << ",\"m2\":" << dmm.m2
                  << ",\"checks\":";
        checks_json(checks);
        if (verified) {
            std::cout << ",\"probes\":";
            checks_json(probe_quadrants(s, dmm, reference));
        }
        std::cout << "}\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
