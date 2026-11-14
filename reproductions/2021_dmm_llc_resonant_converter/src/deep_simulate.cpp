#include "fixed.hpp"
#include "reference.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>

struct Schedule {
    Tick stop, fault_on, fault_off, ramp_on, ramp_off, dwell, clock, overview, transition;
    int start_length, end_length;
    bool reset;
    using Phase = __int128_t;
    Phase denominator = 1, phase = 0;
    Tick last_time{}, next_change{};
    int length;

    Schedule(Tick stop_t, Tick on, Tick off, Tick start, Tick end, Tick hold, Tick gate_clock,
             Tick overview_period, Tick transition_period, int initial, int final, bool reset_phase)
        : stop(stop_t), fault_on(on), fault_off(off), ramp_on(start), ramp_off(end), dwell(hold), clock(gate_clock),
          overview(overview_period), transition(transition_period), start_length(initial), end_length(final), reset(reset_phase),
          next_change(start), length(initial) {
        for (int n = final; n <= initial; ++n) {
            denominator = denominator / std::gcd(int(denominator % n), n) * n;
        }
    }

    void update(Tick t) {
        while (t >= next_change && length > end_length) {
            advance_to(next_change);
            if (reset) phase = 0;
            next_change += dwell;
            --length;
        }
        advance_to(t);
    }

    void advance_to(Tick t) {
        // The common denominator spans all integer half periods in the source schedule.
        Tick elapsed = t / clock - last_time / clock;
        if (elapsed) {
            phase += (denominator / length) * elapsed;
            const Phase period = 2 * denominator;
            while (phase >= period) phase -= period;
        }
        last_time = t;
    }

    int gate() const { return phase < denominator ? 9 : 6; }
    int load_index(Tick t) const { return fault_on <= t && t < fault_off ? 1 : 0; }
    double hz() const { return double(ps_per_s) / (2 * clock * length); }

    bool dense(Tick t) const {
        return std::abs(t - ps_per_s / 20) <= 5'000'000
               || std::abs(t - fault_on) <= 10'000'000
               || std::abs(t - fault_off) <= 15'000'000
               || std::abs(t - (ramp_on + dwell)) <= 400'000'000
               || std::abs(t - 55 * ps_per_s / 100) <= 4'000'000;
    }

    bool sampling(Tick t, Tick dt) const {
        if (t == stop || dense(t)) return true;
        const bool restoring = t > fault_off && t - fault_off <= 1'000'000'000;
        Tick stride = restoring ? transition : overview;
        return t % stride < dt;
    }
};

struct SegmentError {
    std::array<long double, 3> residual{}, baseline{};
    std::array<double, 3> max_abs{};
    std::uint64_t steps{};

    void add(const Result& first, const Result& second) {
        constexpr std::array<int, 3> state{VO, IR, IM};
        for (int i = 0; i < 3; ++i) {
            double base = first.x[state[i]];
            double delta = second.x[state[i]] - base;
            residual[i] += delta * delta;
            baseline[i] += base * base;
            max_abs[i] = std::max(max_abs[i], std::abs(delta));
        }
        ++steps;
    }

    void print() const {
        std::cout << "{\"steps\":" << steps << ",\"percent\":[";
        for (int i = 0; i < 3; ++i) {
            if (i) std::cout << ',';
            if (baseline[i] > 0) std::cout << 100 * std::sqrt(residual[i] / baseline[i]);
            else std::cout << "null";
        }
        std::cout << "],\"max_abs\":[" << max_abs[0] << ',' << max_abs[1] << ',' << max_abs[2] << "]}";
    }

};

void print_errors(const std::array<SegmentError, 6>& errors) {
    std::cout << '[';
    for (std::size_t i = 0; i < errors.size(); ++i) {
        if (i) std::cout << ',';
        errors[i].print();
    }
    std::cout << ']';
}

class Stage {
    const std::array<Dmm, 2>& maps;
    Vec4 previous{}, oldest{};
    double previous_drive{};
    int previous_mask{}, previous_load{};
    Tick samples{};
    std::array<std::array<std::array<Vec5, 2>, 16>, 4> joined{};

public:
    explicit Stage(const std::array<Dmm, 2>& m) : maps(m) {
        for (int previous_load = 0; previous_load < 2; ++previous_load) {
            for (int current_load = 0; current_load < 2; ++current_load) {
                for (int mask : modes) {
                    for (int row = 0; row < 2; ++row) {
                        auto& coefficient = joined[2 * previous_load + current_load][mask][row];
                        for (int j = 0; j < 5; ++j) {
                            for (int k = 0; k < 4; ++k) {
                                coefficient[j] += maps[current_load].selection_rows[row][k] * maps[previous_load].state_rows[mask][k][j];
                            }
                        }
                    }
                }
            }
        }
    }

    Result step(int load, double drive) {
        const auto& current = maps[load];
        int mask;
        if (!samples) {
            mask = current.select(current.currents(previous, drive));
        } else {
            std::array<double,2> history{};
            const auto& coefficient = joined[2 * previous_load + load][previous_mask];
            for (int row = 0; row < 2; ++row) {
                const auto& w = coefficient[row];
                history[row] = (oldest[0] * w[0] + oldest[1] * w[1] + oldest[2] * w[2] + oldest[3] * w[3]
                                + w[4] * previous_drive + current.selection_rows[row][4] * drive);
            }
            mask = current.select(history);
        }
        auto out = current.advance(previous, drive, mask, current.port[mask]);
        oldest = previous;
        previous = out.x;
        previous_drive = drive;
        previous_mask = mask;
        previous_load = load;
        ++samples;
        return out;
    }
};

struct FixedPipeline {
    const Quantization& q;
    std::array<FixedMap, 2> maps;
    std::array<FixedClassifier, 4> classifiers;
    StoredState old{}, older{};
    int last_load{}, last_mask{};
    std::int64_t last_drive{};
    bool first = true;
    std::uint64_t vector_saturation{}, coefficients_saturated{};

    FixedPipeline(const std::array<Dmm, 2>& dmm, const Quantization& quant)
        : q(quant), maps{FixedMap(dmm[0], q), FixedMap(dmm[1], q)},
          classifiers{FixedClassifier(dmm[0], dmm[0], q), FixedClassifier(dmm[0], dmm[1], q),
                      FixedClassifier(dmm[1], dmm[0], q), FixedClassifier(dmm[1], dmm[1], q)} {
        for (const auto& m : maps) coefficients_saturated += m.matrix_saturation;
        for (const auto& c : classifiers) coefficients_saturated += c.saturated;
    }

    Result step(int load, double drive) {
        auto drive_code = Quantization::encode(drive / q.input_scale, q.vector_fraction, q.vector_bits, vector_saturation);
        auto& classifier = classifiers[2 * last_load + load];
        int mask = classifier.select(first ? old : older, last_drive, drive_code, last_mask, first);
        auto next = maps[load].advance(old, drive_code, mask, vector_saturation);
        auto physical = maps[load].physical(next, mask, old, drive);
        older = old;
        old = next;
        last_drive = drive_code;
        last_load = load;
        last_mask = mask;
        first = false;
        return physical;
    }
};

void record(std::ofstream& file, Tick t, const Result& value, int gate, double f, bool load) {
    std::array<double, 11> data{double(t), value.x[IR], value.x[IM], value.x[VCR], value.x[VO], value.v_x, value.v_y,
                                double(value.mask), double(gate), f, double(load)};
    file.write(reinterpret_cast<const char*>(data.data()), sizeof(double) * data.size());
}

int main(int argc, char** argv) {
    try {
        if (argc != 32) throw std::runtime_error("Invalid number of deep replay arguments");
        int k = 1;
        const std::string path = argv[k++];
        const auto real = [&](){ return std::stod(argv[k++]); };
        const auto integer = [&](){ return std::stoll(argv[k++]); };
        Circuit nominal{real(), real(), real(), real(), real(), real(), real(), real(), real(), 0};
        nominal.rectifier_ron = real();
        double shunt_ohm = real();
        const Tick dt_ps = integer();
        const double dt = double(dt_ps) / ps_per_s;
        Schedule pulse{integer(), integer(), integer(), integer(), integer(), integer(), integer(), integer(), integer(),
                       int(integer()), int(integer()), bool(integer())};
        const bool compare_reference = bool(integer());
        const bool compare_pipeline = bool(integer());
        const int vector_bits = int(integer()), vector_fraction = int(integer());
        const int coefficient_bits = int(integer()), coefficient_fraction = int(integer());
        if (dt_ps <= 0 || pulse.clock <= 0 || pulse.clock % dt_ps || pulse.dwell <= 0 || pulse.end_length <= 0 ||
            pulse.start_length > 64 || pulse.start_length <= pulse.end_length || pulse.overview <= 0 || pulse.transition <= 0 ||
            pulse.ramp_on + (pulse.start_length - pulse.end_length - 1) * pulse.dwell != pulse.ramp_off ||
            shunt_ohm < 0 || pulse.stop <= pulse.fault_off || pulse.ramp_on < pulse.fault_off ||
            vector_bits <= 1 || vector_bits > 25 || coefficient_bits <= 1 || coefficient_bits > 35 ||
            vector_fraction <= 0 || vector_fraction >= vector_bits || coefficient_fraction <= 0 ||
            coefficient_fraction >= coefficient_bits) {
            throw std::runtime_error("Invalid simulation clock, shunt or event times");
        }
        Circuit changing = nominal;
        changing.load = shunt_ohm == 0 ? 0 : 1 / (1 / nominal.load + 1 / shunt_ohm);
        const std::array<Dmm, 2> maps{Dmm(nominal, dt), Dmm(changing, dt)};
        const std::array<NodalReference, 2> refs{NodalReference(nominal, dt, 1e-9), NodalReference(changing, dt, 1e-9)};
        auto stage = compare_pipeline ? std::make_unique<Stage>(maps) : nullptr;
        auto scale_i = 2 * nominal.vin / std::sqrt(nominal.lr / nominal.cr);
        Quantization quant{vector_bits, vector_fraction, coefficient_bits, coefficient_fraction,
                           {scale_i, scale_i, 2 * nominal.vin, 2 * nominal.vin / nominal.turns}, nominal.vin};
        auto fixed = compare_pipeline ? std::make_unique<FixedPipeline>(maps, quant) : nullptr;
        constexpr int delay = 2;
        std::array<Result, delay> pipeline_out{};
        std::array<SegmentError, 6> nodal_error, stage_error, fixed_error;
        SegmentError latency;
        std::ofstream base_file(path, std::ios::binary), stage_file, fixed_file;
        if (compare_pipeline) {
            stage_file.open(path + ".stage", std::ios::binary);
            fixed_file.open(path + ".fixed", std::ios::binary);
        }
        if (!base_file || (compare_pipeline && (!stage_file || !fixed_file))) throw std::runtime_error("Cannot write trajectories");
        Vec4 x{}, reference{};
        int old_mask = 0, old_load = 0;
        std::int64_t tries = 0, mask_diff = 0, stage_mask_diff = 0, fixed_mask_diff = 0, delay_steps_checked = 0;
        double diode_violation = 0, max_kcl = 0, shock_energy = 0, peak_fault_current = 0;
        double min_fault_vo = INFINITY, prefault_volts = 0, last_fault_vo = 0;
        Tick fault_first = -1, clear_first = -1, recovery = -1;
        const std::array<Tick, 4> cuts{pulse.fault_on, pulse.fault_off, pulse.ramp_on, pulse.ramp_off};
        const Tick start_recovery = pulse.fault_off;
        int interval = 0;
        Result initial{{}, 0, 0, 0};
        record(base_file, 0, initial, 9, pulse.hz(), false);
        if (compare_pipeline) {
            record(stage_file, delay * dt_ps, initial, 9, pulse.hz(), false);
            record(fixed_file, 0, initial, 9, pulse.hz(), false);
        }
        for (Tick t = dt_ps; t <= pulse.stop; t += dt_ps) {
            pulse.update(t);
            const int active = pulse.load_index(t), gate = pulse.gate();
            if (interval < 4 && t >= cuts[interval]) ++interval;
            const auto& model = active ? changing : nominal;
            const auto& dmm = maps[active];
            const double drive = model.drive(gate);
            auto now = dmm.solve(x, drive);
            if (!std::all_of(now.x.begin(), now.x.end(), [](double state){ return std::isfinite(state); }) || std::abs(now.x[IR]) > 10000) {
                throw std::runtime_error("Out-of-range LLC transient");
            }
            const auto vd = now.diode_v();
            std::array<double, 4> currents{};
            double min_edge = INFINITY;
            for (int d = 0; d < 4; ++d) {
                currents[d] = model.diode_g(now.mask, d) * vd[d];
                diode_violation = std::max(diode_violation, (now.mask >> d) & 1 ? -vd[d] : vd[d]);
                min_edge = std::min(min_edge, std::abs(vd[d]));
            }
            const double iin = model.turns * (now.x[IR] - now.x[IM]);
            double kcl = std::max(std::abs(iin - currents[0] + currents[1]), std::abs(-iin - currents[2] + currents[3]));
            double normalizer = std::max({1.0, std::abs(iin), std::abs(currents[0]), std::abs(currents[2])});
            if (!model.shorted()) {
                double cp = model.co / dt * (now.x[VO] - x[VO]);
                double consumption = now.x[VO] / model.load;
                kcl = std::max(kcl, std::abs(currents[0] + currents[2] - cp - consumption));
                normalizer = std::max({normalizer, std::abs(cp), std::abs(consumption)});
            }
            max_kcl = std::max(max_kcl, kcl / normalizer);
            if (active && !old_load) {
                fault_first = t;
                prefault_volts = x[VO];
                if (model.shorted()) shock_energy = model.co * x[VO] * x[VO] / 2;
            }
            if (old_load && !active) clear_first = t;
            if (active) {
                peak_fault_current = std::max(peak_fault_current, std::abs(now.x[IR]));
                min_fault_vo = std::min(min_fault_vo, now.x[VO]);
                last_fault_vo = now.x[VO];
            }
            if (t >= start_recovery && t < pulse.ramp_on && recovery < 0 && now.x[VO] >= 0.9 * prefault_volts) recovery = t;
            if (compare_reference) {
                auto independent = refs[active].solve(reference, drive, old_mask, tries);
                nodal_error[interval].add(now, independent);
                nodal_error.back().add(now, independent);
                if (independent.mask != now.mask && min_edge > 1e-9) ++mask_diff;
                old_mask = independent.mask;
                reference = independent.x;
            }
            bool save = pulse.sampling(t, dt_ps) || t == fault_first || t == clear_first;
            if (save) record(base_file, t, now, gate, pulse.hz(), active);
            if (compare_pipeline) {
                auto staged = stage->step(active, drive);
                auto quantized = fixed->step(active, drive);
                stage_error[interval].add(now, staged);
                stage_error.back().add(now, staged);
                fixed_error[interval].add(now, quantized);
                fixed_error.back().add(now, quantized);
                if (staged.mask != now.mask && min_edge > 1e-9) ++stage_mask_diff;
                if (quantized.mask != now.mask) ++fixed_mask_diff;
                auto slot = (t / dt_ps) % delay;
                if (t > delay * dt_ps) {
                    latency.add(now, pipeline_out[slot]);
                    ++delay_steps_checked;
                }
                pipeline_out[slot] = staged;
                if (save) {
                    record(stage_file, t + delay * dt_ps, staged, gate, pulse.hz(), active);
                    record(fixed_file, t, quantized, gate, pulse.hz(), active);
                }
            }
            x = now.x;
            old_load = active;
        }
        if (!base_file || (compare_pipeline && (!stage_file || !fixed_file))) throw std::runtime_error("Incomplete transient output");
        std::cout << std::setprecision(17) << "{\"steps\":" << pulse.stop / dt_ps
                  << ",\"first_fault_tick_ps\":" << fault_first << ",\"first_clear_tick_ps\":" << clear_first
                  << ",\"first_recovery_90pct_ps\":" << recovery
                  << ",\"prefault_voltage_v\":" << prefault_volts
                  << ",\"ideal_capacitor_energy_removed_j\":" << shock_energy
                  << ",\"fault_peak_resonant_current_a\":" << peak_fault_current
                  << ",\"fault_min_output_v\":" << min_fault_vo << ",\"max_relative_kcl\":" << max_kcl
                  << ",\"fault_end_output_v\":" << last_fault_vo
                  << ",\"max_diode_polarity_violation_v\":" << diode_violation
                  << ",\"nodal_solves\":" << tries << ",\"nodal_mode_mismatches_outside_tie\":" << mask_diff
                  << ",\"single_stage_mode_mismatches_outside_tie\":" << stage_mask_diff
                  << ",\"fixed_point_mode_mismatches\":" << fixed_mask_diff;
        if (compare_reference) {
            std::cout << ",\"float_nodal_segments\":";
            print_errors(nodal_error);
        }
        if (compare_pipeline) {
            std::cout << ",\"assumed_io_delay_steps\":" << delay << ",\"unaligned_steps\":" << delay_steps_checked
                      << ",\"fixed_vector_saturations\":" << fixed->vector_saturation
                      << ",\"fixed_coefficient_saturations\":" << fixed->coefficients_saturated
                      << ",\"single_stage_segments\":";
            print_errors(stage_error);
            std::cout << ",\"single_stage_uncompensated_delay\":";
            latency.print();
            std::cout << ",\"fixed_point_segments\":";
            print_errors(fixed_error);
        }
        std::cout << "}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
