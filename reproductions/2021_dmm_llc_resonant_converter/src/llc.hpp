#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

using Tick = std::int64_t;
using Vec4 = std::array<double, 4>;
using Vec5 = std::array<double, 5>;
constexpr std::array<int,4> modes{0, 6, 9, 15};
constexpr Tick ps_per_s = 1000000000000LL;

enum StateIndex { IR, IM, VCR, VO };

struct Circuit {
    double lr, lm, cr, co, load, turns, vin, ron, roff, vo0;
    double rectifier_ron{};
    bool shorted() const { return load == 0; }
    double gon() const { return 1 / ron; }
    double goff() const { return 1 / roff; }
    double bridge_r() const { return 2 / (gon() + goff()); }
    double drive(int gate) const {
        return (gate == 9 ? 1 : -1) * vin * (gon() - goff()) / (gon() + goff());
    }
    double diode_g(int mask, int diode) const {
        double on = rectifier_ron == 0 ? gon() : 1 / rectifier_ron;
        return ((mask >> diode) & 1) ? on : goff();
    }
};

struct Result {
    Vec4 x;
    double v_x, v_y;
    int mask;

    std::array<double, 4> diode_v() const {
        return {v_x - x[VO], -v_x, v_y - x[VO], -v_y};
    }
};

inline double dot(const Vec4& a, const Vec4& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

inline int pair_for_sign(double current) {
    return current >= 0 ? 9 : 6;
}

inline int bridge_mode(double terminal_current, double output_voltage, double off, double on) {
    if (terminal_current == 0 && output_voltage == 0) return 0;
    if (std::abs(terminal_current) <= off * output_voltage) return 0;
    if (std::abs(terminal_current) <= -on * output_voltage) return 15;
    return pair_for_sign(terminal_current);
}

struct Dmm {
    // Symmetric bridge pairs make (v_x + v_y) equal to v_o.
    const Circuit& c;
    double dt, gamma, g1, g2, co_over_dt, m1, m2;
    struct Port { double u_h1, u_h2, vo_h1, vo_h2; };
    std::array<Port,16> port{};
    std::array<std::array<Vec4,2>,16> previous_history_map{};
    std::array<std::array<double,2>,16> previous_drive_map{};
    std::array<Vec5,2> selection_rows{};
    std::array<std::array<Vec5,4>,16> state_rows{};

    Dmm(const Circuit& circuit, double step) : c(circuit), dt(step) {
        gamma = 1 / (c.lr / dt + c.bridge_r() + dt / c.cr);
        g1 = c.turns * c.turns * (gamma + dt / c.lm);
        const double dc = c.shorted() ? 0 : c.co / dt + 1 / c.load;
        const double on = c.diode_g(15, 0);
        g2 = dc;
        co_over_dt = c.shorted() ? 0 : c.co / dt;
        m1 = (c.goff() + dc) / (c.goff() + g1);
        m2 = (on + dc) / (on + g1);
        for (int mask : modes) {
            double a = c.diode_g(mask, 0);
            double b = c.diode_g(mask, 1);
            double gs = (a + b) / 2;
            double gd = (a - b) / 2;
            double det = a * b + gs * (g1 + g2) + g1 * g2;
            if (c.shorted()) {
                port[mask] = {1 / (g1 + gs), 0, 0, 0};
            } else {
                port[mask] = {(g2 + gs) / det, gd / det, gd / det, (g1 + gs) / det};
            }
            const auto& coeff = port[mask];
            for (int j = 0; j < 4; ++j) {
                Vec4 basis{};
                basis[j] = 1;
                Vec4 propagated = advance(basis, 0, mask, coeff).x;
                auto mapped = currents(propagated, 0);
                previous_history_map[mask][0][j] = mapped[0];
                previous_history_map[mask][1][j] = mapped[1];
                for (int row = 0; row < 4; ++row) state_rows[mask][row][j] = propagated[row];
            }
            Vec4 drive_response = advance({}, 1, mask, coeff).x;
            previous_drive_map[mask] = currents(drive_response, 0);
            for (int row = 0; row < 4; ++row) state_rows[mask][row][4] = drive_response[row];
        }
        selection_rows[0] = {c.turns * gamma * (c.lr / dt), -c.turns, -c.turns * gamma, 0, c.turns * gamma};
        selection_rows[1][VO] = co_over_dt;
    }

    std::array<double,2> currents(const Vec4& x, double drive) const {
        double hr = gamma * ((c.lr / dt) * x[IR] + drive - x[VCR]);
        return {c.turns * (hr - x[IM]), co_over_dt * x[VO]};
    }

    std::array<double,2> single_stage_currents(const Vec4& two_steps_ago, double previous_drive,
                                               int previous_mask, double drive) const {
        auto b = previous_drive_map[previous_mask];
        auto& f = previous_history_map[previous_mask];
        return {dot(f[0], two_steps_ago) + b[0] * previous_drive + c.turns * gamma * drive,
                dot(f[1], two_steps_ago) + b[1] * previous_drive};
    }

    int select(const std::array<double,2>& h) const {
        if (c.shorted()) return pair_for_sign(h[0]);
        double magnitude = std::abs(h[0]);
        if (h[1] >= m1 * magnitude) return 0;
        if (h[1] <= -m2 * magnitude) return 15;
        return pair_for_sign(h[0]);
    }

    Result solve(const Vec4& x, double drive, bool pairs_only = false) const {
        auto h = currents(x, drive);
        int mask = pairs_only ? pair_for_sign(h[0]) : select(h);
        return advance(x, drive, mask, port[mask]);
    }

    Result advance(const Vec4& x, double drive, int mask, const Port& coefficients) const {
        auto h = currents(x, drive);
        double u = coefficients.u_h1 * h[0] + coefficients.u_h2 * h[1];
        double vo = coefficients.vo_h1 * h[0] + coefficients.vo_h2 * h[1];
        double ir = gamma * ((c.lr / dt) * x[IR] + drive - x[VCR] - c.turns * u);
        double im = x[IM] + (dt * c.turns / c.lm) * u;
        return {{ir, im, x[VCR] + (dt / c.cr) * ir, vo}, (vo + u) / 2, (vo - u) / 2, mask};
    }
};

struct ForwardEuler {
    const Circuit& c;
    double dt;
    bool full_bridge = false;

    Result point(const Vec4& x) const {
        double i_ac = c.turns * (x[IR] - x[IM]);
        int mask = full_bridge ? bridge_mode(i_ac, x[VO], c.goff(), c.diode_g(15, 0))
                               : pair_for_sign(i_ac);
        double a = c.diode_g(mask, 0), b = c.diode_g(mask, 1);
        double u = (2 * i_ac + (a - b) * x[VO]) / (a + b);
        return {x, (x[VO] + u) / 2, (x[VO] - u) / 2, mask};
    }

    Result solve(const Result& old, double previous_drive) const {
        const auto& x = old.x;
        double a = c.diode_g(old.mask, 0), b = c.diode_g(old.mask, 1);
        double vm = c.turns * (old.v_x - old.v_y);
        double i_dc = (a - b) / (a + b) * c.turns * (x[IR] - x[IM])
                      - 2 * a * b / (a + b) * x[VO];
        Vec4 next{
            x[IR] + (dt / c.lr) * (previous_drive - c.bridge_r() * x[IR] - x[VCR] - vm),
            x[IM] + (dt / c.lm) * vm,
            x[VCR] + (dt / c.cr) * x[IR],
            x[VO] + (dt / c.co) * (i_dc - x[VO] / c.load)
        };
        return point(next);
    }
};
