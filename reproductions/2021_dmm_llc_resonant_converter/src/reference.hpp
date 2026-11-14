#pragma once

#include "llc.hpp"

#include <algorithm>

struct Factorization {
    std::array<Vec5,5> lu{};
    std::array<int,5> pivot{};

    void factor(const std::array<Vec5,5>& a) {
        lu = a;
        for (int k = 0; k < 5; ++k) {
            int row = k;
            for (int i = k + 1; i < 5; ++i) {
                if (std::abs(lu[i][k]) > std::abs(lu[row][k])) row = i;
            }
            if (lu[row][k] == 0) throw std::runtime_error("Singular full nodal equation");
            pivot[k] = row;
            std::swap(lu[k], lu[row]);
            for (int i = k + 1; i < 5; ++i) {
                lu[i][k] /= lu[k][k];
                for (int j = k + 1; j < 5; ++j) lu[i][j] -= lu[i][k] * lu[k][j];
            }
        }
    }

    Vec5 solve(Vec5 b) const {
        for (int k = 0; k < 5; ++k) std::swap(b[k], b[pivot[k]]);
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < i; ++j) b[i] -= lu[i][j] * b[j];
        }
        for (int i = 4; i >= 0; --i) {
            for (int j = i + 1; j < 5; ++j) b[i] -= lu[i][j] * b[j];
            b[i] /= lu[i][i];
        }
        return b;
    }
};

struct NodalReference {
    const Circuit& c;
    double dt, tie;
    std::array<Factorization,16> matrices;

    NodalReference(const Circuit& circuit, double step, double tie_v) : c(circuit), dt(step), tie(tie_v) {
        for (int mask = 0; mask < 16; ++mask) {
            // Unknowns are i_r, i_m, node x, node y, and the secondary output node.
            double a = c.diode_g(mask, 0), b = c.diode_g(mask, 1);
            double d = c.diode_g(mask, 2), e = c.diode_g(mask, 3);
            double n = c.turns;
            std::array<Vec5, 5> equation{{
                {c.lr / dt + c.bridge_r() + dt / c.cr, 0, n, -n, 0},
                {0, c.lm / dt, -n, n, 0},
                {-n, n, a + b, 0, -a},
                {n, -n, 0, d + e, -d},
                {0, 0, -a, -d, a + d + c.co / dt}
            }};
            if (c.shorted()) equation[4] = {0, 0, 0, 0, 1};
            else equation[4][4] = a + d + c.co / dt + 1 / c.load;
            matrices[mask].factor(equation);
        }
    }

    Result evaluate(const Vec4& x, double drive, int mask) const {
        const auto out = matrices[mask].solve({c.lr / dt * x[IR] + drive - x[VCR],
                                              c.lm / dt * x[IM], 0, 0, c.shorted() ? 0 : c.co / dt * x[VO]});
        return {{out[0], out[1], x[VCR] + (dt / c.cr) * out[0], out[4]}, out[2], out[3], mask};
    }

    double margin(const Result& result) const {
        auto vd = result.diode_v();
        double minimum = INFINITY;
        for (int diode = 0; diode < 4; ++diode) {
            double signed_v = ((result.mask >> diode) & 1) ? vd[diode] : -vd[diode];
            minimum = std::min(minimum, signed_v);
        }
        return minimum;
    }

    Result solve(const Vec4& x, double drive, int previous_mask, std::int64_t& attempts) const {
        Result previous = evaluate(x, drive, previous_mask);
        ++attempts;
        if (margin(previous) > tie) return previous;
        // Try every physical diode combination when the previous mode fails.
        // Only the full nodal solution and the individual diode polarities decide.
        Result best = previous;
        double best_margin = margin(previous);
        for (int mask = 0; mask < 16; ++mask) {
            if (mask == previous_mask) continue;
            Result candidate = evaluate(x, drive, mask);
            ++attempts;
            double candidate_margin = margin(candidate);
            if (candidate_margin > best_margin) {
                best = candidate;
                best_margin = candidate_margin;
            }
        }
        if (best_margin < -tie) throw std::runtime_error("No admissible nodal diode state");
        return best;
    }
};
