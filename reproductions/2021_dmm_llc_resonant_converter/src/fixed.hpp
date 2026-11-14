#pragma once

#include "llc.hpp"

#include <algorithm>

struct Quantization {
    int vector_bits, vector_fraction, coeff_bits, coeff_fraction;
    Vec4 state_scale;
    double input_scale;

    static std::int64_t bound(std::int64_t n, int bits, std::uint64_t& saturated) {
        const std::int64_t limit = std::int64_t{1} << (bits - 1);
        if (n < -limit || n >= limit) ++saturated;
        return std::clamp(n, -limit, limit - 1);
    }

    static std::int64_t encode(double value, int fraction, int bits, std::uint64_t& saturated) {
        return bound(std::llround(std::ldexp(value, fraction)), bits, saturated);
    }

    static std::int64_t round_shift(std::int64_t value, int fraction) {
        const std::int64_t half = std::int64_t{1} << (fraction - 1);
        return value < 0 ? -((-value + half) >> fraction) : (value + half) >> fraction;
    }
};

using StoredState = std::array<std::int64_t, 4>;
using CoeffRow = std::array<std::int64_t, 6>;
using ScaledState = std::array<std::int64_t, 6>;

inline std::int64_t product_sum(const CoeffRow& row, const ScaledState& state) {
    std::int64_t result = 0;
    for (int i = 0; i < 6; ++i) result += row[i] * state[i];
    return result;
}

struct FixedMap {
    const Dmm& map;
    const Quantization& q;
    std::array<std::array<CoeffRow, 4>, 16> step_rows{};
    std::uint64_t matrix_saturation{};

    FixedMap(const Dmm& m, const Quantization& f) : map(m), q(f) {
        for (int mode : modes) {
            for (int row = 0; row < 4; ++row) {
                const auto& values = map.state_rows[mode][row];
                for (int j = 0; j < 5; ++j) {
                    const double scale = j < 4 ? q.state_scale[j] : q.input_scale;
                    step_rows[mode][row][j] = q.encode(values[j] * scale / q.state_scale[row],
                                                       q.coeff_fraction, q.coeff_bits, matrix_saturation);
                }
            }
        }
    }

    StoredState advance(const StoredState& stored, std::int64_t drive, int mode, std::uint64_t& saturated) const {
        ScaledState args{stored[0], stored[1], stored[2], stored[3], drive, 0};
        StoredState result{};
        for (int row = 0; row < 4; ++row) {
            auto scaled = q.round_shift(product_sum(step_rows[mode][row], args), q.coeff_fraction);
            result[row] = q.bound(scaled, q.vector_bits, saturated);
        }
        return result;
    }

    Result physical(const StoredState& x, int mask, const StoredState& previous, double drive) const {
        Vec4 y{}, old{};
        for (int i = 0; i < 4; ++i) {
            y[i] = std::ldexp(double(x[i]), -q.vector_fraction) * q.state_scale[i];
            old[i] = std::ldexp(double(previous[i]), -q.vector_fraction) * q.state_scale[i];
        }
        const auto z = map.advance(old, drive, mask, map.port[mask]);
        return {y, z.v_x, z.v_y, mask};
    }
};

struct FixedClassifier {
    const Dmm& current;
    const Quantization& q;
    std::array<CoeffRow,4> first{};
    std::array<std::array<CoeffRow,4>,16> composed{};
    std::uint64_t saturated{};

    FixedClassifier(const Dmm& previous, const Dmm& next, const Quantization& qn) : current(next), q(qn) {
        const std::array<double,4> slopes{-next.m1, next.m1, -next.m2, next.m2};
        for (int row = 0; row < 4; ++row) {
            std::array<double,6> weights{};
            for (int j = 0; j < 5; ++j) {
                weights[j] = next.c.shorted() ? next.selection_rows[0][j]
                           : slopes[row] * next.selection_rows[0][j] + next.selection_rows[1][j];
            }
            first[row] = encode(weights);
            for (int mask : modes) {
                std::array<double,6> joined{};
                for (int j = 0; j < 5; ++j) {
                    for (int k = 0; k < 4; ++k) joined[j] += weights[k] * previous.state_rows[mask][k][j];
                }
                joined[5] = weights[4];
                composed[mask][row] = encode(joined);
            }
        }
    }

    CoeffRow encode(const std::array<double,6>& w) {
        std::array<double,6> scaled{};
        double maximum = 0;
        for (int i = 0; i < 6; ++i) {
            scaled[i] = w[i] * (i < 4 ? q.state_scale[i] : q.input_scale);
            maximum = std::max(maximum, std::abs(scaled[i]));
        }
        CoeffRow out{};
        if (maximum == 0) return out;
        for (int i = 0; i < 6; ++i) {
            out[i] = q.encode(scaled[i] / maximum, q.coeff_fraction, q.coeff_bits, saturated);
        }
        return out;
    }

    int select(const StoredState& x, std::int64_t old_drive, std::int64_t drive, int prev_mask, bool startup) const {
        const ScaledState input{x[0], x[1], x[2], x[3], startup ? drive : old_drive, startup ? 0 : drive};
        const auto& rows = startup ? first : composed[prev_mask];
        const auto p1 = product_sum(rows[0], input);
        if (current.c.shorted()) return p1 >= 0 ? 9 : 6;
        const auto p2 = product_sum(rows[1], input);
        if (p1 >= 0 && p2 >= 0) return 0;
        const auto p3 = product_sum(rows[2], input);
        const auto p4 = product_sum(rows[3], input);
        if (p3 <= 0 && p4 <= 0) return 15;
        return p2 < 0 ? 6 : 9;
    }
};
