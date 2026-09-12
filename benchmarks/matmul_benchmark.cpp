#include "tinyinfer/backend/cpu/matmul_kernel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

tinyinfer::Tensor make_matrix(std::size_t size, std::size_t seed) {
    std::vector<float> values(size * size);
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<float>(static_cast<int>((i * 17 + seed * 13) % 101) - 50) / 50.0F;
    }
    return tinyinfer::Tensor::from_vector(
        {static_cast<std::int64_t>(size), static_cast<std::int64_t>(size)}, values);
}

void verify_close(const tinyinfer::Tensor& expected, const tinyinfer::Tensor& actual) {
    for (std::size_t i = 0; i < expected.numel(); ++i) {
        if (std::fabs(expected.at(i) - actual.at(i)) > 1e-3F) {
            throw std::runtime_error("MatMul benchmark correctness check failed");
        }
    }
}

template <typename Function>
double measure(Function&& function, std::size_t iterations) {
    function();  // Warm-up.
    const auto start = Clock::now();
    for (std::size_t i = 0; i < iterations; ++i) function();
    return std::chrono::duration<double>(Clock::now() - start).count() /
           static_cast<double>(iterations);
}

void print_result(const char* kernel, std::size_t size, double seconds) {
    const auto operations = 2.0 * size * size * size;
    std::cout << std::left << std::setw(12) << kernel << std::right
              << std::setw(8) << size << std::setw(14) << std::fixed
              << std::setprecision(3) << seconds * 1000.0 << std::setw(14)
              << operations / seconds / 1e9 << '\n';
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<std::size_t> sizes{64, 128, 256, 512};
    if (argc > 1) {
        sizes.clear();
        for (int i = 1; i < argc; ++i) sizes.push_back(std::stoull(argv[i]));
    }
    std::cout << std::left << std::setw(12) << "kernel" << std::right
              << std::setw(8) << "size" << std::setw(14) << "time_ms"
              << std::setw(14) << "GFLOP/s" << '\n';

    for (const auto size : sizes) {
        const auto lhs = make_matrix(size, 1);
        const auto rhs = make_matrix(size, 2);
        tinyinfer::Tensor reference({static_cast<std::int64_t>(size), static_cast<std::int64_t>(size)});
        tinyinfer::Tensor strided(reference.shape());
        tinyinfer::Tensor blocked(reference.shape());
        tinyinfer::Tensor packed_simd(reference.shape());
        const tinyinfer::cpu::PackedMatMulRhs packed_rhs(rhs);
        const auto iterations = std::max<std::size_t>(1, 256 / size);

        if (size <= 128) {
            const auto seconds = measure(
                [&] { tinyinfer::cpu::matmul_reference(lhs, rhs, reference); }, iterations);
            print_result("reference", size, seconds);
        } else {
            tinyinfer::cpu::matmul_strided(lhs, rhs, reference);
        }
        auto seconds = measure(
            [&] { tinyinfer::cpu::matmul_strided(lhs, rhs, strided); }, iterations);
        print_result("strided", size, seconds);
        seconds = measure(
            [&] { tinyinfer::cpu::matmul_blocked(lhs, rhs, blocked); }, iterations);
        print_result("blocked", size, seconds);
        seconds = measure(
            [&] {
                tinyinfer::cpu::matmul_packed_simd(
                    lhs, packed_rhs, packed_simd);
            },
            iterations);
        print_result("packed_simd", size, seconds);
        verify_close(reference, strided);
        verify_close(reference, blocked);
        verify_close(reference, packed_simd);
    }
}
