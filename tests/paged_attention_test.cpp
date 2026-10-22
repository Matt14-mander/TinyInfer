#include "../examples/serving_scheduler/native/paged_kv_cache.h"
#include <iostream>
#include <string>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template <typename F> void rejects(F action) {
    bool failed = false;
    try { action(); } catch (const std::exception&) { failed = true; }
    require(failed, "expected operation to fail");
}
void close(const tinyinfer::Tensor& a, const tinyinfer::Tensor& b) {
    for (std::size_t i = 0; i < a.numel(); ++i)
        require(std::isfinite(a.at(i)) && std::isfinite(b.at(i)) &&
                std::abs(a.at(i) - b.at(i)) <= 2e-5F + 2e-5F * std::abs(b.at(i)), "attention differs");
}

void allocation_transaction() {
    serving::PagedKVCache pool(3, 2);
    pool.prepare_append({{1, 2}});
    const auto original = pool.table(1);
    rejects([&] { pool.prepare_append({{1, 4}, {2, 4}}); });
    require(pool.table(1) == original && pool.used_blocks() == 1 && pool.peak_blocks() == 1,
            "failed append mutated pool");
    rejects([&] { pool.table(2); });
    pool.prepare_append({{2, 2}, {1, 4}});
    require(pool.table(1) == std::vector<int>({0, 2}), "growth must support noncontiguous blocks");
    require(pool.used_blocks() == 3, "incorrect block count");
    rejects([&] { pool.prepare_append({{1, 1}}); });
}

void causal_tail_and_stability() {
    serving::PagedKVCache pool(5, 2);
    pool.prepare_append({{1, 2}, {2, 2}});
    pool.prepare_append({{1, 5}});
    tinyinfer::Tensor query({1, 16});
    std::array<float, 16> key{}, value{};
    const float key_values[] = {-1000, -999, 1000, 999, 1001};
    for (int d = 0; d < 16; ++d) query.at(d) = 1;
    for (int token = 0; token < 5; ++token) {
        for (int d = 0; d < 16; ++d) { key[d] = key_values[token]; value[d] = float(token) + float(d) / 10; }
        pool.write(1, token, key.data(), value.data());
    }
    // Poison unused tail storage. No future query/unused slot may enter softmax.
    key.fill(std::numeric_limits<float>::quiet_NaN());value = key;
    pool.write(1, 5, key.data(), value.data());
    for (int length = 1; length <= 5; ++length) {
        const auto direct = pool.attention(1, length, query);
        close(direct, pool.gather_attention(1, length, query));
        const double maximum = double(*std::max_element(key_values, key_values + length)) * 4;
        double denom = 0, numerator = 0;
        for (int t = 0; t < length; ++t) {
            const double weight = std::exp(double(key_values[t]) * 4 - maximum);
            denom += weight;numerator += weight * t;
        }
        for (int d = 0; d < 16; ++d)
            require(std::abs(direct.at(d) - (numerator / denom + double(d) / 10)) < 2e-5,
                    "online softmax differs from double precision reference");
    }
    rejects([&] { pool.attention(1, 0, query); });
    rejects([&] { pool.attention(1, 7, query); });
    rejects([&] { pool.attention(1, 1, tinyinfer::Tensor({2, 16})); });
    pool.release(2);
    pool.prepare_append({{3, 1}});
    require(pool.table(3) == std::vector<int>({1}), "released block not reused");
    key.fill(123);value.fill(-77);pool.write(3, 0, key.data(), value.data());
    close(pool.attention(1, 5, query), pool.gather_attention(1, 5, query));
    for (int d = 0; d < 16; ++d) require(pool.attention(3, 1, query).at(d) == -77, "request contamination");
}

void lifecycle_and_validation() {
    rejects([] { serving::PagedKVCache pool(-1, 4); });
    rejects([] { serving::PagedKVCache pool(1, 0); });
    rejects([] { serving::PagedKVCache pool(1, 4097); });
    serving::PagedKVCache empty(0, 4);
    rejects([&] { empty.prepare_append({{1, 1}}); });
    require(empty.used_blocks() == 0 && empty.pool_bytes() == 0, "zero capacity allocated");
    serving::PagedKVCache pool(2, 3);
    const auto bytes = pool.pool_bytes();
    require(bytes == 2 * 3 * 16 * 2 * sizeof(float), "pool bytes mismatch");
    pool.prepare_append({{1, 4}});
    require(pool.live_bytes() == bytes, "live bytes mismatch");
    pool.release(1);
    rejects([&] { pool.release(1); });
    require(pool.used_blocks() == 0 && pool.peak_blocks() == 2 && pool.pool_bytes() == bytes, "release lifecycle");
    pool.prepare_append({{2, 1}});
    pool.clear();
    require(pool.used_blocks() == 0 && pool.peak_blocks() == 2, "clear must retain peak");
    pool.clear(true);
    require(pool.peak_blocks() == 0 && pool.pool_bytes() == bytes, "reset must retain pool only");
}
} // namespace

int main() {
    try {
        allocation_transaction();causal_tail_and_stability();lifecycle_and_validation();
        std::cout << "Paged KV allocation, attention and lifecycle checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';return 1;
    }
}
