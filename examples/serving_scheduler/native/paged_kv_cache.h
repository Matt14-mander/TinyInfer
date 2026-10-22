#pragma once

#include "tinyinfer/ops/basic_ops.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace serving {

// Single-layer, single-KV-head FP32 learning backend. The physical pool persists
// across requests; tables own blocks exclusively (no prefix sharing or COW).
class PagedKVCache {
public:
    static constexpr int dimension = 16;

    PagedKVCache(int blocks, int block_size)
        : block_size_(checked_block_size(block_size)), capacity_(checked_capacity(blocks)),
          keys_({capacity_, block_size_, dimension}),
          values_({capacity_, block_size_, dimension}) {
        free_.reserve(capacity_);
        clear();
    }

    int block_size() const { return block_size_; }
    int capacity() const { return capacity_; }
    std::size_t used_blocks() const { return capacity_ - free_.size(); }
    std::size_t peak_blocks() const { return peak_; }
    std::size_t pool_bytes() const { return keys_.size_bytes() + values_.size_bytes(); }
    std::size_t live_bytes() const { return used_blocks() * block_size_ * dimension * 2 * sizeof(float); }
    const std::vector<int>& table(int id) const { return tables_.at(id); }

    // Preflight the entire batch; build changes off to the side. Allocation
    // failure leaves the live tables, free list and peak count unchanged.
    void prepare_append(const std::vector<std::pair<int, int>>& targets) {
        auto next_tables = tables_;
        auto next_free = free_;
        for (const auto& target : targets) {
            if (target.first < 1 || target.second < 1 || target.second > 4096)
                throw std::invalid_argument("invalid paged append target");
            auto& table = next_tables[target.first];
            const auto needed = static_cast<std::size_t>((target.second + block_size_ - 1) / block_size_);
            if (needed < table.size()) throw std::invalid_argument("paged append cannot shrink");
            if (needed - table.size() > next_free.size())
                throw std::runtime_error("paged KV capacity exhausted");
            while (table.size() < needed) {
                table.push_back(next_free.back());
                next_free.pop_back();
            }
        }
        tables_.swap(next_tables);
        free_.swap(next_free);
        peak_ = std::max(peak_, used_blocks());
    }

    void write(int id, int position, const float* key, const float* value) {
        const auto offset = physical_offset(id, position);
        std::copy_n(key, dimension, keys_.data<float>() + offset);
        std::copy_n(value, dimension, values_.data<float>() + offset);
    }

    // Direct block-table traversal with online softmax. No sequence-sized K/V
    // gather, scores or probability tensor; scratch is O(block_size + dimension).
    tinyinfer::Tensor attention(int id, int length, const tinyinfer::Tensor& query) const {
        validate_query(id, length, query);
        std::array<float, dimension> numerator{};
        std::vector<float> scores(std::min(block_size_, length));
        float maximum = -std::numeric_limits<float>::infinity(), denominator = 0.0F;
        const auto* q = query.data<float>();
        const auto& blocks = table(id);
        for (int base = 0; base < length; base += block_size_) {
            const int count = std::min(block_size_, length - base);
            const auto offset = std::size_t(blocks[base / block_size_]) * block_size_ * dimension;
            const auto* keys = keys_.data<float>() + offset;
            const auto* values = values_.data<float>() + offset;
            float block_max = -std::numeric_limits<float>::infinity();
            for (int token = 0; token < count; ++token) {
                float score = 0.0F;
                for (int d = 0; d < dimension; ++d) score += q[d] * keys[token * dimension + d];
                score /= std::sqrt(float(dimension));
                if (!std::isfinite(score)) throw std::runtime_error("nonfinite paged attention score");
                scores[token] = score;
                block_max = std::max(block_max, score);
            }
            const float next_max = std::max(maximum, block_max);
            const float correction = std::exp(maximum - next_max);
            denominator *= correction;
            for (auto& v : numerator) v *= correction;
            for (int token = 0; token < count; ++token) {
                const float weight = std::exp(scores[token] - next_max);
                denominator += weight;
                for (int d = 0; d < dimension; ++d) numerator[d] += weight * values[token * dimension + d];
            }
            maximum = next_max;
        }
        tinyinfer::Tensor result({1, dimension});
        for (int d = 0; d < dimension; ++d) {
            result.at(d) = numerator[d] / denominator;
            if (!std::isfinite(result.at(d))) throw std::runtime_error("nonfinite paged attention output");
        }
        return result;
    }

    // Diagnostic storage oracle: materialize the same logical prefix and use
    // existing TinyInfer MatMul/Softmax, independently of the online reduction.
    tinyinfer::Tensor gather_attention(int id, int length, const tinyinfer::Tensor& query) const {
        validate_query(id, length, query);
        tinyinfer::Tensor keys({length, dimension}), values({length, dimension});
        for (int token = 0; token < length; ++token) {
            const auto offset = physical_offset(id, token);
            std::copy_n(keys_.data<float>() + offset, dimension, keys.data<float>() + token * dimension);
            std::copy_n(values_.data<float>() + offset, dimension, values.data<float>() + token * dimension);
        }
        keys.transpose(0, 1);
        auto scores = tinyinfer::ops::matmul(query, keys);
        for (std::size_t i = 0; i < scores.numel(); ++i) scores.at(i) /= std::sqrt(float(dimension));
        return tinyinfer::ops::matmul(tinyinfer::ops::softmax(scores, -1), values);
    }

    void release(int id) {
        const auto found = tables_.find(id);
        if (found == tables_.end()) throw std::invalid_argument("unknown paged release");
        for (int block : found->second) free_.push_back(block);
        tables_.erase(found);
    }

    void clear(bool reset_peak = false) {
        tables_.clear();
        free_.clear();
        for (int block = capacity_; block > 0; --block) free_.push_back(block - 1);
        if (reset_peak) peak_ = 0;
    }

private:
    static int checked_block_size(int size) {
        if (size < 1 || size > 4096) throw std::invalid_argument("paged block size must be in [1, 4096]");
        return size;
    }
    static int checked_capacity(int blocks) {
        if (blocks < 0) throw std::invalid_argument("negative paged capacity");
        return blocks;
    }
    std::size_t physical_offset(int id, int position) const {
        if (position < 0) throw std::invalid_argument("negative KV position");
        const auto block = table(id).at(position / block_size_);
        return (std::size_t(block) * block_size_ + position % block_size_) * dimension;
    }
    void validate_query(int id, int length, const tinyinfer::Tensor& query) const {
        if (length < 1 || std::size_t(length) > table(id).size() * block_size_ ||
            query.shape() != tinyinfer::Shape({1, dimension}) || !query.is_contiguous() ||
            query.dtype() != tinyinfer::DataType::Float32)
            throw std::invalid_argument("invalid paged attention query");
    }
    int block_size_, capacity_;
    tinyinfer::Tensor keys_, values_;
    std::map<int, std::vector<int>> tables_;
    std::vector<int> free_;
    std::size_t peak_ = 0;
};

} // namespace serving
