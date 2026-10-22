#pragma once

#include "tinyinfer/ops/basic_ops.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace serving {

// Single-layer, single-KV-head FP32 learning backend. The physical pool persists
// across requests. Optional exact-prefix snapshots retain blocks with refcounts;
// appending to a shared partial block requires copy-on-write.
class PagedKVCache {
public:
    static constexpr int dimension = 16;

    PagedKVCache(int blocks, int block_size)
        : block_size_(checked_block_size(block_size)), capacity_(checked_capacity(blocks)),
          keys_({capacity_, block_size_, dimension}),
          values_({capacity_, block_size_, dimension}) {
        free_.reserve(capacity_);
        references_.resize(capacity_);
        clear();
    }

    int block_size() const { return block_size_; }
    int capacity() const { return capacity_; }
    std::size_t used_blocks() const { return capacity_ - free_.size(); }
    std::size_t peak_blocks() const { return peak_; }
    std::size_t pool_bytes() const { return keys_.size_bytes() + values_.size_bytes(); }
    std::size_t live_bytes() const { return used_blocks() * block_size_ * dimension * 2 * sizeof(float); }
    const std::vector<int>& table(int id) const { return tables_.at(id); }
    const float* keys_data() const { return keys_.data<float>(); }
    const float* values_data() const { return values_.data<float>(); }
    std::size_t prefix_hits() const { return prefix_hits_; }
    std::size_t prefix_tokens() const { return prefix_tokens_; }
    std::size_t cow_copies() const { return cow_copies_; }
    std::size_t cache_entries() const { return prefixes_.size(); }
    std::size_t cache_evictions() const { return evictions_; }

    // Longest exact causal prefix, scoped to this worker/model. Keep at least
    // one query row for the caller to produce logits (max_tokens is exclusive
    // of that row). A cache entry can include a partial final block.
    int reuse_prefix(int id, const std::vector<int>& tokens, int max_tokens) {
        if (tables_.count(id)) throw std::invalid_argument("prefix target already exists");
        auto best = prefixes_.end();
        for (auto it = prefixes_.begin(); it != prefixes_.end(); ++it) {
            if (int(it->first.size()) <= max_tokens && it->first.size() <= tokens.size() &&
                (best == prefixes_.end() || it->first.size() > best->first.size()) &&
                std::equal(it->first.begin(), it->first.end(), tokens.begin())) best = it;
        }
        if (best == prefixes_.end()) return 0;
        auto mapping = best->second;
        tables_.emplace(id, std::move(mapping));
        lengths_[id] = int(best->first.size());
        for (int block : best->second) ++references_[block];
        ++prefix_hits_;prefix_tokens_ += best->first.size();
        return int(best->first.size());
    }

    void publish_prefix(int id, const std::vector<int>& tokens) {
        if (tokens.empty() || lengths_.at(id) != int(tokens.size()))
            throw std::invalid_argument("prefix length mismatch");
        std::vector<int> ends;
        for (int end = block_size_; end < int(tokens.size()); end += block_size_) ends.push_back(end);
        if (tokens.size() > 1) ends.push_back(int(tokens.size()) - 1);
        ends.push_back(int(tokens.size()));
        for (int end : ends) {
            std::vector<int> key(tokens.begin(), tokens.begin() + end);
            if (prefixes_.count(key)) continue;
            const auto& source = table(id);
            std::vector<int> mapping(source.begin(), source.begin() + (end + block_size_ - 1) / block_size_);
            prefixes_.emplace(std::move(key), mapping);
            for (int block : mapping) ++references_[block];
        }
    }

    bool can_append(const std::vector<std::pair<int, int>>& targets) {
        try { prepare_append(targets, false);return true; }
        catch (const std::length_error&) { return false; }
    }

    // Preflight the entire batch; build changes off to the side. Allocation
    // failure leaves the live tables, free list and peak count unchanged.
    void prepare_append(const std::vector<std::pair<int, int>>& targets, bool commit = true) {
        auto next_tables = tables_;
        auto next_free = free_;
        auto next_refs = references_;
        auto next_lengths = lengths_;
        auto next_prefixes = prefixes_;
        std::vector<std::pair<int, int>> copies;
        std::size_t evicted = 0;
        auto evict_one = [&]() {
            auto it = next_prefixes.begin();
            for (int block : it->second) if (--next_refs[block] == 0) next_free.push_back(block);
            next_prefixes.erase(it);++evicted;
        };
        auto allocate = [&]() {
            while (next_free.empty() && !next_prefixes.empty()) {
                // Deterministic bounded cache: evict snapshots before rejecting
                // a live allocation. Active table references remain valid.
                evict_one();
            }
            if (next_free.empty()) throw std::length_error("paged KV capacity exhausted");
            const int block = next_free.back();next_free.pop_back();next_refs[block] = 1;return block;
        };
        for (const auto& target : targets) {
            if (target.first < 1 || target.second < 1 || target.second > 4096)
                throw std::invalid_argument("invalid paged append target");
            auto& table = next_tables[target.first];
            auto& length = next_lengths[target.first];
            const auto needed = static_cast<std::size_t>((target.second + block_size_ - 1) / block_size_);
            if (target.second < length) throw std::invalid_argument("paged append cannot shrink");
            if (target.second > length && length % block_size_ && next_refs[table.back()] > 1) {
                const int old = table.back();
                while (next_free.empty() && next_refs[old] > 1 && !next_prefixes.empty()) evict_one();
                if (next_refs[old] > 1) {
                    const int replacement = allocate();
                    --next_refs[old];
                    table.back() = replacement;copies.emplace_back(old, replacement);
                }
            }
            while (table.size() < needed) {
                table.push_back(allocate());
            }
            length = target.second;
        }
        if (!commit) return;
        // All fallible metadata operations completed before touching payloads.
        for (const auto& copy : copies) {
            const auto size = std::size_t(block_size_) * dimension;
            std::memcpy(keys_.data<float>() + copy.second * size, keys_.data<float>() + copy.first * size, size * sizeof(float));
            std::memcpy(values_.data<float>() + copy.second * size, values_.data<float>() + copy.first * size, size * sizeof(float));
        }
        tables_.swap(next_tables);
        free_.swap(next_free);
        references_.swap(next_refs);lengths_.swap(next_lengths);prefixes_.swap(next_prefixes);
        cow_copies_ += copies.size();evictions_ += evicted;
        peak_ = std::max(peak_, used_blocks());
    }

    void write(int id, int position, const float* key, const float* value) {
        if (references_.at(table(id).at(position / block_size_)) != 1)
            throw std::invalid_argument("write to shared KV block without COW");
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
        for (int block : found->second) if (--references_[block] == 0) free_.push_back(block);
        tables_.erase(found);
        lengths_.erase(id);
    }

    void clear(bool reset_peak = false) {
        tables_.clear();lengths_.clear();prefixes_.clear();
        std::fill(references_.begin(), references_.end(), 0);
        free_.clear();
        for (int block = capacity_; block > 0; --block) free_.push_back(block - 1);
        if (reset_peak) { peak_ = 0;prefix_hits_ = prefix_tokens_ = cow_copies_ = evictions_ = 0; }
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
    std::map<int, int> lengths_;
    std::map<std::vector<int>, std::vector<int>> prefixes_;
    std::vector<int> references_;
    std::vector<int> free_;
    std::size_t peak_ = 0;
    std::size_t prefix_hits_ = 0, prefix_tokens_ = 0, cow_copies_ = 0, evictions_ = 0;
};

} // namespace serving
