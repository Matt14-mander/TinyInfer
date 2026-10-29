#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace miniprefix {
class KvCapacityError : public std::runtime_error {
public: using std::runtime_error::runtime_error;
};
enum class KvAdmission { Ready, Wait, Impossible };

struct KvCacheConfig {
    int block_tokens=16;
    int max_blocks=128;
    int prefix_cache_blocks=0;
};

// Only complete immutable pages may be shared; a partial tail stays private.
struct PagedSequence {
    int used=0;
    std::vector<int> blocks;
    std::vector<std::int32_t> tokens;
};

// Backend-independent ownership. Storage is [physical block, layer, slot, KV].
// Reserve is atomic on capacity/host allocation failure. Callers synchronize
// outstanding device accesses before rolling back or returning any pages.
class KvBlockPool {
    KvCacheConfig config_;
    std::size_t block_elements_,live_=0,peak_=0;
    std::vector<int> free_;
    std::vector<std::size_t> references_;
public:
    KvBlockPool(KvCacheConfig config,int layers,int width);
    const KvCacheConfig& config() const;
    std::size_t block_elements() const;
    std::size_t block_bytes() const;
    std::size_t reserved_bytes() const;
    std::size_t live_blocks() const;
    std::size_t peak_blocks() const;
    std::size_t free_blocks() const;
    std::size_t references(int block) const;
    void retain(int block);
    void unref(int block) noexcept;
    void reserve(PagedSequence& sequence,int end);
    void rollback(PagedSequence& sequence,std::size_t original_blocks) noexcept;
    void release(PagedSequence& sequence) noexcept;
    std::size_t offset(const PagedSequence& sequence,int layer,int token,int layers,int width) const;
};
} // namespace miniprefix
