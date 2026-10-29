#include "miniprefix/prefix_cache.h"
#include <algorithm>
#include <map>
#include <utility>

namespace miniprefix {
struct ExactPrefixCache::Index {
    struct Page {
        const std::int32_t* first;
        const std::int32_t* last;
        Page(const std::vector<std::int32_t>& v):first(v.data()),last(v.data()+v.size()){}
        Page(const std::int32_t* begin,const std::int32_t* end):first(begin),last(end){}
    };
    struct Less {
        using is_transparent=void;
        template<class A,class B> bool operator()(const A& a,const B& b) const{
            const Page x(a),y(b);return std::lexicographical_compare(x.first,x.last,y.first,y.last);
        }
    };
    struct Node {
        Node* parent=nullptr;
        const std::vector<std::int32_t>* edge=nullptr; // stable owning map key
        std::map<std::vector<std::int32_t>,std::unique_ptr<Node>,Less> children;
        int block=-1;
        Node* previous=nullptr;
        Node* next=nullptr;
    };
    Node root;
    Node* oldest=nullptr;
    Node* newest=nullptr;
    std::size_t cached=0,nodes=0,key_tokens=0;
    void unlink(Node& n) noexcept{
        if(n.previous)n.previous->next=n.next;else oldest=n.next;
        if(n.next)n.next->previous=n.previous;else newest=n.previous;
        n.previous=n.next=nullptr;
    }
    void append(Node& n) noexcept{
        n.previous=newest;n.next=nullptr;
        if(newest)newest->next=&n;else oldest=&n;
        newest=&n;
    }
    void touch(Node& n) noexcept{unlink(n);append(n);}
    void prune(Node* n,Node* protected_node=nullptr) noexcept{
        // Erase only empty leaves: destruction never recursively traverses a deep trie.
        while(n!=&root&&n!=protected_node&&n->block<0&&n->children.empty()){
            auto* parent=n->parent;key_tokens-=n->edge->size();
            const auto found=parent->children.find(*n->edge);
            parent->children.erase(found);--nodes;n=parent;
        }
    }
    void drop_node(KvBlockPool& pool,Node& node,Node* protected_node=nullptr) noexcept{
        unlink(node);pool.unref(node.block);node.block=-1;--cached;prune(&node,protected_node);
    }
    void drop(KvBlockPool& pool,Node* protected_node=nullptr) noexcept{drop_node(pool,*oldest,protected_node);}
    Node* successor(Node* n) noexcept{
        if(!n->children.empty())return n->children.begin()->second.get();
        while(n!=&root){
            auto* parent=n->parent;auto next=parent->children.upper_bound(*n->edge);
            if(next!=parent->children.end())return next->second.get();
            n=parent;
        }
        return nullptr;
    }
};
bool ExactPrefixCache::evict_one(){
    if(!index_->oldest)return false;
    index_->drop(pool_);++evictions_;return true;
}
ExactPrefixCache::ExactPrefixCache(KvBlockPool& pool):pool_(pool),index_(std::make_unique<Index>()){}
ExactPrefixCache::~ExactPrefixCache(){clear();}
std::size_t ExactPrefixCache::size() const{return index_->cached;}
std::size_t ExactPrefixCache::index_nodes() const{return index_->nodes;}
std::size_t ExactPrefixCache::key_token_bytes() const{return index_->key_tokens*sizeof(std::int32_t);}
std::size_t ExactPrefixCache::hits() const{return hits_;}
std::size_t ExactPrefixCache::misses() const{return misses_;}
std::size_t ExactPrefixCache::reused_tokens() const{return reused_;}
std::size_t ExactPrefixCache::evictions() const{return evictions_;}
std::size_t ExactPrefixCache::cache_only_blocks() const{
    std::size_t count=0;
    for(auto* n=index_->oldest;n;n=n->next)if(pool_.references(n->block)==1)++count;
    return count;
}
void ExactPrefixCache::clear() noexcept{
    // Preserve the original lexicographic release order (and pool free-list order).
    // Precompute successors before leaf pruning; no allocation or recursive walk.
    auto* node=index_->root.children.empty()?nullptr:index_->root.children.begin()->second.get();
    while(node){auto* next=index_->successor(node);if(node->block>=0)index_->drop_node(pool_,*node);node=next;}
}
int ExactPrefixCache::reuse(PagedSequence& sequence,const std::vector<std::int32_t>& prompt){
    if(sequence.used||!sequence.blocks.empty()||!sequence.tokens.empty())
        throw std::invalid_argument("prefix reuse requires an empty sequence");
    if(!pool_.config().prefix_cache_blocks)return 0;
    const auto block_tokens=std::size_t(pool_.config().block_tokens);
    const auto limit=prompt.empty()?0:(prompt.size()-1)/block_tokens;
    std::vector<int> pages;auto* parent=&index_->root;
    for(std::size_t i=0;i<limit;++i){
        const auto* begin=prompt.data()+i*block_tokens;
        auto found=parent->children.find(Index::Page(begin,begin+block_tokens));
        if(found==parent->children.end()||found->second->block<0)break;
        auto& node=*found->second;pages.push_back(node.block);index_->touch(node);parent=&node;
    }
    const auto count=pages.size()*block_tokens;
    std::vector<std::int32_t> tokens(prompt.begin(),prompt.begin()+count);
    for(int page:pages)pool_.retain(page);
    sequence.blocks=std::move(pages);sequence.tokens=std::move(tokens);sequence.used=static_cast<int>(count);
    if(count){++hits_;reused_+=count;}else ++misses_;
    return sequence.used;
}
void ExactPrefixCache::publish(const PagedSequence& sequence){
    if(sequence.tokens.size()!=std::size_t(sequence.used))throw std::logic_error("invalid committed KV token history");
    const auto block_tokens=std::size_t(pool_.config().block_tokens);
    const auto count=std::min(sequence.blocks.size(),std::min(std::size_t(sequence.used)/block_tokens,
                             std::size_t(pool_.config().prefix_cache_blocks)));
    auto* parent=&index_->root;
    for(std::size_t i=0;i<count;++i){
        const auto* begin=sequence.tokens.data()+i*block_tokens;
        auto found=parent->children.find(Index::Page(begin,begin+block_tokens));
        if(found==parent->children.end()){
            auto node=std::make_unique<Index::Node>();node->parent=parent;
            found=parent->children.emplace(std::vector<std::int32_t>(begin,begin+block_tokens),std::move(node)).first;
            found->second->edge=&found->first;++index_->nodes;index_->key_tokens+=block_tokens;
        }
        auto& node=*found->second;
        if(node.block>=0){index_->touch(node);parent=&node;continue;}
        // Install the path before eviction so pruning cannot delete our parent.
        if(size()==std::size_t(pool_.config().prefix_cache_blocks)){
            // A missing ancestor can be the parent of the last cached victim.
            // Keep this node alive until its replacement lease is installed.
            index_->drop(pool_,&node);++evictions_;
        }
        const int page=sequence.blocks[i];
        try{pool_.retain(page);}
        catch(...){index_->prune(&node);throw;}
        node.block=page;index_->append(node);++index_->cached;parent=&node;
    }
}
} // namespace miniprefix
