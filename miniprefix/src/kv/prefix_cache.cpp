#include "miniprefix/prefix_cache.h"
#include <algorithm>
#include <map>
#include <iterator>
#include <utility>

namespace miniprefix {
struct ExactPrefixCache::Index {
    struct Page {
        const std::int32_t* first;
        const std::int32_t* last;
        Page(const std::vector<std::int32_t>& v):first(v.data()),last(v.data()+v.size()){}
        Page(const std::vector<std::int32_t>* v):Page(*v){}
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
        std::size_t depth=0;
        const std::vector<std::int32_t>* edge=nullptr; // stable owning map key
        using Children=std::map<std::vector<std::int32_t>,std::unique_ptr<Node>,Less>;
        Children children;
        // Non-owning keys point into children; only nodes with a cache lease.
        using Available=std::map<const std::vector<std::int32_t>*,Node*,Less>;
        Available available;
        int block=-1;
        Node* previous=nullptr;
        Node* next=nullptr;
    };
    // A shorter snapshot may refer to a page already leased by a longer one.
    std::vector<std::size_t> leases;
    explicit Index(std::size_t blocks):leases(blocks,0){}
    Node root;
    Node* oldest=nullptr;
    Node* newest=nullptr;
    std::size_t cached=0,nodes=0,key_tokens=0;
    // A non-owning hint for the last published path. Every edge is still checked
    // against the input; this is safe across requests, branches and shorter histories.
    std::vector<Node*> publication;
    // Keep only one empty leaf and one lookup allocation for snapshot churn.
    // Neither spare owns a cache lease; extracted map keys are never searched.
    Node::Children::node_type spare_node;
    Node::Available::node_type spare_available;
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
            if(n->depth<=publication.size()&&publication[n->depth-1]==n)
                publication.resize(n->depth-1);
            const auto found=parent->children.find(*n->edge);
            spare_node=parent->children.extract(found);--nodes;n=parent;
        }
    }
    void drop_node(KvBlockPool& pool,Node& node,Node* protected_node=nullptr) noexcept{
        spare_available=node.parent->available.extract(node.edge);
        --leases[node.block];unlink(node);pool.unref(node.block);node.block=-1;--cached;prune(&node,protected_node);
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
ExactPrefixCache::ExactPrefixCache(KvBlockPool& pool):pool_(pool),index_(std::make_unique<Index>(pool.config().max_blocks)){}
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
    for(std::size_t block=0;block<index_->leases.size();++block)
        if(index_->leases[block]&&pool_.references(static_cast<int>(block))==index_->leases[block])++count;
    return count;
}
void ExactPrefixCache::clear() noexcept{
    // Preserve the original lexicographic release order (and pool free-list order).
    // Precompute successors before leaf pruning; no allocation or recursive walk.
    auto* node=index_->root.children.empty()?nullptr:index_->root.children.begin()->second.get();
    while(node){auto* next=index_->successor(node);if(node->block>=0)index_->drop_node(pool_,*node);node=next;}
    index_->publication.clear();index_->spare_available={};index_->spare_node={};
}
int ExactPrefixCache::reuse(PagedSequence& sequence,const std::vector<std::int32_t>& prompt){
    if(sequence.used||!sequence.blocks.empty()||!sequence.tokens.empty())
        throw std::invalid_argument("prefix reuse requires an empty sequence");
    if(!pool_.config().prefix_cache_blocks)return 0;
    const auto block_tokens=std::size_t(pool_.config().block_tokens);
    const auto limit=prompt.empty()?0:prompt.size()-1;
    std::vector<int> pages;auto* parent=&index_->root;std::size_t count=0;
    while(count<limit){
        const auto length=std::min(block_tokens,limit-count);
        const auto* begin=prompt.data()+count;
        const Index::Page query(begin,begin+length);
        auto next=parent->available.lower_bound(query);
        Index::Node* best=nullptr;std::size_t matched=0;
        auto consider=[&](Index::Node* node){
            const auto& edge=*node->edge;std::size_t n=0;
            while(n<length&&n<edge.size()&&edge[n]==begin[n])++n;
            if(n>matched){matched=n;best=node;}
        };
        // The lexicographic neighbors contain a longest common prefix. Prefer
        // the lower_bound neighbor on ties for deterministic LRU behavior.
        if(next!=parent->available.end())consider(next->second);
        if(next!=parent->available.begin())consider(std::prev(next)->second);
        if(!matched)break;
        pages.push_back(best->block);index_->touch(*best);count+=matched;
        if(matched!=block_tokens)break;
        parent=best;
    }
    std::vector<std::int32_t> tokens(prompt.begin(),prompt.begin()+count);
    for(int page:pages)pool_.retain(page);
    sequence.blocks=std::move(pages);sequence.tokens=std::move(tokens);sequence.used=static_cast<int>(count);
    if(count){++hits_;reused_+=count;}else ++misses_;
    return sequence.used;
}
void ExactPrefixCache::publish(const PagedSequence& sequence){
    if(sequence.tokens.size()!=std::size_t(sequence.used))throw std::logic_error("invalid committed KV token history");
    const auto block_tokens=std::size_t(pool_.config().block_tokens);
    const auto count=std::min(sequence.blocks.size(),std::min((std::size_t(sequence.used)+block_tokens-1)/block_tokens,
                             std::size_t(pool_.config().prefix_cache_blocks)));
    auto* parent=&index_->root;
    index_->publication.reserve(count);
    for(std::size_t i=0;i<count;++i){
        const auto* begin=sequence.tokens.data()+i*block_tokens;
        const auto length=std::min(block_tokens,std::size_t(sequence.used)-i*block_tokens);
        Index::Node* selected=nullptr;
        if(i<index_->publication.size()){
            auto* hint=index_->publication[i];
            if(hint->parent==parent&&hint->edge->size()==length&&std::equal(begin,begin+length,hint->edge->begin()))selected=hint;
            else index_->publication.resize(i);
        }
        if(!selected){
            auto found=parent->children.find(Index::Page(begin,begin+length));
            if(found==parent->children.end()){
                if(index_->spare_node){
                    index_->spare_node.key().assign(begin,begin+length);
                    found=parent->children.insert(std::move(index_->spare_node)).position;
                }else{
                    auto node=std::make_unique<Index::Node>();
                    found=parent->children.emplace(std::vector<std::int32_t>(begin,begin+length),std::move(node)).first;
                }
                found->second->parent=parent;found->second->depth=parent->depth+1;
                found->second->edge=&found->first;++index_->nodes;index_->key_tokens+=length;
            }
            selected=found->second.get();
        }
        auto& node=*selected;
        if(node.block>=0)index_->touch(node);
        else{
            const int page=sequence.blocks[i];
            // Prepare lookup metadata before acquiring a lease or evicting a victim.
            try{
                if(index_->spare_available){
                    index_->spare_available.key()=node.edge;index_->spare_available.mapped()=&node;
                    parent->available.insert(std::move(index_->spare_available));
                }else parent->available.emplace(node.edge,&node);
                pool_.retain(page);
            }
            catch(...){index_->spare_available=parent->available.extract(node.edge);index_->prune(&node);throw;}
            if(size()==std::size_t(pool_.config().prefix_cache_blocks)){
                index_->drop(pool_,&node);++evictions_;
            }
            ++index_->leases[page];node.block=page;index_->append(node);++index_->cached;
        }
        // Eviction may have truncated the hint. Never leave gaps or stale nodes.
        if(index_->publication.size()==i)index_->publication.push_back(&node);
        parent=&node;
    }
    if(index_->publication.size()>count)index_->publication.resize(count);
}
} // namespace miniprefix
