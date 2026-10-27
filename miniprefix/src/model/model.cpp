#include "miniprefix/model.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <set>

namespace miniprefix {
void ModelConfig::validate() const {
    if(layers<1||layers>128||hidden<2||hidden>8192||intermediate<1||intermediate>65536||heads<1||kv_heads<1||
       hidden%heads||heads%kv_heads||(hidden/heads)%2||hidden/heads>256||vocabulary<2||vocabulary>262144||
       context<1||context>131072||eos<0||eos>=vocabulary||!std::isfinite(rms_epsilon)||rms_epsilon<=0||
       !std::isfinite(rope_theta)||rope_theta<=1)
        throw std::invalid_argument("unsupported/invalid Llama configuration");
}
const tinyinfer::Tensor& ModelWeights::get(const std::string& name) const { return tensors.at(name); }
namespace {
template<class T>T read(std::ifstream& file){T value{};if(!file.read(reinterpret_cast<char*>(&value),sizeof(T)))throw std::runtime_error("truncated model");return value;}
}
std::shared_ptr<ModelWeights> ModelWeights::load(const std::string& path) {
    std::ifstream file(path,std::ios::binary);if(!file)throw std::runtime_error("cannot open model: "+path);
    char magic[8];if(!file.read(magic,8)||std::memcmp(magic,"MPFX0001",8))throw std::runtime_error("invalid model format");
    auto model=std::make_shared<ModelWeights>();auto& c=model->config;
    int* fields[]={&c.layers,&c.hidden,&c.intermediate,&c.heads,&c.kv_heads,&c.vocabulary,&c.context,&c.eos};
    for(auto* field:fields)*field=read<std::int32_t>(file);
    c.rms_epsilon=read<float>(file);c.rope_theta=read<float>(file);
    auto tied=read<std::uint32_t>(file);if(tied>1)throw std::runtime_error("invalid tied flag");c.tied=tied!=0;c.validate();
    std::map<std::string,tinyinfer::Shape> expected{{"model.embed_tokens.weight",{c.vocabulary,c.hidden}},{"model.norm.weight",{c.hidden}}};
    if(!c.tied)expected["lm_head.weight"]={c.hidden,c.vocabulary};
    for(int layer=0;layer<c.layers;++layer){
        auto p="model.layers."+std::to_string(layer)+".";
        expected[p+"input_layernorm.weight"]={c.hidden};expected[p+"post_attention_layernorm.weight"]={c.hidden};
        expected[p+"self_attn.q_proj.weight"]={c.hidden,c.hidden};
        expected[p+"self_attn.k_proj.weight"]={c.hidden,c.kv_width()};expected[p+"self_attn.v_proj.weight"]={c.hidden,c.kv_width()};
        expected[p+"self_attn.o_proj.weight"]={c.hidden,c.hidden};
        expected[p+"mlp.gate_proj.weight"]={c.hidden,c.intermediate};expected[p+"mlp.up_proj.weight"]={c.hidden,c.intermediate};
        expected[p+"mlp.down_proj.weight"]={c.intermediate,c.hidden};
    }
    const auto count=read<std::uint32_t>(file);if(count!=expected.size())throw std::runtime_error("model tensor count mismatch");
    const auto file_bytes=std::filesystem::file_size(path);
    for(std::uint32_t i=0;i<count;++i){
        auto length=read<std::uint32_t>(file);if(length<1||length>256)throw std::runtime_error("invalid tensor name");
        std::string name(length,'\0');if(!file.read(name.data(),length))throw std::runtime_error("truncated name");
        auto found=expected.find(name);if(found==expected.end()||model->tensors.count(name))throw std::runtime_error("unknown/duplicate tensor: "+name);
        auto rank=read<std::uint32_t>(file);if(rank!=found->second.size())throw std::runtime_error("tensor rank mismatch");
        tinyinfer::Shape shape;std::uint64_t elements=1;
        for(std::uint32_t d=0;d<rank;++d){auto dim=read<std::int32_t>(file);if(dim!=found->second[d])throw std::runtime_error("tensor shape mismatch: "+name);shape.push_back(dim);elements*=dim;}
        auto bytes=read<std::uint64_t>(file);const auto offset=static_cast<std::uint64_t>(file.tellg());
        if(bytes!=elements*sizeof(float)||offset>file_bytes||bytes>file_bytes-offset)throw std::runtime_error("invalid tensor bytes");
        tinyinfer::Tensor tensor(shape);if(!file.read(reinterpret_cast<char*>(tensor.data<float>()),static_cast<std::streamsize>(bytes)))throw std::runtime_error("truncated tensor");
        for(std::size_t j=0;j<tensor.numel();++j)if(!std::isfinite(tensor.at(j)))throw std::runtime_error("nonfinite weights");
        model->tensors.emplace(name,std::move(tensor));
    }
    if(file.peek()!=std::char_traits<char>::eof())throw std::runtime_error("trailing model data");
    return model;
}
void RunnerStats::update_kv_stats(const KvStats& stats){
    live_sequences=stats.live_sequences;live_kv_bytes=stats.live_kv_bytes;
    peak_kv_bytes=stats.peak_kv_bytes;forward_tokens=stats.forward_tokens;
    kv_block_tokens=stats.kv_block_tokens;kv_total_blocks=stats.kv_total_blocks;
    kv_live_blocks=stats.kv_live_blocks;kv_peak_blocks=stats.kv_peak_blocks;kv_reserved_bytes=stats.kv_reserved_bytes;
    prefix_cache_capacity=stats.prefix_cache_capacity;kv_cached_blocks=stats.kv_cached_blocks;
    kv_cache_only_blocks=stats.kv_cache_only_blocks;prefix_hits=stats.prefix_hits;prefix_misses=stats.prefix_misses;
    prefix_reused_tokens=stats.prefix_reused_tokens;prefix_evictions=stats.prefix_evictions;
}
} // namespace miniprefix
