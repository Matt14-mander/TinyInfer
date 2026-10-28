#include "miniprefix/engine.h"
#include "protocol/session.h"
#include "protocol/response.h"
#include <iostream>
using namespace miniprefix;
int main(int argc,char** argv){try{
    if(argc<4||argc>10)throw std::invalid_argument("usage: miniprefix_worker MODEL cpu|cuda CONTEXT [BATCH TOKENS CHUNK BLOCK_TOKENS KV_BLOCKS PREFIX_BLOCKS]");
    auto weights=ModelWeights::load(argv[1]);const int context=std::stoi(argv[3]);std::unique_ptr<ModelRunner> runner;
    KvCacheConfig kv;if(argc>7)kv.block_tokens=std::stoi(argv[7]);if(argc>8)kv.max_blocks=std::stoi(argv[8]);
    if(argc>9)kv.prefix_cache_blocks=std::stoi(argv[9]);
    const std::string backend=argv[2];
    if(backend=="cpu")runner=make_cpu_runner(weights,context,kv);
#ifdef MINIPREFIX_ENABLE_CUDA
    else if(backend=="cuda")runner=make_cuda_runner(weights,context,kv);
#endif
    else throw std::invalid_argument("backend unavailable in this build");
    EngineConfig config;config.context=context;if(argc>4)config.max_batch_size=std::stoi(argv[4]);if(argc>5)config.token_budget=std::stoi(argv[5]);if(argc>6)config.prefill_chunk=std::stoi(argv[6]);
    Engine engine(std::move(runner),config);
    protocol::ready(std::cout,weights->config.vocabulary,weights->config.eos,context,backend);
    return protocol::serve(engine,std::cin,std::cout);
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
