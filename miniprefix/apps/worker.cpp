#include "miniprefix/engine.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <limits>
using namespace miniprefix;
namespace {
int integer(std::istringstream& in){int n;if(!(in>>n))throw std::invalid_argument("missing integer");return n;}
std::vector<TokenId> tokens(std::istringstream& in,int count){if(count<1||count>131072)throw std::invalid_argument("invalid token count");std::vector<TokenId> result;for(int i=0;i<count;++i)result.push_back(integer(in));return result;}
void events(Engine& engine){auto batch=engine.poll_events();std::cout<<"OK "<<batch.size();for(auto& e:batch)std::cout<<' '<<e.request_id<<' '<<(e.token_id?*e.token_id:-1)<<' '<<(e.finish_reason?static_cast<int>(*e.finish_reason):-1);std::cout<<'\n';}
}
int main(int argc,char** argv){try{
    if(argc<4||argc>9)throw std::invalid_argument("usage: miniprefix_worker MODEL cpu|cuda CONTEXT [BATCH TOKENS CHUNK BLOCK_TOKENS KV_BLOCKS]");
    auto weights=ModelWeights::load(argv[1]);const int context=std::stoi(argv[3]);std::unique_ptr<ModelRunner> runner;
    KvCacheConfig kv;if(argc>7)kv.block_tokens=std::stoi(argv[7]);if(argc>8)kv.max_blocks=std::stoi(argv[8]);
    const std::string backend=argv[2];
    if(backend=="cpu")runner=make_cpu_runner(weights,context,kv);
#ifdef MINIPREFIX_ENABLE_CUDA
    else if(backend=="cuda")runner=make_cuda_runner(weights,context,kv);
#endif
    else throw std::invalid_argument("backend unavailable in this build");
    EngineConfig config;config.context=context;if(argc>4)config.max_batch_size=std::stoi(argv[4]);if(argc>5)config.token_budget=std::stoi(argv[5]);if(argc>6)config.prefill_chunk=std::stoi(argv[6]);
    Engine engine(std::move(runner),config);
    std::cout<<"READY MINIPREFIX 2 "<<weights->config.vocabulary<<' '<<weights->config.eos<<' '<<context<<' '<<backend<<'\n'<<std::flush;
    std::set<std::string> created;
    std::string line;while(std::getline(std::cin,line)){try{std::istringstream in(line);std::string op;in>>op;
        if(op=="SUBMIT"){int id=integer(in),limit=integer(in),eos=integer(in),count=integer(in);if(id<1||limit<1)throw std::invalid_argument("invalid request");
            Request r;r.id=std::to_string(id);r.prompt_token_ids=tokens(in,count);r.generation.max_new_tokens=limit;if(eos>=0)r.generation.eos_token_id=eos;else if(eos!=-1)throw std::invalid_argument("invalid EOS");engine.submit(std::move(r));std::cout<<"OK\n";
        }else if(op=="STEP"){engine.step();events(engine);
        }else if(op=="CANCEL"){auto id=integer(in);engine.cancel(std::to_string(id));events(engine);
        }else if(op=="FORWARD"){int id=integer(in),count=integer(in);auto input=tokens(in,count);auto key=std::to_string(id);
            // Diagnostic IDs use a separate namespace from Engine request IDs.
            key="reference/"+key;auto& r=engine.runner();
            if(id<1)throw std::invalid_argument("invalid diagnostic sequence");
            if(!created.count(key)){r.create(key);created.insert(key);}
            auto logits=r.append(key,input);std::cout<<"OK"<<std::setprecision(9);for(float v:logits)std::cout<<' '<<v;std::cout<<'\n';
        }else if(op=="RELEASE"){auto id=integer(in);auto key="reference/"+std::to_string(id);engine.runner().release(key);created.erase(key);std::cout<<"OK\n";
        }else if(op=="STATS"){auto s=engine.runner().stats();std::cout<<"OK "<<s.live_sequences<<' '<<s.live_kv_bytes<<' '<<s.peak_kv_bytes<<' '<<s.forward_tokens<<' '<<s.device_weight_bytes<<' '<<s.weight_uploaded_bytes<<' '<<s.token_uploaded_bytes<<' '<<s.kv_uploaded_bytes<<' '<<s.logits_downloaded_bytes<<' '<<s.device_allocations<<' '<<s.device_workspace_bytes<<' '<<s.kv_block_tokens<<' '<<s.kv_total_blocks<<' '<<s.kv_live_blocks<<' '<<s.kv_peak_blocks<<' '<<s.kv_reserved_bytes<<' '<<s.block_table_uploaded_bytes<<' '<<s.block_table_device_bytes<<'\n';
        }else if(op=="QUIT"){engine.shutdown();std::cout<<"OK\n"<<std::flush;return 0;
        }else throw std::invalid_argument("unknown command");
    }catch(const std::exception& e){std::cout<<"ERROR "<<e.what()<<'\n';}std::cout<<std::flush;}
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
