#include "response.h"
#include <iomanip>
namespace miniprefix::protocol {
void ready(std::ostream& out,int vocabulary,int eos,int context,const std::string& backend){out<<"READY MINIPREFIX 3 "<<vocabulary<<' '<<eos<<' '<<context<<' '<<backend<<'\n'<<std::flush;}
void ok(std::ostream& out){out<<"OK\n";}
void count(std::ostream& out,std::size_t value){out<<"OK "<<value<<'\n';}
void logits(std::ostream& out,const std::vector<float>& values){out<<"OK"<<std::setprecision(9);for(float v:values)out<<' '<<v;out<<'\n';}
void events(std::ostream& out,const std::vector<GenerationEvent>& batch){out<<"OK "<<batch.size();for(const auto& e:batch)out<<' '<<e.request_id<<' '<<(e.token_id?*e.token_id:-1)<<' '<<(e.finish_reason?static_cast<int>(*e.finish_reason):-1);out<<'\n';}
void stats(std::ostream& out,const RunnerStats& s){out<<"OK "<<s.live_sequences<<' '<<s.live_kv_bytes<<' '<<s.peak_kv_bytes<<' '<<s.forward_tokens<<' '<<s.device_weight_bytes<<' '<<s.weight_uploaded_bytes<<' '<<s.token_uploaded_bytes<<' '<<s.kv_uploaded_bytes<<' '<<s.logits_downloaded_bytes<<' '<<s.device_allocations<<' '<<s.device_workspace_bytes<<' '<<s.kv_block_tokens<<' '<<s.kv_total_blocks<<' '<<s.kv_live_blocks<<' '<<s.kv_peak_blocks<<' '<<s.kv_reserved_bytes<<' '<<s.block_table_uploaded_bytes<<' '<<s.block_table_device_bytes<<' '<<s.prefix_cache_capacity<<' '<<s.kv_cached_blocks<<' '<<s.kv_cache_only_blocks<<' '<<s.prefix_hits<<' '<<s.prefix_misses<<' '<<s.prefix_reused_tokens<<' '<<s.prefix_evictions<<'\n';}
void step_stats(std::ostream& out,const EngineStepStats& s){out<<"OK "<<s.prefix_reused_tokens<<' '<<s.prefill_tokens<<' '<<s.decode_tokens<<'\n';}
void error(std::ostream& out,const std::string& message){out<<"ERROR "<<message<<'\n';}
}
