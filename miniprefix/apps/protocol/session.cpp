#include "session.h"
#include "command.h"
#include "response.h"
#include "diagnostics.h"
namespace miniprefix::protocol {
int serve(Engine& engine,std::istream& in,std::ostream& out){
    Diagnostics diagnostic(engine.runner());std::string line;
    while(std::getline(in,line)){
        try{
            auto command=parse(line);
            // Diagnostics share the runner with Engine. Once execution has
            // failed, even a direct diagnostic call must not reuse that backend.
            if(engine.failed()&&(command.kind==Kind::Forward||command.kind==Kind::Reuse||
               command.kind==Kind::Cache||command.kind==Kind::Release||command.kind==Kind::ClearPrefix||command.kind==Kind::CostTiming||command.kind==Kind::ExecutionTiming))
                throw std::runtime_error("engine failed; create a new engine");
            switch(command.kind){
            case Kind::Submit:engine.submit(std::move(command.request));ok(out);break;
            case Kind::LifecycleStats:lifecycle_stats(out,engine.lifecycle_stats());break;
            case Kind::Step:engine.step();events(out,engine.poll_events());break;
            case Kind::StepStats:step_stats(out,engine.last_step_stats());break;
            case Kind::Cancel:engine.cancel(std::to_string(command.id));events(out,engine.poll_events());break;
            case Kind::ClearPrefix:diagnostic.clear_prefix();ok(out);break;
            case Kind::Cache:diagnostic.cache(command.id);ok(out);break;
            case Kind::Reuse:count(out,diagnostic.reuse(command.id,command.input));break;
            case Kind::Forward:logits(out,diagnostic.forward(command.id,command.input));break;
            case Kind::Release:diagnostic.release(command.id);ok(out);break;
            case Kind::ExecutionTiming:engine.set_execution_timing(command.id!=0);ok(out);break;
            case Kind::ExecutionStats:execution_stats(out,engine.execution_stats());break;
            case Kind::CostStats:cost_stats(out,engine.runner().cost_stats());break;
            case Kind::PrefillStats:prefill_stats(out,engine.runner().prefill_batch_stats());break;
            case Kind::DecodeStats:decode_stats(out,engine.runner().decode_batch_stats());break;
            case Kind::CostTiming:engine.runner().set_cost_timing(command.id!=0);ok(out);break;
            case Kind::SampleBytes:count(out,engine.runner().sampled_id_downloaded_bytes());break;
            case Kind::Stats:stats(out,engine.runner().stats());break;
            case Kind::Quit:engine.shutdown();ok(out);out<<std::flush;return 0;
            }
        }catch(const std::exception& e){error(out,e.what());}
        out<<std::flush;
    }
    return 0;
}
}
