#pragma once
#include "miniprefix/engine.h"
#include <ostream>
namespace miniprefix::protocol {
void ready(std::ostream&,int vocabulary,int eos,int context,const std::string& backend);
void ok(std::ostream&);
void count(std::ostream&,std::size_t);
void logits(std::ostream&,const std::vector<float>&);
void events(std::ostream&,const std::vector<GenerationEvent>&);
void stats(std::ostream&,const RunnerStats&);
void step_stats(std::ostream&,const EngineStepStats&);
void error(std::ostream&,const std::string&);
}
