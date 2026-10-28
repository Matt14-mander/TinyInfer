#include "protocol/command.h"
#include "protocol/response.h"
#include "protocol/session.h"
#include <cassert>
#include <sstream>
#include <iostream>
using namespace miniprefix;
namespace {
void parsing(){
    const std::vector<std::pair<std::string,std::string>> errors={
        {"","unknown command"},{"step","unknown command"},{"UNKNOWN 1","unknown command"},
        {"CANCEL","missing integer"},{"CANCEL 2147483648","missing integer"},
        {"SUBMIT 0 0 -2 0","invalid request"},{"SUBMIT 0 0 -2","missing integer"},
        {"SUBMIT 1 1 -2 0","invalid token count"},{"SUBMIT 1 1 -2 1","missing integer"},
        {"SUBMIT 1 1 -2 1 0","invalid EOS"},{"FORWARD 0 0","invalid token count"},
        {"REUSE 0 1","missing integer"},{"REUSE 0 1 0","invalid diagnostic sequence"},
        {"FORWARD 1 131073","invalid token count"},{"FORWARD 1 -1","invalid token count"}};
    for(const auto& item:errors){bool caught=false;try{protocol::parse(item.first);}
        catch(const std::invalid_argument& e){caught=true;assert(e.what()==item.second);}assert(caught);}
    auto c=protocol::parse(" \tSUBMIT +1 2 -1 2 0 3 trailing ignored\r");
    assert(c.kind==protocol::Kind::Submit&&c.request.id=="1"&&c.request.generation.max_new_tokens==2);
    assert(!c.request.generation.eos_token_id&&(c.request.prompt_token_ids==std::vector<TokenId>{0,3}));
    assert(protocol::parse("CACHE -2").id==-2&&protocol::parse("RELEASE 0").id==0);
    assert(protocol::parse("STEP junk").kind==protocol::Kind::Step);
    assert(protocol::parse("CANCEL 1x").id==1); // formatted integer extraction, not strict token parsing
}
void encoding(){
    std::ostringstream out;protocol::ready(out,8,2,16,"cuda");protocol::ok(out);protocol::count(out,4);
    protocol::logits(out,{0.125F,1.23456789F,-0.0F});
    protocol::events(out,{{"1",3,std::nullopt,{}},{"2",std::nullopt,FinishReason::Error,"not on wire"}});
    protocol::events(out,{});protocol::step_stats(out,{4,3,2});
    protocol::stats(out,RunnerStats{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25});
    protocol::error(out,"bad input");
    assert(out.str()=="READY MINIPREFIX 3 8 2 16 cuda\nOK\nOK 4\nOK 0.125 1.23456788 -0\n"
        "OK 2 1 3 -1 2 -1 3\nOK 0\nOK 4 3 2\n"
        "OK 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25\nERROR bad input\n");
}
class Runner final:public ModelRunner {
    ModelConfig config_;
public:
    std::map<RequestId,int> positions,creates;
    int clears=0,flushes=0;
    Runner(){config_.context=16;config_.vocabulary=8;}
    const ModelConfig& config() const override{return config_;}
    void create(const RequestId& id) override{if(positions.count(id))throw std::runtime_error("duplicate sequence");positions[id]=0;++creates[id];}
    std::vector<float> append(const RequestId& id,const std::vector<TokenId>& tokens) override{
        if(!positions.count(id))throw std::runtime_error("unknown sequence");
        for(auto token:tokens)if(token<0||token>=8)throw std::runtime_error("invalid token");
        positions[id]+=int(tokens.size());std::vector<float> result(8,0);result[2]=1;return result;
    }
    std::size_t reuse_prefix(const RequestId& id,const std::vector<TokenId>&) override{
        if(positions.at(id))throw std::runtime_error("nonempty sequence");return 0;
    }
    void cache_prefix(const RequestId& id) override{if(!positions.count(id))throw std::runtime_error("unknown sequence");}
    void clear_prefix_cache() override{++flushes;}
    void release(const RequestId& id) override{if(!positions.erase(id))throw std::runtime_error("unknown sequence");}
    void clear() override{positions.clear();++clears;}
    RunnerStats stats() const override{RunnerStats s;s.live_sequences=positions.size();s.kv_block_tokens=4;return s;}
};
struct Buffer:std::stringbuf {int flushes=0;int sync() override{++flushes;return 0;}};
void transcript(){
    auto runner=std::make_unique<Runner>();auto* trace=runner.get();Engine engine(std::move(runner),{1,2,2,16});
    std::istringstream in("FORWARD 1 1 1\nSUBMIT 1 1 -1 1 1\nCACHE 1\nREUSE 2 2 1 1\nSTEP\nSTEP_STATS\n"
        "FORWARD 1 1 8\nFORWARD 1 1 1\nRELEASE 1\nRELEASE 1\nCACHE -2\nCANCEL 0\nCLEAR_PREFIX\nQUIT\nUNKNOWN\n");
    Buffer buffer;std::ostream out(&buffer);assert(protocol::serve(engine,in,out)==0);
    assert(buffer.str()=="OK 0 0 1 0 0 0 0 0\nOK\nOK\nOK 0\nOK 1 1 2 1\nOK 0 1 0\n"
        "ERROR invalid token\nOK 0 0 1 0 0 0 0 0\nOK\nERROR unknown sequence\nERROR unknown sequence\n"
        "ERROR request is unknown or already terminal\nOK\nOK\n");
    assert(buffer.flushes==14&&trace->creates["reference/1"]==1&&trace->creates["1"]==1);
    assert(trace->creates["reference/2"]==1&&trace->flushes==1&&trace->clears==1);
    std::string remaining;std::getline(in,remaining);assert(remaining=="UNKNOWN");
}
void recovery_and_eof(){
    auto runner=std::make_unique<Runner>();auto* trace=runner.get();Engine engine(std::move(runner),{1,1,1,16});
    std::istringstream in("\nSUBMIT 1 2 -1 2 1\nSUBMIT 1 2 -1 2 1 1\nSUBMIT 2 1 -1 1 1\nSTEP\nCANCEL 2\nCANCEL 1\nSTEP\n"
        "FORWARD 7 1 8\nFORWARD 7 1 1\nREUSE 7 1 1\nRELEASE 7\nFORWARD 7 1 1");
    std::ostringstream out;protocol::serve(engine,in,out);
    assert(out.str()=="ERROR unknown command\nERROR missing integer\nOK\nOK\nOK 0\nOK 1 2 -1 2\nOK 1 1 -1 2\nOK 0\n"
        "ERROR invalid token\nOK 0 0 1 0 0 0 0 0\nERROR nonempty sequence\nOK\nOK 0 0 1 0 0 0 0 0\n");
    assert(trace->creates["reference/7"]==2&&trace->positions.count("reference/7"));
    assert(trace->clears==0); // EOF itself does not perform QUIT; Engine destructor owns cleanup.
}
}
int main(){parsing();encoding();transcript();recovery_and_eof();std::cout<<"worker protocol parsing, encoding and state traces passed\n";}
