#include "command.h"
#include <sstream>
#include <stdexcept>
namespace miniprefix::protocol {
namespace {
int integer(std::istringstream& in){int n;if(!(in>>n))throw std::invalid_argument("missing integer");return n;}
std::vector<TokenId> tokens(std::istringstream& in,int count){
    if(count<1||count>131072)throw std::invalid_argument("invalid token count");
    std::vector<TokenId> result;for(int i=0;i<count;++i)result.push_back(integer(in));return result;
}
}
Command parse(const std::string& line){
    std::istringstream in(line);std::string op;in>>op;
    if(op=="SUBMIT"){
        int id=integer(in),limit=integer(in),eos=integer(in),count=integer(in);
        if(id<1||limit<1)throw std::invalid_argument("invalid request");
        Command c{Kind::Submit};c.request.id=std::to_string(id);c.request.prompt_token_ids=tokens(in,count);
        c.request.generation.max_new_tokens=limit;
        if(eos>=0)c.request.generation.eos_token_id=eos;else if(eos!=-1)throw std::invalid_argument("invalid EOS");
        return c;
    }
    if(op=="FORWARD"||op=="REUSE"){
        Command c{op=="FORWARD"?Kind::Forward:Kind::Reuse};c.id=integer(in);int count=integer(in);c.input=tokens(in,count);
        if(c.id<1)throw std::invalid_argument("invalid diagnostic sequence");return c;
    }
    if(op=="CANCEL"||op=="CACHE"||op=="RELEASE"){
        Command c{op=="CANCEL"?Kind::Cancel:(op=="CACHE"?Kind::Cache:Kind::Release)};c.id=integer(in);return c;
    }
    if(op=="STEP")return {Kind::Step};
    if(op=="STEP_STATS")return {Kind::StepStats};
    if(op=="CLEAR_PREFIX")return {Kind::ClearPrefix};
    if(op=="STATS")return {Kind::Stats};
    if(op=="QUIT")return {Kind::Quit};
    throw std::invalid_argument("unknown command");
}
}
