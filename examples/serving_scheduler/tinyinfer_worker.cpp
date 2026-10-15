// Local CPU reference decoder. Random weights; no pretrained language ability.
#include "tinyinfer/tinyinfer.h"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
using namespace tinyinfer;
namespace {
constexpr int D=16,V=32,F=32,MAX_CONTEXT=4096;
std::uint32_t rng=20261015;
Tensor weights(Shape shape) {
    Tensor t(shape);for(std::size_t i=0;i<t.numel();++i){rng=rng*1664525U+1013904223U;t.at(i)=(float((rng>>8)%2001)-1000.F)/4000.F;}return t;
}
struct State {
    Tensor keys,values;int used=0,generated=0,prompt=0,budget=0,last=-1;
    State(int capacity,int p,int b):keys({capacity,D}),values({capacity,D}),prompt(p),budget(b){}
    std::size_t bytes() const {return keys.size_bytes()+values.size_bytes();}
};
class TinyDecoder {
    Tensor embedding=weights({V,D}),wq=weights({D,D}),wk=weights({D,D}),wv=weights({D,D});
    Tensor wo=weights({D,D}),w1=weights({D,F}),w2=weights({F,D}),head=weights({D,V});
public:
    std::size_t forwards=0,peak_bytes=0,peak_tokens=0;std::map<int,std::unique_ptr<State>> states;
    Tensor forward(State& state,int token) {
        if(token<0||token>=V||state.used>=state.keys.shape()[0])throw std::invalid_argument("invalid context/token");
        Tensor x({1,D});for(int d=0;d<D;++d)x.at(d)=embedding.at(std::size_t(token)*D+d)+0.1F*std::sin(float(state.used+1)*float(d+1)/17.F);
        auto q=ops::matmul(x,wq),k=ops::matmul(x,wk),v=ops::matmul(x,wv);
        for(int d=0;d<D;++d){state.keys.at(std::size_t(state.used)*D+d)=k.at(d);state.values.at(std::size_t(state.used)*D+d)=v.at(d);}
        ++state.used;++forwards;
        auto keys=state.keys.narrow(0,0,state.used);keys.transpose(0,1);
        auto scores=ops::matmul(q,keys);for(std::size_t i=0;i<scores.numel();++i)scores.at(i)/=std::sqrt(float(D));
        auto probabilities=ops::softmax(scores,-1);
        auto context=ops::matmul(probabilities,state.values.narrow(0,0,state.used));
        auto residual=ops::add(x,ops::matmul(context,wo));
        auto ff=ops::matmul(ops::relu(ops::matmul(ops::layer_norm(residual),w1)),w2);
        return ops::matmul(ops::layer_norm(ops::add(residual,ff)),head);
    }
    static int greedy(const Tensor& logits){int best=0;for(int i=0;i<V;++i){if(!std::isfinite(logits.at(i)))throw std::runtime_error("nonfinite logits");if(logits.at(i)>logits.at(best))best=i;}return best;}
    void observe(){std::size_t bytes=0,tokens=0;for(const auto& p:states){bytes+=p.second->bytes();tokens+=p.second->used;}peak_bytes=std::max(peak_bytes,bytes);peak_tokens=std::max(peak_tokens,tokens);}
    int step(int id,int generated,int start,int count,int prompt,int budget,int capacity,const std::vector<int>& input) {
        if(id<1||generated<0||start<0||prompt<1||prompt>MAX_CONTEXT||budget<1||budget>MAX_CONTEXT||prompt>MAX_CONTEXT-budget||capacity<prompt+budget||capacity>MAX_CONTEXT||int(input.size())!=count)throw std::invalid_argument("context limits");
        auto found=states.find(id);
        if(found==states.end()) {
            if(generated||start)throw std::invalid_argument("unknown request state");
            found=states.emplace(id,std::make_unique<State>(capacity,prompt,budget)).first;
        }
        auto& s=*found->second;
        if(s.generated!=generated||s.prompt!=prompt||s.budget!=budget||generated>=budget)throw std::invalid_argument("generation state mismatch");
        if(generated==0) {
            if(start!=s.used||count<1||start+count>prompt)throw std::invalid_argument("prefill progress mismatch");
        } else if(count!=1||input[0]!=s.last||s.used!=prompt+generated-1)throw std::invalid_argument("decode input mismatch");
        Tensor logits;
        for(int token:input)logits=forward(s,token);
        observe();
        if(generated==0 && s.used<prompt)return -1;
        s.last=greedy(logits);++s.generated;return s.last;
    }
    // Independent no-KV-reuse oracle: rebuild each full prefix for every token.
    std::vector<int> reference(std::vector<int> prefix,int outputs) {
        if(prefix.empty()||outputs<1||prefix.size()+outputs>MAX_CONTEXT)throw std::invalid_argument("reference limits");
        std::vector<int> result;
        for(int i=0;i<outputs;++i){State fresh(int(prefix.size())+1,int(prefix.size()),1);Tensor logits;for(int t:prefix)logits=forward(fresh,t);auto next=greedy(logits);result.push_back(next);prefix.push_back(next);}return result;
    }
};
int integer(std::istringstream& in){int v;if(!(in>>v))throw std::invalid_argument("missing integer");return v;}
}
int main(){
    TinyDecoder model;std::cout<<"READY 16 32 4096\n"<<std::flush;std::string line;
    while(std::getline(std::cin,line))try{
        std::istringstream in(line);std::string op;in>>op;
        if(op=="STEP"){
            int n=integer(in);if(n<1||n>1024)throw std::invalid_argument("batch size");std::vector<int> result;
            for(int i=0;i<n;++i){int id=integer(in),g=integer(in),s=integer(in),c=integer(in),p=integer(in),b=integer(in),cap=integer(in);if(c<1||c>MAX_CONTEXT)throw std::invalid_argument("input count");std::vector<int> tokens;for(int j=0;j<c;++j)tokens.push_back(integer(in));result.push_back(model.step(id,g,s,c,p,b,cap,tokens));}
            std::cout<<"OK";for(int t:result)std::cout<<' '<<t;std::cout<<'\n';
        }else if(op=="REFERENCE"){
            int p=integer(in),b=integer(in);if(p<1||p>MAX_CONTEXT)throw std::invalid_argument("prompt size");std::vector<int> tokens;for(int i=0;i<p;++i)tokens.push_back(integer(in));auto result=model.reference(tokens,b);std::cout<<"OK";for(int t:result)std::cout<<' '<<t;std::cout<<'\n';
        }else if(op=="RELEASE"){if(model.states.erase(integer(in))!=1)throw std::invalid_argument("unknown release");std::cout<<"OK\n";
        }else if(op=="RESET"){model.states.clear();model.forwards=model.peak_bytes=model.peak_tokens=0;std::cout<<"OK\n";
        }else if(op=="CLEAR"){model.states.clear();std::cout<<"OK\n";
        }else if(op=="STATS"){std::size_t bytes=0;for(const auto& p:model.states)bytes+=p.second->bytes();std::cout<<"OK "<<model.states.size()<<' '<<bytes<<' '<<model.peak_bytes<<' '<<model.peak_tokens<<' '<<model.forwards<<'\n';
        }else if(op=="QUIT"){std::cout<<"OK\n"<<std::flush;return 0;
        }else throw std::invalid_argument("unknown command");
        std::cout<<std::flush;
    }catch(const std::exception& e){model.states.clear();std::cout<<"ERROR "<<e.what()<<'\n'<<std::flush;}
}
