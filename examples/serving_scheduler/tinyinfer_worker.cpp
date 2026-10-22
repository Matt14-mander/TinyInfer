// Local CPU reference decoder. Random weights; no pretrained language ability.
#include "tinyinfer/tinyinfer.h"
#include "native/paged_kv_cache.h"
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <set>
using namespace tinyinfer;
namespace {
constexpr int D=16,V=32,F=32,MAX_CONTEXT=4096;
std::uint32_t rng=20261015;
Tensor weights(Shape shape) {
    Tensor t(shape);for(std::size_t i=0;i<t.numel();++i){rng=rng*1664525U+1013904223U;t.at(i)=(float((rng>>8)%2001)-1000.F)/4000.F;}return t;
}
struct State {
    Tensor keys,values;int used=0,generated=0,prompt=0,budget=0,last=-1,capacity=0,id=0;
    State(int cap,int p,int b,bool paged=false)
        :keys(paged?Tensor():Tensor({cap,D})),values(paged?Tensor():Tensor({cap,D})),prompt(p),budget(b),capacity(cap){}
    std::size_t bytes() const {return keys.size_bytes()+values.size_bytes();}
};
struct BatchItem {
    int id, generated, start, count, prompt, budget, capacity;
    std::vector<int> input;
};
class TinyDecoder {
    Tensor embedding=weights({V,D}),wq=weights({D,D}),wk=weights({D,D}),wv=weights({D,D});
    Tensor wo=weights({D,D}),w1=weights({D,F}),w2=weights({F,D}),head=weights({D,V});
public:
    std::size_t forwards=0,peak_bytes=0,peak_tokens=0;std::map<int,std::unique_ptr<State>> states;
    std::size_t batch_steps=0,linear_calls=0,linear_rows=0,peak_batch_requests=0,peak_batch_tokens=0;
    std::unique_ptr<serving::PagedKVCache> pool;
    bool gather_attention=false;
    std::size_t paged_queries=0,gather_queries=0;
    std::vector<float> last_logits;
    Tensor forward(State& state,int token) {
        if(token<0||token>=V||state.used>=state.keys.shape()[0])throw std::invalid_argument("invalid context/token");
        Tensor x({1,D});for(int d=0;d<D;++d)x.at(d)=embedding.at(std::size_t(token)*D+d)+0.1F*std::sin(float(state.used+1)*float(d+1)/17.F);
        auto q=ops::matmul(x,wq),k=ops::matmul(x,wk),v=ops::matmul(x,wv);
        for(int d=0;d<D;++d){state.keys.at(std::size_t(state.used)*D+d)=k.at(d);state.values.at(std::size_t(state.used)*D+d)=v.at(d);}
        ++state.used; // Serial reference work is excluded from serving counters.
        auto keys=state.keys.narrow(0,0,state.used);keys.transpose(0,1);
        auto scores=ops::matmul(q,keys);for(std::size_t i=0;i<scores.numel();++i)scores.at(i)/=std::sqrt(float(D));
        auto probabilities=ops::softmax(scores,-1);
        auto context=ops::matmul(probabilities,state.values.narrow(0,0,state.used));
        auto residual=ops::add(x,ops::matmul(context,wo));
        auto ff=ops::matmul(ops::relu(ops::matmul(ops::layer_norm(residual),w1)),w2);
        return ops::matmul(ops::layer_norm(ops::add(residual,ff)),head);
    }
    static int greedy(const Tensor& logits){int best=0;for(int i=0;i<V;++i){if(!std::isfinite(logits.at(i)))throw std::runtime_error("nonfinite logits");if(logits.at(i)>logits.at(best))best=i;}return best;}
    std::size_t live_bytes() const {
        if(pool)return pool->live_bytes();
        std::size_t bytes=0;for(const auto& p:states)bytes+=p.second->bytes();return bytes;
    }
    void observe(){std::size_t tokens=0;for(const auto& p:states)tokens+=p.second->used;peak_bytes=std::max(peak_bytes,live_bytes());peak_tokens=std::max(peak_tokens,tokens);}
    void clear(bool reset=false) {
        states.clear();last_logits.clear();if(pool)pool->clear(reset);
        if(reset){forwards=peak_bytes=peak_tokens=0;batch_steps=linear_calls=linear_rows=peak_batch_requests=peak_batch_tokens=0;paged_queries=gather_queries=0;}
    }
    void release(int id) {
        if(states.find(id)==states.end())throw std::invalid_argument("unknown release");
        if(pool)pool->release(id);
        states.erase(id);
    }
    State& prepare(const BatchItem& item) {
        const auto& input=item.input;
        const auto id=item.id,generated=item.generated,start=item.start,count=item.count,
                   prompt=item.prompt,budget=item.budget,capacity=item.capacity;
        if(id<1||generated<0||start<0||prompt<1||prompt>MAX_CONTEXT||start>prompt||budget<1||budget>MAX_CONTEXT||prompt>MAX_CONTEXT-budget||capacity<prompt+budget||capacity>MAX_CONTEXT||int(input.size())!=count)throw std::invalid_argument("context limits");
        auto found=states.find(id);
        if(found==states.end()) {
            if(generated||start)throw std::invalid_argument("unknown request state");
            found=states.emplace(id,std::make_unique<State>(capacity,prompt,budget,bool(pool))).first;
            found->second->id=id;
        }
        auto& s=*found->second;
        if(s.generated!=generated||s.prompt!=prompt||s.budget!=budget||generated>=budget||s.capacity!=capacity)throw std::invalid_argument("generation state mismatch");
        if(pool&&capacity%pool->block_size()!=0)throw std::invalid_argument("paged capacity must align to block size");
        if(generated==0) {
            if(start!=s.used||count<1||start+count>prompt)throw std::invalid_argument("prefill progress mismatch");
        } else if(count!=1||input[0]!=s.last||s.used!=prompt+generated-1)throw std::invalid_argument("decode input mismatch");
        if(count<1||s.used+count>capacity)throw std::invalid_argument("context overflow");
        for(int token:input)if(token<0||token>=V)throw std::invalid_argument("invalid token");
        return s;
    }
    Tensor linear(const Tensor& x,const Tensor& weight) {
        auto result=ops::matmul(x,weight);
        ++linear_calls;linear_rows+=std::size_t(x.shape()[0]);return result;
    }
    std::vector<int> step(const std::vector<BatchItem>& batch) {
        // Pack all real input tokens, with no padding. Each row retains its
        // request and absolute KV position; item order is also output order.
        std::set<int> ids;
        std::vector<State*> owners;
        std::vector<int> positions;
        std::vector<std::size_t> ends;
        std::vector<std::pair<int,int>> targets;
        for(const auto& item:batch) {
            if(!ids.insert(item.id).second)throw std::invalid_argument("duplicate batch request");
            auto& state=prepare(item);
            targets.emplace_back(item.id,state.used+item.count);
            for(int i=0;i<item.count;++i){owners.push_back(&state);positions.push_back(state.used+i);}
            ends.push_back(owners.size());
        }
        if(pool)pool->prepare_append(targets);
        const auto rows=static_cast<std::int64_t>(owners.size());
        Tensor x({rows,D});std::size_t row=0;
        for(const auto& item:batch)for(int token:item.input) {
            for(int d=0;d<D;++d)x.at(row*D+d)=embedding.at(std::size_t(token)*D+d)+0.1F*std::sin(float(positions[row]+1)*float(d+1)/17.F);
            ++row;
        }
        auto q=linear(x,wq),k=linear(x,wk),v=linear(x,wv);
        for(row=0;row<owners.size();++row) {
            auto& s=*owners[row];
            if(pool)pool->write(s.id,positions[row],k.data<float>()+row*D,v.data<float>()+row*D);
            else for(int d=0;d<D;++d){s.keys.at(std::size_t(positions[row])*D+d)=k.at(row*D+d);s.values.at(std::size_t(positions[row])*D+d)=v.at(row*D+d);}
        }
        Tensor contexts({rows,D});
        // Ragged attention stays per query. Limit every query to its own
        // causal prefix, even though later rows' K/V have already been written.
        for(row=0;row<owners.size();++row) {
            auto& s=*owners[row];const int length=positions[row]+1;
            if(pool) {
                auto query=q.narrow(0,static_cast<std::int64_t>(row),1);
                auto context=gather_attention?pool->gather_attention(s.id,length,query):pool->attention(s.id,length,query);
                for(int d=0;d<D;++d)contexts.at(row*D+d)=context.at(d);
                continue;
            }
            auto keys=s.keys.narrow(0,0,length);keys.transpose(0,1);
            auto scores=ops::matmul(q.narrow(0,static_cast<std::int64_t>(row),1),keys);
            for(std::size_t j=0;j<scores.numel();++j)scores.at(j)/=std::sqrt(float(D));
            auto context=ops::matmul(ops::softmax(scores,-1),s.values.narrow(0,0,length));
            for(int d=0;d<D;++d)contexts.at(row*D+d)=context.at(d);
        }
        auto residual=ops::add(x,linear(contexts,wo));
        auto ff=linear(ops::relu(linear(ops::layer_norm(residual),w1)),w2);
        auto logits=linear(ops::layer_norm(ops::add(residual,ff)),head);
        std::vector<int> result;last_logits.clear();
        for(std::size_t i=0;i<batch.size();++i) {
            auto& s=*owners[ends[i]-1];s.used+=batch[i].count;
            const auto last=logits.narrow(0,static_cast<std::int64_t>(ends[i]-1),1);
            for(int j=0;j<V;++j)last_logits.push_back(last.at(j));
            if(batch[i].generated==0&&s.used<s.prompt)result.push_back(-1);
            else{s.last=greedy(last);++s.generated;result.push_back(s.last);}
        }
        forwards+=owners.size();++batch_steps;
        if(pool){if(gather_attention)gather_queries+=owners.size();else paged_queries+=owners.size();}
        peak_batch_requests=std::max(peak_batch_requests,batch.size());
        peak_batch_tokens=std::max(peak_batch_tokens,owners.size());observe();
        return result;
    }
    Tensor reference_logits(const std::vector<int>& prefix) {
        if(prefix.empty()||prefix.size()>MAX_CONTEXT)throw std::invalid_argument("reference limits");
        State fresh(int(prefix.size()),int(prefix.size()),1);Tensor logits;
        for(int token:prefix)logits=forward(fresh,token);
        return logits;
    }
    // Independent no-KV-reuse oracle: rebuild each full prefix for every token.
    std::vector<int> reference(std::vector<int> prefix,int outputs) {
        if(prefix.empty()||outputs<1||prefix.size()+outputs>MAX_CONTEXT)throw std::invalid_argument("reference limits");
        std::vector<int> result;
        for(int i=0;i<outputs;++i){auto next=greedy(reference_logits(prefix));result.push_back(next);prefix.push_back(next);}return result;
    }
};
int integer(std::istringstream& in){int v;if(!(in>>v))throw std::invalid_argument("missing integer");return v;}
}
int main(){
    TinyDecoder model;std::cout<<"READY 16 32 4096 BATCHED 1 PAGED 1\n"<<std::flush;std::string line;
    while(std::getline(std::cin,line))try{
        std::istringstream in(line);std::string op;in>>op;
        if(op=="CONFIG_PAGED"){
            int blocks=integer(in),size=integer(in),gather=integer(in);
            if(model.pool||!model.states.empty()||model.forwards|| (gather!=0&&gather!=1))throw std::invalid_argument("invalid paged configuration");
            model.pool=std::make_unique<serving::PagedKVCache>(blocks,size);model.gather_attention=gather!=0;std::cout<<"OK\n";
        }else if(op=="BLOCK_TABLE"){
            if(!model.pool)throw std::invalid_argument("paged mode required");
            const auto& table=model.pool->table(integer(in));std::cout<<"OK";for(int block:table)std::cout<<' '<<block;std::cout<<'\n';
        }else if(op=="STEP"){
            int n=integer(in);if(n<1||n>1024)throw std::invalid_argument("batch size");std::vector<BatchItem> batch;
            for(int i=0;i<n;++i){BatchItem item;item.id=integer(in);item.generated=integer(in);item.start=integer(in);item.count=integer(in);item.prompt=integer(in);item.budget=integer(in);item.capacity=integer(in);if(item.count<1||item.count>MAX_CONTEXT)throw std::invalid_argument("input count");for(int j=0;j<item.count;++j)item.input.push_back(integer(in));batch.push_back(std::move(item));}
            auto result=model.step(batch);
            std::cout<<"OK";for(int t:result)std::cout<<' '<<t;std::cout<<'\n';
        }else if(op=="LOGITS"){
            std::cout<<"OK"<<std::setprecision(9);for(float v:model.last_logits)std::cout<<' '<<v;std::cout<<'\n';
        }else if(op=="REFERENCE_LOGITS"){
            int p=integer(in);if(p<1||p>MAX_CONTEXT)throw std::invalid_argument("prompt size");std::vector<int> tokens;for(int i=0;i<p;++i)tokens.push_back(integer(in));auto logits=model.reference_logits(tokens);std::cout<<"OK"<<std::setprecision(9);for(int j=0;j<V;++j)std::cout<<' '<<logits.at(j);std::cout<<'\n';
        }else if(op=="REFERENCE"){
            int p=integer(in),b=integer(in);if(p<1||p>MAX_CONTEXT)throw std::invalid_argument("prompt size");std::vector<int> tokens;for(int i=0;i<p;++i)tokens.push_back(integer(in));auto result=model.reference(tokens,b);std::cout<<"OK";for(int t:result)std::cout<<' '<<t;std::cout<<'\n';
        }else if(op=="RELEASE"){model.release(integer(in));std::cout<<"OK\n";
        }else if(op=="RESET"){model.clear(true);std::cout<<"OK\n";
        }else if(op=="CLEAR"){model.clear();std::cout<<"OK\n";
        }else if(op=="STATS"){
            auto* pool=model.pool.get();
            std::cout<<"OK "<<model.states.size()<<' '<<model.live_bytes()<<' '<<model.peak_bytes<<' '<<model.peak_tokens<<' '<<model.forwards<<' '<<model.batch_steps<<' '<<model.linear_calls<<' '<<model.linear_rows<<' '<<model.peak_batch_requests<<' '<<model.peak_batch_tokens
                     <<' '<<(pool?pool->capacity():0)<<' '<<(pool?pool->used_blocks():0)<<' '<<(pool?pool->peak_blocks():0)<<' '<<(pool?pool->pool_bytes():0)<<' '<<(pool?pool->capacity()-pool->used_blocks():0)<<' '<<model.paged_queries<<' '<<model.gather_queries<<'\n';
        }else if(op=="QUIT"){std::cout<<"OK\n"<<std::flush;return 0;
        }else throw std::invalid_argument("unknown command");
        std::cout<<std::flush;
    }catch(const std::exception& e){model.clear();std::cout<<"ERROR "<<e.what()<<'\n'<<std::flush;}
}
