// Standalone correctness/stream-time comparison; reference is the pre-optimization kernel.
#include "../src/backend/cuda/attention.cuh"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
void check(cudaError_t s){if(s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(s));}
void require(bool b,const char* s){if(!b)throw std::runtime_error(s);}
template<class T> struct Device {
    T* p=nullptr;std::size_t count;
    explicit Device(std::size_t n):count(n){check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T)));}
    ~Device(){cudaFree(p);}
    Device(const Device&)=delete;Device& operator=(const Device&)=delete;
    void put(const std::vector<T>& x){require(x.size()==count,"upload arity");check(cudaMemcpy(p,x.data(),count*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T> x(count);check(cudaMemcpy(x.data(),p,count*sizeof(T),cudaMemcpyDeviceToHost));return x;}
};
struct Events {cudaEvent_t a=nullptr,b=nullptr;
    Events(){check(cudaEventCreate(&a));try{check(cudaEventCreate(&b));}catch(...){cudaEventDestroy(a);throw;}}
    ~Events(){cudaEventDestroy(a);cudaEventDestroy(b);}
};
__global__ void reference_attention_kernel(const float* q,const float* k,const float* v,float* out,const int* table,
                                 int heads,int kvheads,int dim,int start,int layer,int layers,int block_tokens,const int* positions,const int* const* tables){
    const int row=blockIdx.x,head=blockIdx.y,lane=threadIdx.x,kh=head/(heads/kvheads),length=(positions?positions[row]:start+row)+1;
    if(tables)table=tables[row];
    __shared__ float reduction[256];
    float maximum=-CUDART_INF_F,denom=0,numerator=0;
    const float query=lane<dim?q[(row*heads+head)*dim+lane]:0;
    for(int t=0;t<length;++t){
        const std::size_t offset=(((std::size_t(table[t/block_tokens])*layers+layer)*block_tokens+t%block_tokens)*kvheads+kh)*dim;
        reduction[lane]=lane<dim?query*k[offset+lane]:0;__syncthreads();
        for(int step=128;step;step/=2){if(lane<step)reduction[lane]+=reduction[lane+step];__syncthreads();}
        float score=reduction[0]/sqrtf(float(dim));float next=fmaxf(maximum,score),scale=expf(maximum-next),weight=expf(score-next);
        denom=denom*scale+weight;if(lane<dim)numerator=numerator*scale+weight*v[offset+lane];maximum=next;
        __syncthreads();
    }
    if(lane<dim)out[(row*heads+head)*dim+lane]=numerator/denom;
}

void run(int dim,int length,int requests,int chunk,bool timing){
    const int heads=8,kvheads=dim==4?1:(dim==256?8:2),layers=3,layer=2;
    const int page=dim==34?7:8,per=std::min(length,chunk),rows=requests*per,blocks=(length+page-1)/page;
    const bool ragged=requests>1;
    const std::size_t payload=std::size_t(blocks)*requests*layers*page*kvheads*dim;
    std::vector<int> table(blocks*requests),position(rows);std::iota(table.begin(),table.end(),0);
    std::mt19937 rng(91);std::shuffle(table.begin(),table.end(),rng);
    // Exercise shared full pages across requests in addition to permuted page IDs.
    if(blocks>2)for(int r=1;r<requests;++r)table[r*blocks]=table[0];
    std::vector<float> q(std::size_t(rows)*heads*dim),k(payload),v(payload);
    const float magnitude=length==128?20.0F:0.01F;
    for(std::size_t i=0;i<q.size();++i)q[i]=float(int(i*17%127)-63)*magnitude;
    for(std::size_t i=0;i<payload;++i){k[i]=float(int(i*19%131)-65)*.01F;v[i]=float(int(i*23%137)-68)*.01F;}
    Device<float> dq(q.size()),dk(payload),dv(payload),old(q.size()),out(q.size());
    Device<int> dt(table.size()),dp(rows);Device<const int*> dptr(rows);
    std::vector<const int*> ptr(rows);
    for(int r=0;r<requests;++r)for(int t=0;t<per;++t){
        position[r*per+t]=std::max(per,length-r*3)-per+t;ptr[r*per+t]=dt.p+r*blocks;
    }
    dt.put(table);dp.put(position);dptr.put(ptr);dq.put(q);dk.put(k);dv.put(v);
    auto launch=[&](bool improved){
        if(improved)miniprefix::cuda_detail::launch_attention(rows,nullptr,dq.p,dk.p,dv.p,out.p,dt.p,
            heads,kvheads,dim,length-per,layer,layers,page,ragged?dp.p:nullptr,ragged?dptr.p:nullptr);
        else reference_attention_kernel<<<dim3(rows,heads),256>>>(dq.p,dk.p,dv.p,old.p,dt.p,
            heads,kvheads,dim,length-per,layer,layers,page,ragged?dp.p:nullptr,ragged?dptr.p:nullptr);
    };
    launch(false);launch(true);check(cudaGetLastError());check(cudaDeviceSynchronize());
    const auto expected=old.get(),actual=out.get();double difference=0,oracle_error=0;
    for(std::size_t i=0;i<actual.size();++i){
        require(std::isfinite(actual[i])&&std::abs(actual[i]-expected[i])<2e-6F,"reference mismatch");
        difference=std::max(difference,double(std::abs(actual[i]-expected[i])));
    }
    // Independent dense FP64 softmax for the first and last query rows.
    for(int row:{0,rows-1})for(int head=0;head<heads;++head){
        const int n=(ragged?position[row]:length-per+row)+1,request=row/per;
        std::vector<double> scores(n),sum(dim,0);double maximum=-INFINITY,denom=0;
        auto offset=[&](int t){return (((std::size_t(table[request*blocks+t/page])*layers+layer)*page+t%page)*kvheads+head/(heads/kvheads))*dim;};
        for(int t=0;t<n;++t){double dot=0;const auto at=offset(t);
            for(int d=0;d<dim;++d)dot+=double(q[(row*heads+head)*dim+d])*k[at+d];
            scores[t]=dot/std::sqrt(double(dim));maximum=std::max(maximum,scores[t]);}
        for(int t=0;t<n;++t){const double weight=std::exp(scores[t]-maximum);denom+=weight;const auto at=offset(t);
            for(int d=0;d<dim;++d)sum[d]+=weight*v[at+d];}
        for(int d=0;d<dim;++d){const auto error=std::abs(actual[(row*heads+head)*dim+d]-sum[d]/denom);
            // Large queries (2000x normal amplitude) amplify the baseline FP32
            // dot-product rounding relative to FP64. Keep a separate stress bound;
            // every output still has the strict old/new comparison above.
            oracle_error=std::max(oracle_error,error);require(error<(length==128?1e-3:2e-5),"FP64 oracle mismatch");}
    }
    std::vector<double> before,after;
    if(timing){
        Events events;for(int i=0;i<3;++i){launch(false);launch(true);}check(cudaDeviceSynchronize());
        for(int repeat=0;repeat<3;++repeat)for(int order=0;order<2;++order){
            const bool improved=(repeat+order)%2;check(cudaEventRecord(events.a));
            for(int i=0;i<20;++i)launch(improved);
            check(cudaEventRecord(events.b));check(cudaEventSynchronize(events.b));check(cudaGetLastError());
            float ms=0;check(cudaEventElapsedTime(&ms,events.a,events.b));(improved?after:before).push_back(ms*1000/20);
        }
    }
    std::cout<<std::setprecision(9)<<"{\"passed\":true,\"dim\":"<<dim<<",\"length\":"<<length<<",\"requests\":"<<requests
        <<",\"chunk\":"<<chunk<<",\"rows\":"<<rows<<",\"kv_heads\":"<<kvheads<<",\"page\":"<<page
        <<",\"max_difference\":"<<difference<<",\"oracle_error\":"<<oracle_error<<",\"before_us\":[";
    for(std::size_t i=0;i<before.size();++i){if(i)std::cout<<',';std::cout<<before[i];}std::cout<<"],\"after_us\":[";
    for(std::size_t i=0;i<after.size();++i){if(i)std::cout<<',';std::cout<<after[i];}std::cout<<"]}\n";
}
int main(int argc,char** argv){try{
    bool timing=true,required=false,quick=false;
    for(int i=1;i<argc;++i){const std::string arg=argv[i];if(arg=="--check")timing=false;
        else if(arg=="--require-device")required=true;else if(arg=="--quick")quick=true;else throw std::invalid_argument("unknown option");}
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||!devices)return required?1:77;
    for(int dim:{4,32,34,64,128,256})for(int length:{1,7,33,128,512,2048})
        for(int requests:{1,4})for(int chunk:{1,16}){
            if(quick&&((dim!=34&&dim!=256)||(length!=7&&length!=33)||requests!=4))continue;
            run(dim,length,requests,chunk,timing);
        }
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
