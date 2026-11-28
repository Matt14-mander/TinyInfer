#pragma once
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstddef>

namespace miniprefix::cuda_detail {
// One complete warp per query/head. Elements is the next power-of-two number
// of 32-wide slices (1/2/4/8), covering the model's validated dim <= 256.
// Reduce the slices in the same high-to-low tree as the previous 256-lane
// shared-memory reduction, then reduce within the warp. Keep online softmax's
// token order, FP32 accumulators and paged addressing unchanged.
template<int Elements>
__global__ void warp_attention_kernel(const float* q,const float* k,const float* v,float* out,const int* table,
    int heads,int kvheads,int dim,int start,int layer,int layers,int block_tokens,const int* positions,const int* const* tables){
    const int row=blockIdx.x,head=blockIdx.y,lane=threadIdx.x;
    const int kh=head/(heads/kvheads),length=(positions?positions[row]:start+row)+1;
    if(tables)table=tables[row];
    float query[Elements],numerator[Elements];
    #pragma unroll
    for(int i=0;i<Elements;++i){
        const int d=lane+32*i;
        query[i]=d<dim?q[(row*heads+head)*dim+d]:0;
        numerator[i]=0;
    }
    float maximum=-CUDART_INF_F,denom=0;
    for(int t=0;t<length;++t){
        const std::size_t offset=(((std::size_t(table[t/block_tokens])*layers+layer)*block_tokens+t%block_tokens)*kvheads+kh)*dim;
        float products[Elements];
        #pragma unroll
        for(int i=0;i<Elements;++i){const int d=lane+32*i;products[i]=d<dim?query[i]*k[offset+d]:0;}
        #pragma unroll
        for(int stride=Elements/2;stride;stride/=2){
            #pragma unroll
            for(int i=0;i<stride;++i)products[i]+=products[i+stride];
        }
        float dot=products[0];
        #pragma unroll
        for(int stride=16;stride;stride/=2)dot+=__shfl_down_sync(0xffffffffu,dot,stride);
        const float score=__shfl_sync(0xffffffffu,dot,0)/sqrtf(float(dim));
        const float next=fmaxf(maximum,score),scale=expf(maximum-next),weight=expf(score-next);
        denom=denom*scale+weight;
        #pragma unroll
        for(int i=0;i<Elements;++i){const int d=lane+32*i;if(d<dim)numerator[i]=numerator[i]*scale+weight*v[offset+d];}
        maximum=next;
    }
    #pragma unroll
    for(int i=0;i<Elements;++i){const int d=lane+32*i;if(d<dim)out[(row*heads+head)*dim+d]=numerator[i]/denom;}
}

inline void launch_attention(int rows,cudaStream_t stream,const float* q,const float* k,const float* v,float* out,const int* table,
    int heads,int kvheads,int dim,int start,int layer,int layers,int block_tokens,const int* positions,const int* const* tables){
    #define MINIPREFIX_ATTENTION(E) warp_attention_kernel<E><<<dim3(rows,heads),32,0,stream>>>(q,k,v,out,table,heads,kvheads,dim,start,layer,layers,block_tokens,positions,tables)
    if(dim<=32){MINIPREFIX_ATTENTION(1);}
    else if(dim<=64){MINIPREFIX_ATTENTION(2);}
    else if(dim<=128){MINIPREFIX_ATTENTION(4);}
    else{MINIPREFIX_ATTENTION(8);}
    #undef MINIPREFIX_ATTENTION
}
} // namespace miniprefix::cuda_detail
