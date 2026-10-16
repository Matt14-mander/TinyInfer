// Installation/precision smoke check, not acceptance of TinyInfer CUDA operators.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void check(cudaError_t s) { if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
void blas(cublasStatus_t s) { if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS failure " + std::to_string(int(s))); }
struct Resources {
    cudaStream_t stream{};
    cublasHandle_t handle{};
    float *a{}, *b{}, *c{};
    ~Resources() {
        if (stream) (void)cudaStreamSynchronize(stream);
        if (handle) (void)cublasDestroy(handle);
        if (a) (void)cudaFree(a); if (b) (void)cudaFree(b); if (c) (void)cudaFree(c);
        if (stream) (void)cudaStreamDestroy(stream);
    }
};
__global__ void double_values(float* data) {
    const auto i = threadIdx.x;
    if (i < 4) data[i] *= 2.0F;
}
}
int main(int argc, char**) {
    try {
        int count{}; const auto status = cudaGetDeviceCount(&count);
        if (status == cudaErrorNoDevice || (status == cudaSuccess && count == 0))
            return argc > 1 ? 1 : 77;
        check(status);
        cudaDeviceProp props{}; check(cudaGetDeviceProperties(&props, 0));
        Resources r;
        check(cudaStreamCreateWithFlags(&r.stream, cudaStreamNonBlocking));
        blas(cublasCreate(&r.handle)); blas(cublasSetStream(r.handle, r.stream));
        blas(cublasSetMathMode(r.handle, CUBLAS_PEDANTIC_MATH));
        blas(cublasSetAtomicsMode(r.handle, CUBLAS_ATOMICS_NOT_ALLOWED));
        cublasMath_t mode{}; blas(cublasGetMathMode(r.handle, &mode));
        if (mode != CUBLAS_PEDANTIC_MATH) throw std::runtime_error("FP32 pedantic mode not retained");
        check(cudaMalloc(&r.a, 16)); check(cudaMalloc(&r.b, 16)); check(cudaMalloc(&r.c, 16));
        const std::array<float,4> a{1,2,3,4}, b{5,6,7,8}; std::array<float,4> output{};
        check(cudaMemcpyAsync(r.a, a.data(), 16, cudaMemcpyHostToDevice, r.stream));
        check(cudaMemcpyAsync(r.b, b.data(), 16, cudaMemcpyHostToDevice, r.stream));
        double_values<<<1,4,0,r.stream>>>(r.a); check(cudaGetLastError());
        const float alpha = 1, beta = 0;
        blas(cublasSgemm(r.handle, CUBLAS_OP_N, CUBLAS_OP_N, 2,2,2,
                         &alpha, r.b,2,r.a,2,&beta,r.c,2));
        check(cudaMemcpyAsync(output.data(), r.c, 16, cudaMemcpyDeviceToHost, r.stream));
        check(cudaStreamSynchronize(r.stream));
        if (output != std::array<float,4>{38,44,86,100}) throw std::runtime_error("native CUDA/cuBLAS result differs");
        int runtime{}, driver{}, version{};
        check(cudaRuntimeGetVersion(&runtime)); check(cudaDriverGetVersion(&driver));
        blas(cublasGetVersion(r.handle, &version));
        std::cout << props.name << "; sm_" << props.major << props.minor
                  << "; runtime=" << runtime << "; driver=" << driver << "; cuBLAS=" << version
                  << "; pedantic FP32; kernel + row-major SGEMM checked\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
