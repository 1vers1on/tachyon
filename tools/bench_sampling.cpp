// Isolated candidate-compaction benchmark. Build via `make build/bench-sampling`.
// Optional argument: a different revision's src/kernels/kernels.cu, for comparisons.
#include "common.h"
#include "cuda_api.h"
#include <algorithm>
#include <vector>

int g_log_level = 0;

int main(int argc, char** argv) {
    const auto& cu = cuda_api();
    if (!cu.ok) die("%s", cu.error.c_str());
    auto check = [&](CUresult r) { if (r != CUDA_SUCCESS) { const char* err; cu.cuGetErrorString(r, &err); die("%s", err); } };
    check(cu.cuInit(0));
    CUdevice dev; check(cu.cuDeviceGet(&dev, 0));
    CUcontext ctx; check(cu.cuDevicePrimaryCtxRetain(&ctx, dev)); check(cu.cuCtxSetCurrent(ctx));
    int major, minor;
    check(cu.cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    check(cu.cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    auto source = read_file(argc > 1 ? argv[1] : "src/kernels/kernels.cu");
    auto start = source.find("extern \"C\" __global__ void __launch_bounds__(", source.find("// ---- sampling support:"));
    auto end = source.find("// bandwidth probe", start);
    if (start == std::string::npos || end == std::string::npos) die("sampling kernels not found");
    const bool split_max = source.substr(start, source.find("// pass 2", start) - start).find("blockIdx.y") != std::string::npos;
    source = "#define INFINITY __int_as_float(0x7f800000)\n__device__ float warp_max(float v) { for(int i=16;i;i/=2) v=fmaxf(v,__shfl_xor_sync(0xffffffff,v,i)); return v; }\n" + source.substr(start, end - start);
    nvrtcProgram prog;
    cu.nvrtcCreateProgram(&prog, source.c_str(), "sampling.cu", 0, nullptr, nullptr);
    std::string arch = "--gpu-architecture=sm_" + std::to_string(major) + std::to_string(minor);
    const char* opts[] = {arch.c_str(), "-std=c++17"};
    if (cu.nvrtcCompileProgram(prog, 2, opts) != NVRTC_SUCCESS) {
        size_t size; cu.nvrtcGetProgramLogSize(prog, &size);
        std::vector<char> log(size); cu.nvrtcGetProgramLog(prog, log.data()); die("%s", log.data());
    }
    size_t size; cu.nvrtcGetCUBINSize(prog, &size);
    std::vector<char> cubin(size); cu.nvrtcGetCUBIN(prog, cubin.data()); cu.nvrtcDestroyProgram(&prog);
    CUmodule mod; check(cu.cuModuleLoadData(&mod, cubin.data()));
    CUfunction rowmax, filter;
    check(cu.cuModuleGetFunction(&rowmax, mod, "row_max")); check(cu.cuModuleGetFunction(&filter, mod, "cand_filter"));
    int vocab = 151936, cap = 1024;
    CUdeviceptr logits, maxima, counts, out;
    check(cu.cuMemAlloc(&logits, 4ull * vocab * 4)); check(cu.cuMemAlloc(&maxima, 4 * 32 * 4));
    check(cu.cuMemAlloc(&counts, 16)); check(cu.cuMemAlloc(&out, 4ull * cap * 8));
    std::vector<float> host(4 * vocab);
    for (int i = 0; i < (int)host.size(); i++) host[i] = i % 512;
    check(cu.cuMemcpyHtoD(logits, host.data(), host.size() * 4));
    CUstream stream; check(cu.cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    CUevent begin, finish; check(cu.cuEventCreate(&begin, 0)); check(cu.cuEventCreate(&finish, 0));
    puts("rows,delta,candidates_per_row,microseconds");
    for (int rows : {1, 4}) for (float delta : {0.f, 14.f, 1e9f}) {
        void* maxargs[] = {&logits, &vocab, &maxima, &counts};
        void* args[] = {&logits, &vocab, &maxima, &delta, &cap, &counts, &out};
        auto run = [&] {
            check(cu.cuLaunchKernel(rowmax, split_max ? 32 : rows, split_max ? rows : 1, 1, split_max ? 256 : 1024, 1, 1, 0, stream, maxargs, nullptr));
            check(cu.cuLaunchKernel(filter, 32, rows, 1, 256, 1, 1, 0, stream, args, nullptr));
        };
        // Capture many pairs per replay so host launch latency does not dominate small kernels.
        CUgraph graph; CUgraphExec exec;
        check(cu.cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_RELAXED));
        for (int i = 0; i < 100; i++) run();
        check(cu.cuStreamEndCapture(stream, &graph));
        check(cu.cuGraphInstantiate(&exec, graph, 0));
        for (int i = 0; i < 5; i++) check(cu.cuGraphLaunch(exec, stream));
        check(cu.cuEventRecord(begin, stream));
        const int repeats = 20;
        for (int i = 0; i < repeats; i++) check(cu.cuGraphLaunch(exec, stream));
        check(cu.cuEventRecord(finish, stream)); check(cu.cuEventSynchronize(finish));
        float ms; check(cu.cuEventElapsedTime(&ms, begin, finish));
        int n; check(cu.cuMemcpyDtoH(&n, counts, sizeof n));
        printf("%d,%g,%d,%.3f\n", rows, delta, n, ms * 1000 / (100 * repeats));
        cu.cuGraphExecDestroy(exec); cu.cuGraphDestroy(graph);
    }
    cu.cuEventDestroy(begin); cu.cuEventDestroy(finish); cu.cuStreamDestroy(stream);
    for (auto p : {logits, maxima, counts, out}) cu.cuMemFree(p);
    cu.cuModuleUnload(mod); cu.cuDevicePrimaryCtxRelease(dev);
}
