// Tachyon - CUDA driver + NVRTC entry points resolved at runtime (dlopen), so the binary starts and runs on the CPU
// backend on machines without an NVIDIA driver or CUDA toolkit. Only the headers are needed at build time.
//
// Calls are written `dcu.cuFoo(...)`. cuda.h maps some names to versioned symbols (cuMemAlloc -> cuMemAlloc_v2);
// the same macro expansion applies to the member names below, so the loaded symbol always matches the header.
#pragma once
#include <cuda.h>
#include <nvrtc.h>
#include <string>

#define TACH_CU_FUNCS(X) \
    X(cuInit) X(cuDeviceGet) X(cuDeviceGetCount) X(cuDeviceGetName) X(cuDeviceGetAttribute) \
    X(cuDevicePrimaryCtxRetain) X(cuDevicePrimaryCtxRelease) X(cuCtxSetCurrent) X(cuCtxSynchronize) \
    X(cuGetErrorString) X(cuMemGetInfo) X(cuMemAlloc) X(cuMemFree) X(cuMemAllocHost) X(cuMemFreeHost) \
    X(cuMemHostAlloc) X(cuMemHostGetDevicePointer) X(cuMemHostRegister) X(cuMemHostUnregister) \
    X(cuMemcpyHtoD) X(cuMemcpyDtoH) X(cuMemcpyHtoDAsync) X(cuMemcpyDtoHAsync) X(cuMemcpyDtoDAsync) \
    X(cuMemsetD8) X(cuMemsetD32) X(cuMemsetD8Async) X(cuMemsetD32Async) \
    X(cuModuleLoadData) X(cuModuleUnload) X(cuModuleGetFunction) X(cuFuncGetAttribute) X(cuFuncSetAttribute) \
    X(cuOccupancyMaxActiveBlocksPerMultiprocessor) X(cuLaunchKernel) \
    X(cuStreamCreate) X(cuStreamDestroy) X(cuStreamSynchronize) X(cuStreamWaitEvent) X(cuStreamBeginCapture) X(cuStreamEndCapture) \
    X(cuGraphInstantiate) X(cuGraphLaunch) X(cuGraphDestroy) X(cuGraphExecDestroy) \
    X(cuEventCreate) X(cuEventDestroy) X(cuEventRecord) X(cuEventSynchronize) X(cuEventElapsedTime)

#define TACH_NVRTC_FUNCS(X) \
    X(nvrtcVersion) X(nvrtcGetErrorString) X(nvrtcCreateProgram) X(nvrtcDestroyProgram) X(nvrtcCompileProgram) \
    X(nvrtcAddNameExpression) X(nvrtcGetLoweredName) X(nvrtcGetProgramLogSize) X(nvrtcGetProgramLog) \
    X(nvrtcGetCUBINSize) X(nvrtcGetCUBIN)

struct CudaApi {
#define TACH_CU_DECL(f) decltype(&::f) f = nullptr;
    TACH_CU_FUNCS(TACH_CU_DECL)
    TACH_NVRTC_FUNCS(TACH_CU_DECL)
#undef TACH_CU_DECL
    bool ok = false;
    std::string error;     // why loading failed
    std::string cuda_dir;  // toolkit directory next to the loaded libnvrtc (for NVRTC include paths), may be empty
};

// Loads the libraries once (thread-safe). Check `ok` before use.
const CudaApi& cuda_api();
