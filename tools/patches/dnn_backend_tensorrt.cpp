/*
 * Copyright 2026 Joshua V. Dillon
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file
 * DNN TensorRT backend implementation.
 *
 * This backend loads pre-compiled TensorRT engine files (.engine) for
 * high-performance GPU inference. Use tools/export-tensorrt.py to convert
 * PyTorch models to TensorRT engines.
 *
 * All libraries are loaded at runtime via dlopen - no CUDA or TensorRT
 * dependency at ffmpeg load time. Errors only occur when the TRT backend
 * is actually used.
 *
 * Usage:
 *   ffmpeg -i input.mp4 -vf "dnn_processing=dnn_backend=tensorrt:model=model.engine" output.mp4
 */

#include <NvInfer.h>
#include <dlfcn.h>
#include <fstream>
#include <vector>
#include <memory>
#include <cstring>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <string>

// ============================================================================
// Engine cache - avoid reloading same engine file multiple times
// ============================================================================
struct CachedEngine {
    nvinfer1::ICudaEngine *engine;
    nvinfer1::IRuntime *runtime;
    std::atomic<int> refcount;

    CachedEngine(nvinfer1::ICudaEngine *e, nvinfer1::IRuntime *r)
        : engine(e), runtime(r), refcount(1) {}
};

static std::mutex g_engine_cache_mutex;
static std::unordered_map<std::string, CachedEngine*> g_engine_cache;

// ============================================================================
// CUDA Driver API types (from cuda.h - we dlopen libcuda.so instead of linking)
// ============================================================================
typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;
typedef unsigned long long CUdeviceptr;

#define CUDA_SUCCESS 0

// CUDA Driver API function pointer types
typedef CUresult (*fn_cuInit)(unsigned int);
typedef CUresult (*fn_cuDeviceGet)(CUdevice*, int);
typedef CUresult (*fn_cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
typedef CUresult (*fn_cuCtxGetCurrent)(CUcontext*);
typedef CUresult (*fn_cuCtxSetCurrent)(CUcontext);
typedef CUresult (*fn_cuCtxPushCurrent)(CUcontext);
typedef CUresult (*fn_cuCtxPopCurrent)(CUcontext*);
typedef CUresult (*fn_cuMemAlloc)(CUdeviceptr*, size_t);
typedef CUresult (*fn_cuMemFree)(CUdeviceptr);

// CUDA Runtime API function pointers (for compatibility with TensorRT which uses Runtime API)
// Note: cudaError_t is already defined via NvInfer.h -> cuda_runtime_api.h
typedef cudaError_t (*fn_cudaMalloc)(void**, size_t);
typedef cudaError_t (*fn_cudaFree)(void*);
typedef cudaError_t (*fn_cudaSetDevice)(int);
typedef cudaError_t (*fn_cudaMemcpy)(void*, const void*, size_t, int);
typedef cudaError_t (*fn_cudaMemcpyAsync)(void*, const void*, size_t, int, cudaStream_t);
typedef cudaError_t (*fn_cudaStreamSynchronize)(cudaStream_t);
typedef cudaError_t (*fn_cudaStreamCreate)(cudaStream_t*, unsigned int);
typedef cudaError_t (*fn_cudaStreamDestroy)(cudaStream_t);
typedef cudaError_t (*fn_cudaMemGetInfo)(size_t*, size_t*);
#define cudaMemcpyHostToDevice 1
#define cudaMemcpyDeviceToHost 2
typedef CUresult (*fn_cuMemcpyHtoD)(CUdeviceptr, const void*, size_t);
typedef CUresult (*fn_cuMemcpyDtoH)(void*, CUdeviceptr, size_t);
typedef CUresult (*fn_cuMemcpyHtoDAsync)(CUdeviceptr, const void*, size_t, CUstream);
typedef CUresult (*fn_cuMemcpyDtoHAsync)(void*, CUdeviceptr, size_t, CUstream);
typedef CUresult (*fn_cuStreamCreate)(CUstream*, unsigned int);
typedef CUresult (*fn_cuStreamDestroy)(CUstream);
typedef CUresult (*fn_cuStreamSynchronize)(CUstream);
typedef CUresult (*fn_cuModuleLoadData)(CUmodule*, const void*);
typedef CUresult (*fn_cuModuleUnload)(CUmodule);
typedef CUresult (*fn_cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
typedef CUresult (*fn_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
                                       unsigned int, unsigned int, unsigned int,
                                       unsigned int, CUstream, void**, void**);
typedef CUresult (*fn_cuGetErrorString)(CUresult, const char**);

// CUDA Graph API function pointer types
typedef void* CUgraph;
typedef void* CUgraphExec;
typedef CUresult (*fn_cuStreamBeginCapture)(CUstream, int);
typedef CUresult (*fn_cuStreamEndCapture)(CUstream, CUgraph*);
typedef CUresult (*fn_cuGraphInstantiate)(CUgraphExec*, CUgraph, unsigned long long);
typedef CUresult (*fn_cuGraphLaunch)(CUgraphExec, CUstream);
typedef CUresult (*fn_cuGraphDestroy)(CUgraph);
typedef CUresult (*fn_cuGraphExecDestroy)(CUgraphExec);

#define CU_STREAM_CAPTURE_MODE_GLOBAL 0

// ============================================================================
// Dynamic library loading for CUDA and TensorRT
// NOTE: These handles are intentionally never dlclose'd. CUDA/TensorRT libraries
// have complex cleanup requirements and calling dlclose can cause crashes.
// The OS reclaims resources on process exit.
// ============================================================================
static void *libcuda_handle = NULL;
static void *libnvinfer_handle = NULL;
static int cuda_loaded = 0;
static int tensorrt_loaded = 0;
static std::atomic<int> libs_load_attempted(0);
static std::mutex libs_load_mutex;

// CUDA Driver API function pointers
static fn_cuInit p_cuInit = NULL;
static fn_cuDeviceGet p_cuDeviceGet = NULL;
static fn_cuDevicePrimaryCtxRetain p_cuDevicePrimaryCtxRetain = NULL;
static fn_cuCtxGetCurrent p_cuCtxGetCurrent = NULL;
static fn_cuCtxSetCurrent p_cuCtxSetCurrent = NULL;
static fn_cuCtxPushCurrent p_cuCtxPushCurrent = NULL;
static fn_cuCtxPopCurrent p_cuCtxPopCurrent = NULL;
static fn_cuMemAlloc p_cuMemAlloc = NULL;
static fn_cuMemFree p_cuMemFree = NULL;
static fn_cudaMalloc p_cudaMalloc = NULL;
static fn_cudaFree p_cudaFree = NULL;
static fn_cudaSetDevice p_cudaSetDevice = NULL;
static fn_cudaMemcpy p_cudaMemcpy = NULL;
static fn_cudaMemcpyAsync p_cudaMemcpyAsync = NULL;
static fn_cudaStreamSynchronize p_cudaStreamSynchronize_rt = NULL;  // Runtime API stream sync
static fn_cudaStreamCreate p_cudaStreamCreate_rt = NULL;  // Runtime API stream create
static fn_cudaMemGetInfo p_cudaMemGetInfo = NULL;  // For memory diagnostics
static void *cuda_rt_handle = NULL;  // libcudart.so handle
static fn_cuMemcpyHtoD p_cuMemcpyHtoD = NULL;
static fn_cuMemcpyDtoH p_cuMemcpyDtoH = NULL;
static fn_cuMemcpyHtoDAsync p_cuMemcpyHtoDAsync = NULL;
static fn_cuMemcpyDtoHAsync p_cuMemcpyDtoHAsync = NULL;
static fn_cuStreamCreate p_cuStreamCreate = NULL;
static fn_cuStreamDestroy p_cuStreamDestroy = NULL;
static fn_cuStreamSynchronize p_cuStreamSynchronize = NULL;
static fn_cuModuleLoadData p_cuModuleLoadData = NULL;
static fn_cuModuleUnload p_cuModuleUnload = NULL;
static fn_cuModuleGetFunction p_cuModuleGetFunction = NULL;
static fn_cuLaunchKernel p_cuLaunchKernel = NULL;
static fn_cuGetErrorString p_cuGetErrorString = NULL;

// CUDA Graph API function pointers (optional - graceful fallback if unavailable)
static fn_cuStreamBeginCapture p_cuStreamBeginCapture = NULL;
static fn_cuStreamEndCapture p_cuStreamEndCapture = NULL;
static fn_cuGraphInstantiate p_cuGraphInstantiate = NULL;
static fn_cuGraphLaunch p_cuGraphLaunch = NULL;
static fn_cuGraphDestroy p_cuGraphDestroy = NULL;
static fn_cuGraphExecDestroy p_cuGraphExecDestroy = NULL;
static int cuda_graphs_available = 0;

// TensorRT factory function pointer
typedef nvinfer1::IRuntime* (*fn_createInferRuntime)(nvinfer1::ILogger&);
static fn_createInferRuntime p_createInferRuntime = NULL;

// Forward declaration
static int load_libs(void *log_ctx);

extern "C" {
#include "dnn_io_proc.h"
#include "dnn_backend_common.h"
#include "libavutil/opt.h"
#include "libavutil/mem.h"
#include "libavutil/avassert.h"
#include "libavutil/internal.h"
#include "libavutil/hwcontext.h"
#include "libavutil/pixfmt.h"
#include "libavutil/pixdesc.h"
#include "queue.h"
#include "safe_queue.h"
#include "dnn_cuda_kernels.h"
}

// Get CUDA error string
static const char* cuda_error_string(CUresult err) {
    const char *str = NULL;
    if (p_cuGetErrorString && p_cuGetErrorString(err, &str) == CUDA_SUCCESS && str)
        return str;
    return "unknown CUDA error";
}

// Load CUDA Driver API and TensorRT via dlopen
static int load_libs(void *log_ctx) {
    // Double-checked locking for thread safety
    if (libs_load_attempted.load(std::memory_order_acquire))
        return (cuda_loaded && tensorrt_loaded) ? 0 : AVERROR(ENOSYS);

    std::lock_guard<std::mutex> lock(libs_load_mutex);
    // Check again after acquiring lock
    if (libs_load_attempted.load(std::memory_order_relaxed))
        return (cuda_loaded && tensorrt_loaded) ? 0 : AVERROR(ENOSYS);

    // Set at end of function, not here, to ensure proper initialization
    // before other threads see libs_load_attempted == 1

    // Load CUDA Driver API (libcuda.so - NOT libcudart.so!)
    const char *cuda_names[] = {"libcuda.so.1", "libcuda.so", NULL};
    for (int i = 0; cuda_names[i] && !libcuda_handle; i++) {
        libcuda_handle = dlopen(cuda_names[i], RTLD_NOW);
    }
    if (!libcuda_handle) {
        av_log(log_ctx, AV_LOG_ERROR,
               "CUDA driver not available: %s\n"
               "Install NVIDIA driver or run with --gpus all to use nvidia-container-toolkit\n",
               dlerror());
        libs_load_attempted.store(1, std::memory_order_release);
        return AVERROR(ENOSYS);
    }

    // Load all required CUDA functions
    #define LOAD_CUDA_FUNC(name) \
        p_##name = (fn_##name)dlsym(libcuda_handle, #name); \
        if (!p_##name) { \
            av_log(log_ctx, AV_LOG_ERROR, "Failed to load CUDA function: %s\n", #name); \
            dlclose(libcuda_handle); libcuda_handle = NULL; \
            libs_load_attempted.store(1, std::memory_order_release); \
            return AVERROR(ENOSYS); \
        }

    LOAD_CUDA_FUNC(cuInit);
    LOAD_CUDA_FUNC(cuDeviceGet);
    LOAD_CUDA_FUNC(cuDevicePrimaryCtxRetain);
    LOAD_CUDA_FUNC(cuCtxGetCurrent);
    LOAD_CUDA_FUNC(cuCtxSetCurrent);
    LOAD_CUDA_FUNC(cuCtxPushCurrent);
    LOAD_CUDA_FUNC(cuCtxPopCurrent);
    LOAD_CUDA_FUNC(cuMemAlloc);
    LOAD_CUDA_FUNC(cuMemFree);
    LOAD_CUDA_FUNC(cuMemcpyHtoD);
    LOAD_CUDA_FUNC(cuMemcpyDtoH);
    LOAD_CUDA_FUNC(cuMemcpyHtoDAsync);
    LOAD_CUDA_FUNC(cuMemcpyDtoHAsync);
    LOAD_CUDA_FUNC(cuStreamCreate);
    LOAD_CUDA_FUNC(cuStreamDestroy);
    LOAD_CUDA_FUNC(cuStreamSynchronize);
    LOAD_CUDA_FUNC(cuModuleLoadData);
    LOAD_CUDA_FUNC(cuModuleUnload);
    LOAD_CUDA_FUNC(cuModuleGetFunction);
    LOAD_CUDA_FUNC(cuLaunchKernel);
    // cuGetErrorString is optional (for better error messages)
    p_cuGetErrorString = (fn_cuGetErrorString)dlsym(libcuda_handle, "cuGetErrorString");

    // CUDA Graph API (optional - CUDA 10.0+, graceful fallback if unavailable)
    p_cuStreamBeginCapture = (fn_cuStreamBeginCapture)dlsym(libcuda_handle, "cuStreamBeginCapture");
    p_cuStreamEndCapture = (fn_cuStreamEndCapture)dlsym(libcuda_handle, "cuStreamEndCapture");
    p_cuGraphInstantiate = (fn_cuGraphInstantiate)dlsym(libcuda_handle, "cuGraphInstantiateWithFlags");
    p_cuGraphLaunch = (fn_cuGraphLaunch)dlsym(libcuda_handle, "cuGraphLaunch");
    p_cuGraphDestroy = (fn_cuGraphDestroy)dlsym(libcuda_handle, "cuGraphDestroy");
    p_cuGraphExecDestroy = (fn_cuGraphExecDestroy)dlsym(libcuda_handle, "cuGraphExecDestroy");

    // CUDA Graphs disabled - adds ~1GB memory overhead with no fps improvement
    // (TensorRT is compute-bound in convolutions, not kernel-launch-bound)
    // Keep the function pointers loaded in case we want to re-enable later
    (void)p_cuStreamBeginCapture;
    (void)p_cuStreamEndCapture;
    (void)p_cuGraphInstantiate;
    (void)p_cuGraphLaunch;
    (void)p_cuGraphDestroy;
    (void)p_cuGraphExecDestroy;
    cuda_graphs_available = 0;

    #undef LOAD_CUDA_FUNC

    // Initialize CUDA (needed before any other CUDA calls)
    CUresult err = p_cuInit(0);
    if (err != CUDA_SUCCESS) {
        av_log(log_ctx, AV_LOG_ERROR, "cuInit failed: %s\n", cuda_error_string(err));
        dlclose(libcuda_handle);
        libcuda_handle = NULL;
        libs_load_attempted.store(1, std::memory_order_release);
        return AVERROR(ENOSYS);
    }

    cuda_loaded = 1;
    av_log(log_ctx, AV_LOG_INFO, "CUDA driver API loaded via dlopen\n");

    // Load CUDA Runtime API (required for TensorRT compatibility)
    const char *cudart_names[] = {
        "libcudart.so.13", "libcudart.so.12", "libcudart.so.11", "libcudart.so", NULL
    };
    for (int i = 0; cudart_names[i] && !cuda_rt_handle; i++) {
        cuda_rt_handle = dlopen(cudart_names[i], RTLD_NOW);
    }
    if (!cuda_rt_handle) {
        av_log(log_ctx, AV_LOG_ERROR, "Failed to load CUDA runtime library (libcudart.so)\n");
        dlclose(libcuda_handle);
        libcuda_handle = NULL;
        cuda_loaded = 0;
        libs_load_attempted.store(1, std::memory_order_release);
        return AVERROR(ENOSYS);
    }
    p_cudaMalloc = (fn_cudaMalloc)dlsym(cuda_rt_handle, "cudaMalloc");
    p_cudaFree = (fn_cudaFree)dlsym(cuda_rt_handle, "cudaFree");
    p_cudaSetDevice = (fn_cudaSetDevice)dlsym(cuda_rt_handle, "cudaSetDevice");
    p_cudaMemcpy = (fn_cudaMemcpy)dlsym(cuda_rt_handle, "cudaMemcpy");
    p_cudaMemcpyAsync = (fn_cudaMemcpyAsync)dlsym(cuda_rt_handle, "cudaMemcpyAsync");
    p_cudaStreamSynchronize_rt = (fn_cudaStreamSynchronize)dlsym(cuda_rt_handle, "cudaStreamSynchronize");
    p_cudaStreamCreate_rt = (fn_cudaStreamCreate)dlsym(cuda_rt_handle, "cudaStreamCreate");
    p_cudaMemGetInfo = (fn_cudaMemGetInfo)dlsym(cuda_rt_handle, "cudaMemGetInfo");  // Optional, for diagnostics
    if (!p_cudaMalloc || !p_cudaFree || !p_cudaSetDevice || !p_cudaMemcpy) {
        av_log(log_ctx, AV_LOG_ERROR, "Failed to load CUDA runtime API functions\n");
        dlclose(cuda_rt_handle);
        cuda_rt_handle = NULL;
        dlclose(libcuda_handle);
        libcuda_handle = NULL;
        cuda_loaded = 0;
        libs_load_attempted.store(1, std::memory_order_release);
        return AVERROR(ENOSYS);
    }

    // Load TensorRT
    const char *nvinfer_names[] = {
        "libnvinfer.so.10", "libnvinfer.so.9", "libnvinfer.so.8", "libnvinfer.so", NULL
    };
    for (int i = 0; nvinfer_names[i] && !libnvinfer_handle; i++) {
        libnvinfer_handle = dlopen(nvinfer_names[i], RTLD_NOW);
    }
    if (!libnvinfer_handle) {
        av_log(log_ctx, AV_LOG_ERROR,
               "TensorRT not available: %s\n"
               "Install TensorRT or run with --gpus all to use nvidia-container-toolkit\n",
               dlerror());
        dlclose(cuda_rt_handle);
        cuda_rt_handle = NULL;
        dlclose(libcuda_handle);
        libcuda_handle = NULL;
        cuda_loaded = 0;
        libs_load_attempted.store(1, std::memory_order_release);
        return AVERROR(ENOSYS);
    }

    // Get TensorRT factory function
    const char *create_runtime_names[] = {
        "createInferRuntime_INTERNAL",  // TensorRT 10+
        "_ZN8nvinfer118createInferRuntimeERNS_7ILoggerE",  // GCC mangling (TRT 8-9)
        "createInferRuntime",  // Some builds export unmangled
        NULL
    };
    for (int i = 0; create_runtime_names[i] && !p_createInferRuntime; i++) {
        p_createInferRuntime = (fn_createInferRuntime)dlsym(libnvinfer_handle, create_runtime_names[i]);
    }
    if (!p_createInferRuntime) {
        av_log(log_ctx, AV_LOG_ERROR, "Failed to find createInferRuntime in TensorRT library\n");
        dlclose(libnvinfer_handle);
        libnvinfer_handle = NULL;
        dlclose(cuda_rt_handle);
        cuda_rt_handle = NULL;
        dlclose(libcuda_handle);
        libcuda_handle = NULL;
        cuda_loaded = 0;
        libs_load_attempted.store(1, std::memory_order_release);
        return AVERROR(ENOSYS);
    }

    tensorrt_loaded = 1;
    av_log(log_ctx, AV_LOG_INFO, "TensorRT library loaded via dlopen\n");

    // Mark as attempted AFTER successful initialization (release semantics)
    libs_load_attempted.store(1, std::memory_order_release);
    return 0;
}

// Log GPU memory usage for diagnostics
static void log_gpu_memory(void *log_ctx, const char *label) {
    if (!p_cudaMemGetInfo) return;
    size_t free_mem = 0, total_mem = 0;
    if (p_cudaMemGetInfo(&free_mem, &total_mem) == 0) {
        size_t used_mb = (total_mem - free_mem) / (1024 * 1024);
        av_log(log_ctx, AV_LOG_WARNING, "GPU_MEM [%s]: %zu MiB used\n", label, used_mb);
    }
}

// TensorRT logger - forward to FFmpeg's logging
class TRTLogger : public nvinfer1::ILogger {
public:
    void *log_ctx;
    TRTLogger(void *ctx = nullptr) : log_ctx(ctx) {}

    void log(Severity severity, const char *msg) noexcept override {
        int level;
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
            case Severity::kERROR:
                level = AV_LOG_ERROR;
                break;
            case Severity::kWARNING:
                level = AV_LOG_WARNING;
                break;
            case Severity::kINFO:
                level = AV_LOG_INFO;
                break;
            default:
                level = AV_LOG_DEBUG;
                break;
        }
        av_log(log_ctx, level, "TensorRT: %s\n", msg);
    }
};

// Supported tensor data types
typedef enum TRTDataType {
    TRT_DT_FLOAT32 = 0,  // 4 bytes
    TRT_DT_FLOAT16 = 1,  // 2 bytes
    TRT_DT_BFLOAT16 = 2, // 2 bytes
    TRT_DT_INT8 = 3,     // 1 byte
    TRT_DT_UINT8 = 4,    // 1 byte
    TRT_DT_UNKNOWN = -1
} TRTDataType;

static const char *trt_dtype_name(TRTDataType dt) {
    switch (dt) {
        case TRT_DT_FLOAT32: return "FP32";
        case TRT_DT_FLOAT16: return "FP16";
        case TRT_DT_BFLOAT16: return "BF16";
        case TRT_DT_INT8: return "INT8";
        case TRT_DT_UINT8: return "UINT8";
        default: return "UNKNOWN";
    }
}

static size_t trt_dtype_size(TRTDataType dt) {
    switch (dt) {
        case TRT_DT_FLOAT32: return 4;
        case TRT_DT_FLOAT16: return 2;
        case TRT_DT_BFLOAT16: return 2;
        case TRT_DT_INT8: return 1;
        case TRT_DT_UINT8: return 1;
        default: return 0;
    }
}

static TRTDataType nvinfer_to_trt_dtype(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return TRT_DT_FLOAT32;
        case nvinfer1::DataType::kHALF: return TRT_DT_FLOAT16;
        case nvinfer1::DataType::kBF16: return TRT_DT_BFLOAT16;
        case nvinfer1::DataType::kINT8: return TRT_DT_INT8;
        case nvinfer1::DataType::kUINT8: return TRT_DT_UINT8;
        default: return TRT_DT_UNKNOWN;
    }
}

typedef struct TRTModel {
    DNNModel model;
    DnnContext *ctx;
    nvinfer1::IRuntime *runtime;
    nvinfer1::ICudaEngine *engine;
    nvinfer1::IExecutionContext *context;  // Lazily created on first inference
    TRTLogger *logger;
    CUstream stream;

    // CUDA Graphs for reduced kernel launch overhead
    CUgraph cuda_graph;
    CUgraphExec cuda_graph_exec;
    int cuda_graph_captured;  // 1 if graph has been captured and is ready
    int cuda_graph_failed;    // 1 if capture failed, don't retry

    // Engine cache entry (for refcounting shared engines)
    CachedEngine *cached_engine;
    char *engine_path;  // Key for cache lookup (av_strdup'd)

    // CUDA kernel module (loaded from PTX)
    CUmodule cuda_module;
    // Kernels for each dtype: [0]=FP32, [1]=FP16, [2]=BF16
    CUfunction kernel_hwc_to_nchw[3];
    CUfunction kernel_nchw_to_hwc[3];
    CUfunction kernel_hwc4_to_nchw[3];
    CUfunction kernel_nchw_to_hwc4[3];

    // I/O tensor info (TensorRT 10.x API)
    char *input_name;
    char *output_name;
    nvinfer1::Dims input_dims;
    nvinfer1::Dims output_dims;
    TRTDataType input_dtype;
    TRTDataType output_dtype;

    // CUDA buffers (using CUdeviceptr for Driver API)
    CUdeviceptr input_buffer;
    CUdeviceptr output_buffer;
    size_t input_size;
    size_t output_size;

    // Task management (reuse FFmpeg's queue infrastructure)
    SafeQueue *request_queue;
    Queue *task_queue;
    Queue *lltask_queue;
} TRTModel;

typedef struct TRTInferRequest {
    float *output_data;  // CPU output buffer
} TRTInferRequest;

typedef struct TRTRequestItem {
    TRTInferRequest *infer_request;
    LastLevelTaskItem *lltask;
    DNNAsyncExecModule exec_module;
} TRTRequestItem;

#define OFFSET(x) offsetof(TRTOptions, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM

static const AVOption dnn_trt_options[] = {
    { "device_id", "CUDA device ID", OFFSET(device_id), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 16, FLAGS },
    { NULL }
};

// Check CUDA error and log
#define CUDA_CHECK(call, ctx, ret) do { \
    CUresult cuda_err = (call); \
    if (cuda_err != CUDA_SUCCESS) { \
        av_log(ctx, AV_LOG_ERROR, "CUDA error: %s\n", cuda_error_string(cuda_err)); \
        return ret; \
    } \
} while(0)

// Load CUDA kernels from embedded PTX
static int load_cuda_kernels(TRTModel *trt_model, void *log_ctx) {
    CUresult err;

    // Load PTX module
    err = p_cuModuleLoadData(&trt_model->cuda_module, ff_dnn_cuda_kernels_ptx);
    if (err != CUDA_SUCCESS) {
        av_log(log_ctx, AV_LOG_ERROR, "Failed to load CUDA kernel module: %s\n",
               cuda_error_string(err));
        return AVERROR(ENOSYS);
    }

    // Kernel names for each dtype: [0]=FP32, [1]=FP16, [2]=BF16
    const char *hwc_to_nchw_names[] = {
        DNN_CUDA_KERNEL_HWC_UINT8_TO_NCHW_FLOAT32,
        DNN_CUDA_KERNEL_HWC_UINT8_TO_NCHW_FLOAT16,
        DNN_CUDA_KERNEL_HWC_UINT8_TO_NCHW_BFLOAT16
    };
    const char *nchw_to_hwc_names[] = {
        DNN_CUDA_KERNEL_NCHW_FLOAT32_TO_HWC_UINT8,
        DNN_CUDA_KERNEL_NCHW_FLOAT16_TO_HWC_UINT8,
        DNN_CUDA_KERNEL_NCHW_BFLOAT16_TO_HWC_UINT8
    };
    const char *hwc4_to_nchw_names[] = {
        DNN_CUDA_KERNEL_HWC4_UINT8_TO_NCHW_FLOAT32,
        DNN_CUDA_KERNEL_HWC4_UINT8_TO_NCHW_FLOAT16,
        DNN_CUDA_KERNEL_HWC4_UINT8_TO_NCHW_BFLOAT16
    };
    const char *nchw_to_hwc4_names[] = {
        DNN_CUDA_KERNEL_NCHW_FLOAT32_TO_HWC4_UINT8,
        DNN_CUDA_KERNEL_NCHW_FLOAT16_TO_HWC4_UINT8,
        DNN_CUDA_KERNEL_NCHW_BFLOAT16_TO_HWC4_UINT8
    };

    // Load all kernel variants
    for (int i = 0; i < 3; i++) {
        err = p_cuModuleGetFunction(&trt_model->kernel_hwc_to_nchw[i], trt_model->cuda_module, hwc_to_nchw_names[i]);
        if (err != CUDA_SUCCESS) {
            av_log(log_ctx, AV_LOG_ERROR, "Failed to get kernel %s: %s\n", hwc_to_nchw_names[i], cuda_error_string(err));
            p_cuModuleUnload(trt_model->cuda_module);
            trt_model->cuda_module = NULL;
            return AVERROR(ENOSYS);
        }

        err = p_cuModuleGetFunction(&trt_model->kernel_nchw_to_hwc[i], trt_model->cuda_module, nchw_to_hwc_names[i]);
        if (err != CUDA_SUCCESS) {
            av_log(log_ctx, AV_LOG_ERROR, "Failed to get kernel %s: %s\n", nchw_to_hwc_names[i], cuda_error_string(err));
            p_cuModuleUnload(trt_model->cuda_module);
            trt_model->cuda_module = NULL;
            return AVERROR(ENOSYS);
        }

        err = p_cuModuleGetFunction(&trt_model->kernel_hwc4_to_nchw[i], trt_model->cuda_module, hwc4_to_nchw_names[i]);
        if (err != CUDA_SUCCESS) {
            av_log(log_ctx, AV_LOG_ERROR, "Failed to get kernel %s: %s\n", hwc4_to_nchw_names[i], cuda_error_string(err));
            p_cuModuleUnload(trt_model->cuda_module);
            trt_model->cuda_module = NULL;
            return AVERROR(ENOSYS);
        }

        err = p_cuModuleGetFunction(&trt_model->kernel_nchw_to_hwc4[i], trt_model->cuda_module, nchw_to_hwc4_names[i]);
        if (err != CUDA_SUCCESS) {
            av_log(log_ctx, AV_LOG_ERROR, "Failed to get kernel %s: %s\n", nchw_to_hwc4_names[i], cuda_error_string(err));
            p_cuModuleUnload(trt_model->cuda_module);
            trt_model->cuda_module = NULL;
            return AVERROR(ENOSYS);
        }
    }

    av_log(log_ctx, AV_LOG_INFO, "CUDA format conversion kernels loaded (FP32/FP16/BF16)\n");
    return 0;
}

// Launch kernel with Driver API
static int launch_kernel(CUfunction func, CUstream stream,
                         int width, int height, void **args, void *log_ctx) {
    // 32x8 thread blocks (better warp utilization for row-major image access)
    unsigned int block_x = 32, block_y = 8;
    unsigned int grid_x = (width + block_x - 1) / block_x;
    unsigned int grid_y = (height + block_y - 1) / block_y;

    CUresult err = p_cuLaunchKernel(func,
                                     grid_x, grid_y, 1,      // grid dimensions
                                     block_x, block_y, 1,    // block dimensions
                                     0,                       // shared memory
                                     stream,                  // stream
                                     args,                    // kernel arguments
                                     NULL);                   // extra
    if (err != CUDA_SUCCESS) {
        av_log(log_ctx, AV_LOG_ERROR, "Kernel launch failed: %s\n", cuda_error_string(err));
        return AVERROR(EIO);
    }
    return 0;
}

// Lazily create execution context on first inference
// Returns 0 on success, negative AVERROR on failure
static int ensure_execution_context(TRTModel *trt_model, void *log_ctx)
{
    if (trt_model->context)
        return 0;  // Already created

    av_log(log_ctx, AV_LOG_INFO, "Creating TensorRT execution context (lazy init)\n");

    trt_model->context = trt_model->engine->createExecutionContext();
    if (!trt_model->context) {
        av_log(log_ctx, AV_LOG_ERROR, "Failed to create execution context\n");
        return AVERROR(ENOMEM);
    }
    log_gpu_memory(log_ctx, "after createExecutionContext");

    // Allocate GPU buffers now that we have context
    if (!trt_model->input_buffer) {
        void *input_ptr = NULL, *output_ptr = NULL;
        cudaError_t cuda_err = p_cudaMalloc(&input_ptr, trt_model->input_size);
        if (cuda_err != cudaSuccess) {
            av_log(log_ctx, AV_LOG_ERROR, "cudaMalloc failed for input buffer: %d\n", cuda_err);
            delete trt_model->context;
            trt_model->context = nullptr;
            return AVERROR(ENOMEM);
        }
        trt_model->input_buffer = (CUdeviceptr)input_ptr;

        cuda_err = p_cudaMalloc(&output_ptr, trt_model->output_size);
        if (cuda_err != cudaSuccess) {
            av_log(log_ctx, AV_LOG_ERROR, "cudaMalloc failed for output buffer: %d\n", cuda_err);
            p_cudaFree((void*)trt_model->input_buffer);
            trt_model->input_buffer = 0;
            delete trt_model->context;
            trt_model->context = nullptr;
            return AVERROR(ENOMEM);
        }
        trt_model->output_buffer = (CUdeviceptr)output_ptr;

        // Set tensor addresses
        if (!trt_model->context->setTensorAddress(trt_model->input_name, (void*)trt_model->input_buffer)) {
            av_log(log_ctx, AV_LOG_ERROR, "Failed to set input tensor address\n");
            p_cudaFree((void*)trt_model->input_buffer);
            p_cudaFree((void*)trt_model->output_buffer);
            trt_model->input_buffer = 0;
            trt_model->output_buffer = 0;
            delete trt_model->context;
            trt_model->context = nullptr;
            return AVERROR(EINVAL);
        }
        if (!trt_model->context->setTensorAddress(trt_model->output_name, (void*)trt_model->output_buffer)) {
            av_log(log_ctx, AV_LOG_ERROR, "Failed to set output tensor address\n");
            p_cudaFree((void*)trt_model->input_buffer);
            p_cudaFree((void*)trt_model->output_buffer);
            trt_model->input_buffer = 0;
            trt_model->output_buffer = 0;
            delete trt_model->context;
            trt_model->context = nullptr;
            return AVERROR(EINVAL);
        }

        av_log(log_ctx, AV_LOG_INFO, "  Allocated GPU buffers: input=%zuMB output=%zuMB\n",
               trt_model->input_size / (1024 * 1024), trt_model->output_size / (1024 * 1024));
        log_gpu_memory(log_ctx, "after buffer allocation");
    }

    return 0;
}

static int extract_lltask_from_task(TaskItem *task, Queue *lltask_queue)
{
    TRTModel *trt_model = (TRTModel *)task->model;
    DnnContext *ctx = trt_model->ctx;
    LastLevelTaskItem *lltask = (LastLevelTaskItem *)av_malloc(sizeof(*lltask));
    if (!lltask) {
        av_log(ctx, AV_LOG_ERROR, "Failed to allocate memory for LastLevelTaskItem\n");
        return AVERROR(ENOMEM);
    }
    task->inference_todo = 1;
    task->inference_done = 0;
    lltask->task = task;
    if (ff_queue_push_back(lltask_queue, lltask) < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to push back lltask_queue.\n");
        av_freep(&lltask);
        return AVERROR(ENOMEM);
    }
    return 0;
}

static void trt_free_request(TRTInferRequest *request)
{
    if (!request)
        return;
    if (request->output_data) {
        av_freep(&request->output_data);
    }
}

static inline void destroy_request_item(TRTRequestItem **arg)
{
    TRTRequestItem *item;
    if (!arg || !*arg)
        return;
    item = *arg;
    trt_free_request(item->infer_request);
    av_freep(&item->infer_request);
    av_freep(&item->lltask);
    ff_dnn_async_module_cleanup(&item->exec_module);
    av_freep(arg);
}

static void dnn_free_model_trt(DNNModel **model)
{
    TRTModel *trt_model;
    if (!model || !*model)
        return;

    trt_model = (TRTModel *)(*model);

    // Synchronize stream before cleanup to ensure all GPU operations complete
    if (trt_model->stream && p_cuStreamSynchronize) {
        p_cuStreamSynchronize(trt_model->stream);
    }

    // Free CUDA Graph resources
    if (trt_model->cuda_graph_exec && p_cuGraphExecDestroy) {
        p_cuGraphExecDestroy(trt_model->cuda_graph_exec);
        trt_model->cuda_graph_exec = NULL;
    }
    if (trt_model->cuda_graph && p_cuGraphDestroy) {
        p_cuGraphDestroy(trt_model->cuda_graph);
        trt_model->cuda_graph = NULL;
    }

    // Free CUDA resources (using Runtime API - must match cudaMalloc allocation)
    if (trt_model->input_buffer && p_cudaFree) {
        p_cudaFree((void*)trt_model->input_buffer);
        trt_model->input_buffer = 0;
    }
    if (trt_model->output_buffer && p_cudaFree) {
        p_cudaFree((void*)trt_model->output_buffer);
        trt_model->output_buffer = 0;
    }
    if (trt_model->cuda_module && p_cuModuleUnload) {
        p_cuModuleUnload(trt_model->cuda_module);
        trt_model->cuda_module = NULL;
    }
    if (trt_model->stream && p_cuStreamDestroy) {
        p_cuStreamDestroy(trt_model->stream);
        trt_model->stream = NULL;
    }

    // Free tensor names (engine_path freed after cache cleanup)
    av_freep(&trt_model->input_name);
    av_freep(&trt_model->output_name);

    // Free TensorRT resources
    if (trt_model->context) {
        delete trt_model->context;
        trt_model->context = nullptr;
    }

    // Handle cached engine - decrement refcount and only free when last reference
    if (trt_model->cached_engine) {
        std::lock_guard<std::mutex> lock(g_engine_cache_mutex);
        // Use atomic fetch_sub to avoid race condition - returns OLD value
        int old_refcount = trt_model->cached_engine->refcount.fetch_sub(1, std::memory_order_acq_rel);
        int remaining = old_refcount - 1;
        av_log(trt_model->ctx, AV_LOG_DEBUG, "Engine refcount: %d -> %d (path=%s)\n",
               old_refcount, remaining, trt_model->engine_path ? trt_model->engine_path : "null");
        if (remaining == 0) {
            // Last reference - remove from cache and delete
            if (trt_model->engine_path) {
                std::string path_key(trt_model->engine_path);
                size_t erased = g_engine_cache.erase(path_key);
                av_log(trt_model->ctx, AV_LOG_DEBUG, "Erased %zu entries from cache\n", erased);
            }
            if (trt_model->cached_engine->engine) {
                delete trt_model->cached_engine->engine;
            }
            if (trt_model->cached_engine->runtime) {
                delete trt_model->cached_engine->runtime;
            }
            delete trt_model->cached_engine;
            av_log(trt_model->ctx, AV_LOG_DEBUG, "Released last reference to cached engine\n");
        } else if (remaining < 0) {
            av_log(trt_model->ctx, AV_LOG_ERROR, "BUG: Engine refcount went negative (%d)!\n", remaining);
        } else {
            av_log(trt_model->ctx, AV_LOG_DEBUG, "Released engine reference (remaining=%d)\n", remaining);
        }
        trt_model->cached_engine = nullptr;
        trt_model->engine = nullptr;
        trt_model->runtime = nullptr;
    } else {
        // Not cached (shouldn't happen normally, but handle gracefully)
        if (trt_model->engine) {
            delete trt_model->engine;
            trt_model->engine = nullptr;
        }
        if (trt_model->runtime) {
            delete trt_model->runtime;
            trt_model->runtime = nullptr;
        }
    }

    if (trt_model->logger) {
        delete trt_model->logger;
        trt_model->logger = nullptr;
    }

    // Free engine path (after cache cleanup which uses it)
    av_freep(&trt_model->engine_path);

    // Free queues
    if (trt_model->request_queue) {
        while (ff_safe_queue_size(trt_model->request_queue) != 0) {
            TRTRequestItem *item = (TRTRequestItem *)ff_safe_queue_pop_front(trt_model->request_queue);
            destroy_request_item(&item);
        }
        ff_safe_queue_destroy(trt_model->request_queue);
    }
    if (trt_model->lltask_queue) {
        while (ff_queue_size(trt_model->lltask_queue) != 0) {
            LastLevelTaskItem *item = (LastLevelTaskItem *)ff_queue_pop_front(trt_model->lltask_queue);
            av_freep(&item);
        }
        ff_queue_destroy(trt_model->lltask_queue);
    }
    if (trt_model->task_queue) {
        while (ff_queue_size(trt_model->task_queue) != 0) {
            TaskItem *item = (TaskItem *)ff_queue_pop_front(trt_model->task_queue);
            av_frame_free(&item->in_frame);
            av_frame_free(&item->out_frame);
            av_freep(&item);
        }
        ff_queue_destroy(trt_model->task_queue);
    }

    av_freep(&trt_model);
    *model = NULL;
}

static int get_input_trt(DNNModel *model, DNNData *input, const char *input_name)
{
    TRTModel *trt_model = (TRTModel *)model;

    // Validate tensor has expected dimensions (NCHW = 4)
    if (trt_model->input_dims.nbDims != 4) {
        av_log(trt_model->ctx, AV_LOG_ERROR,
               "Expected 4D input tensor (NCHW), got %d dimensions\n",
               trt_model->input_dims.nbDims);
        return AVERROR(EINVAL);
    }

    input->dt = DNN_FLOAT;
    input->order = DCO_RGB;
    input->layout = DL_NCHW;

    // Get dimensions from engine
    input->dims[0] = trt_model->input_dims.d[0];  // N (batch)
    input->dims[1] = trt_model->input_dims.d[1];  // C (channels)
    input->dims[2] = trt_model->input_dims.d[2];  // H (height)
    input->dims[3] = trt_model->input_dims.d[3];  // W (width)

    return 0;
}

static int fill_model_input_trt(TRTModel *trt_model, TRTRequestItem *request)
{
    LastLevelTaskItem *lltask = NULL;
    TaskItem *task = NULL;
    DNNData input = { 0 };
    DnnContext *ctx = trt_model->ctx;
    int ret;

    // Ensure execution context and buffers are created (lazy initialization)
    ret = ensure_execution_context(trt_model, ctx);
    if (ret < 0) {
        return ret;
    }

    lltask = (LastLevelTaskItem *)ff_queue_pop_front(trt_model->lltask_queue);
    if (!lltask) {
        return AVERROR(EINVAL);
    }
    request->lltask = lltask;
    task = lltask->task;

    ret = get_input_trt(&trt_model->model, &input, NULL);
    if (ret != 0) {
        return ret;
    }

    int height_idx = dnn_get_height_idx_by_layout(input.layout);
    int width_idx = dnn_get_width_idx_by_layout(input.layout);

    // Check input dimensions match engine
    if (task->in_frame->height != input.dims[height_idx] ||
        task->in_frame->width != input.dims[width_idx]) {
        av_log(ctx, AV_LOG_ERROR, "Input size %dx%d doesn't match engine's expected %dx%d\n",
               task->in_frame->width, task->in_frame->height,
               input.dims[width_idx], input.dims[height_idx]);
        return AVERROR(EINVAL);
    }

    int width = task->in_frame->width;
    int height = task->in_frame->height;

    // Check for CUDA hardware frames (zero-copy input path)
    if (task->in_frame->format == AV_PIX_FMT_CUDA && task->in_frame->hw_frames_ctx) {
        AVHWFramesContext *hw_frames = (AVHWFramesContext *)task->in_frame->hw_frames_ctx->data;
        int linesize = task->in_frame->linesize[0];
        CUdeviceptr cuda_data = (CUdeviceptr)task->in_frame->data[0];
        int dtype_idx = (int)trt_model->input_dtype;  // Kernel array index: 0=FP32, 1=FP16, 2=BF16

        // For RGB24/BGR24: convert uint8 HWC to NCHW on GPU (zero-copy)
        if (hw_frames->sw_format == AV_PIX_FMT_RGB24 || hw_frames->sw_format == AV_PIX_FMT_BGR24) {
            void *args[] = {&cuda_data, &trt_model->input_buffer, &height, &width, &linesize};
            ret = launch_kernel(trt_model->kernel_hwc_to_nchw[dtype_idx], trt_model->stream,
                               width, height, args, ctx);
            if (ret != 0) return ret;
            return 0;
        }

        // For 4-channel formats (RGB0, RGBA, BGR0, BGRA)
        if (hw_frames->sw_format == AV_PIX_FMT_RGB0 || hw_frames->sw_format == AV_PIX_FMT_BGR0 ||
            hw_frames->sw_format == AV_PIX_FMT_RGBA || hw_frames->sw_format == AV_PIX_FMT_BGRA) {
            int r_off = 0, g_off = 1, b_off = 2;
            if (hw_frames->sw_format == AV_PIX_FMT_BGR0 || hw_frames->sw_format == AV_PIX_FMT_BGRA) {
                r_off = 2; b_off = 0;
            }
            void *args[] = {&cuda_data, &trt_model->input_buffer, &height, &width, &linesize,
                           &r_off, &g_off, &b_off};
            ret = launch_kernel(trt_model->kernel_hwc4_to_nchw[dtype_idx], trt_model->stream,
                               width, height, args, ctx);
            if (ret != 0) return ret;
            return 0;
        }

        // For 0RGB/ARGB formats (alpha first)
        if (hw_frames->sw_format == AV_PIX_FMT_0RGB || hw_frames->sw_format == AV_PIX_FMT_0BGR ||
            hw_frames->sw_format == AV_PIX_FMT_ARGB || hw_frames->sw_format == AV_PIX_FMT_ABGR) {
            int r_off = 1, g_off = 2, b_off = 3;
            if (hw_frames->sw_format == AV_PIX_FMT_0BGR || hw_frames->sw_format == AV_PIX_FMT_ABGR) {
                r_off = 3; b_off = 1;
            }
            void *args[] = {&cuda_data, &trt_model->input_buffer, &height, &width, &linesize,
                           &r_off, &g_off, &b_off};
            ret = launch_kernel(trt_model->kernel_hwc4_to_nchw[dtype_idx], trt_model->stream,
                               width, height, args, ctx);
            if (ret != 0) return ret;
            return 0;
        }

        av_log(ctx, AV_LOG_WARNING, "CUDA sw_format %s not supported for zero-copy, using CPU path\n",
               av_get_pix_fmt_name(hw_frames->sw_format));
    }

    // Standard CPU path - only supports FP32 engines
    // For FP16/BF16, use CUDA hw frames for zero-copy path
    if (trt_model->input_dtype != TRT_DT_FLOAT32) {
        av_log(ctx, AV_LOG_ERROR, "CPU input path only supports FP32 engines, got %s. "
               "Use hwupload to provide CUDA frames for FP16/BF16 zero-copy.\n",
               trt_dtype_name(trt_model->input_dtype));
        return AVERROR(ENOSYS);
    }

    size_t input_elements = input.dims[0] * input.dims[1] * input.dims[2] * input.dims[3];
    float *input_data = (float *)av_malloc(input_elements * sizeof(float));
    if (!input_data) {
        return AVERROR(ENOMEM);
    }

    input.data = input_data;
    input.scale = 255;

    switch (trt_model->model.func_type) {
    case DFT_PROCESS_FRAME:
        if (task->do_ioproc) {
            if (trt_model->model.frame_pre_proc != NULL) {
                trt_model->model.frame_pre_proc(task->in_frame, &input, trt_model->model.filter_ctx);
            } else {
                ff_proc_from_frame_to_dnn(task->in_frame, &input, ctx);
            }
        }
        break;
    default:
        av_log(ctx, AV_LOG_ERROR, "Unsupported model function type %d\n", trt_model->model.func_type);
        av_freep(&input_data);
        return AVERROR(EINVAL);
    }

    // Copy input to GPU using CUDA Runtime API (compatible with TensorRT's Runtime API context)
    cudaError_t cuda_err = p_cudaMemcpy((void*)trt_model->input_buffer, input_data,
                                         trt_model->input_size, cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        av_log(ctx, AV_LOG_ERROR, "cudaMemcpy failed for input: %d\n", cuda_err);
        av_freep(&input_data);
        return AVERROR(EIO);
    }

    av_freep(&input_data);
    return 0;
}

// Capture TensorRT inference into a CUDA Graph for reduced kernel launch overhead
// Returns 0 on success, negative on failure (non-fatal, falls back to regular enqueue)
static int trt_capture_cuda_graph(TRTModel *trt_model, void *log_ctx)
{
    CUresult err;

    if (!cuda_graphs_available || trt_model->cuda_graph_failed) {
        return -1;
    }

    av_log(log_ctx, AV_LOG_INFO, "Capturing TensorRT inference into CUDA Graph...\n");

    // Synchronize stream before capture to ensure any pending work (e.g., input kernel) completes
    // This prevents undefined behavior from capturing a stream with pending async operations
    err = p_cuStreamSynchronize(trt_model->stream);
    if (err != CUDA_SUCCESS) {
        av_log(log_ctx, AV_LOG_WARNING, "Stream sync before graph capture failed: %s\n", cuda_error_string(err));
        trt_model->cuda_graph_failed = 1;
        return -1;
    }

    // Begin stream capture
    err = p_cuStreamBeginCapture(trt_model->stream, CU_STREAM_CAPTURE_MODE_GLOBAL);
    if (err != CUDA_SUCCESS) {
        av_log(log_ctx, AV_LOG_WARNING, "CUDA Graph capture begin failed: %s\n", cuda_error_string(err));
        trt_model->cuda_graph_failed = 1;
        return -1;
    }

    // Execute TensorRT inference (this gets captured into the graph)
    bool success = trt_model->context->enqueueV3((cudaStream_t)trt_model->stream);
    if (!success) {
        // End capture to clean up, ignore the graph
        CUgraph temp_graph = NULL;
        p_cuStreamEndCapture(trt_model->stream, &temp_graph);
        if (temp_graph) p_cuGraphDestroy(temp_graph);
        av_log(log_ctx, AV_LOG_WARNING, "TensorRT inference failed during graph capture\n");
        trt_model->cuda_graph_failed = 1;
        return -1;
    }

    // End stream capture
    err = p_cuStreamEndCapture(trt_model->stream, &trt_model->cuda_graph);
    if (err != CUDA_SUCCESS || !trt_model->cuda_graph) {
        av_log(log_ctx, AV_LOG_WARNING, "CUDA Graph capture end failed: %s\n", cuda_error_string(err));
        trt_model->cuda_graph_failed = 1;
        return -1;
    }

    // Instantiate the graph for execution
    err = p_cuGraphInstantiate(&trt_model->cuda_graph_exec, trt_model->cuda_graph, 0);
    if (err != CUDA_SUCCESS || !trt_model->cuda_graph_exec) {
        av_log(log_ctx, AV_LOG_WARNING, "CUDA Graph instantiate failed: %s\n", cuda_error_string(err));
        p_cuGraphDestroy(trt_model->cuda_graph);
        trt_model->cuda_graph = NULL;
        trt_model->cuda_graph_failed = 1;
        return -1;
    }

    trt_model->cuda_graph_captured = 1;
    av_log(log_ctx, AV_LOG_INFO, "CUDA Graph captured successfully (reduced kernel launch overhead)\n");
    return 0;
}

static int trt_start_inference(void *args)
{
    TRTRequestItem *request = (TRTRequestItem *)args;
    LastLevelTaskItem *lltask;
    TaskItem *task;
    TRTModel *trt_model;
    DnnContext *ctx;

    if (!request || !request->lltask) {
        av_log(NULL, AV_LOG_ERROR, "TRTRequestItem or lltask is NULL\n");
        return AVERROR(EINVAL);
    }
    lltask = request->lltask;
    task = lltask->task;
    trt_model = (TRTModel *)task->model;
    ctx = trt_model->ctx;

    // Validate required resources exist
    if (!trt_model->context || !trt_model->stream) {
        av_log(ctx, AV_LOG_ERROR, "TensorRT context or CUDA stream not initialized\n");
        return DNN_GENERIC_ERROR;
    }

    // NOTE: Tensor addresses are set once during model load (not per-frame)
    // since input/output buffers are persistent

    // Try to use CUDA Graph for reduced kernel launch overhead
    // First frame: capture the graph; subsequent frames: launch the captured graph
    if (cuda_graphs_available && !trt_model->cuda_graph_captured && !trt_model->cuda_graph_failed) {
        // Capture on first inference
        if (trt_capture_cuda_graph(trt_model, ctx) == 0) {
            // Graph captured - we already ran inference during capture, so return
            return 0;
        }
        // Capture failed - fall through to regular execution
    }

    if (trt_model->cuda_graph_captured && trt_model->cuda_graph_exec) {
        // Execute the captured graph (much lower overhead than enqueueV3)
        CUresult err = p_cuGraphLaunch(trt_model->cuda_graph_exec, trt_model->stream);
        if (err != CUDA_SUCCESS) {
            av_log(ctx, AV_LOG_ERROR, "CUDA Graph launch failed: %s\n", cuda_error_string(err));
            return DNN_GENERIC_ERROR;
        }
    } else {
        // Regular execution path (fallback if graphs not available or capture failed)
        bool success = trt_model->context->enqueueV3((cudaStream_t)trt_model->stream);
        if (!success) {
            av_log(ctx, AV_LOG_ERROR, "TensorRT inference failed\n");
            return DNN_GENERIC_ERROR;
        }
    }

    // NOTE: No sync here - for zero-copy paths, we sync once after the output kernel
    // For CPU paths, we sync before cudaMemcpy DtoH in infer_completion_callback

    return 0;
}

static void infer_completion_callback(void *args)
{
    TRTRequestItem *request = (TRTRequestItem *)args;
    LastLevelTaskItem *lltask = request->lltask;
    TaskItem *task = lltask->task;
    TRTModel *trt_model = (TRTModel *)task->model;
    DnnContext *ctx = trt_model->ctx;
    DNNData outputs = { 0 };
    float *output_data = NULL;
    size_t output_elements;
    int ret;

    // Output dimensions are validated during model loading, safe to access
    outputs.order = DCO_RGB;
    outputs.layout = DL_NCHW;
    outputs.dt = DNN_FLOAT;
    outputs.dims[0] = trt_model->output_dims.d[0];  // N
    outputs.dims[1] = trt_model->output_dims.d[1];  // C
    outputs.dims[2] = trt_model->output_dims.d[2];  // H
    outputs.dims[3] = trt_model->output_dims.d[3];  // W

    int out_height = outputs.dims[2];
    int out_width = outputs.dims[3];

    // Validate stream exists (should always be true if model loaded successfully)
    if (!trt_model->stream) {
        av_log(ctx, AV_LOG_ERROR, "CUDA stream is NULL\n");
        goto err;
    }

    // Check for CUDA output frames (zero-copy output path)
    if (task->out_frame->format == AV_PIX_FMT_CUDA && task->out_frame->hw_frames_ctx) {
        AVHWFramesContext *hw_frames = (AVHWFramesContext *)task->out_frame->hw_frames_ctx->data;
        int out_linesize = task->out_frame->linesize[0];
        CUdeviceptr cuda_out = (CUdeviceptr)task->out_frame->data[0];
        int dtype_idx = (int)trt_model->output_dtype;  // Kernel array index: 0=FP32, 1=FP16, 2=BF16

        // For RGB24/BGR24: convert NCHW to uint8 HWC on GPU (zero-copy)
        if (hw_frames->sw_format == AV_PIX_FMT_RGB24 || hw_frames->sw_format == AV_PIX_FMT_BGR24) {
            void *args[] = {&trt_model->output_buffer, &cuda_out, &out_height, &out_width, &out_linesize};
            ret = launch_kernel(trt_model->kernel_nchw_to_hwc[dtype_idx], trt_model->stream,
                               out_width, out_height, args, ctx);
            if (ret != 0) goto err;

            if (p_cuStreamSynchronize(trt_model->stream) != CUDA_SUCCESS) {
                av_log(ctx, AV_LOG_ERROR, "CUDA stream sync failed\n");
                goto err;
            }

            task->out_frame->width = out_width;
            task->out_frame->height = out_height;
            task->inference_done++;
            goto done;
        }

        // For 4-channel formats (RGB0, RGBA, BGR0, BGRA)
        if (hw_frames->sw_format == AV_PIX_FMT_RGB0 || hw_frames->sw_format == AV_PIX_FMT_BGR0 ||
            hw_frames->sw_format == AV_PIX_FMT_RGBA || hw_frames->sw_format == AV_PIX_FMT_BGRA) {
            int r_off = 0, g_off = 1, b_off = 2, a_off = 3;
            if (hw_frames->sw_format == AV_PIX_FMT_BGR0 || hw_frames->sw_format == AV_PIX_FMT_BGRA) {
                r_off = 2; b_off = 0;
            }
            void *args[] = {&trt_model->output_buffer, &cuda_out, &out_height, &out_width, &out_linesize,
                           &r_off, &g_off, &b_off, &a_off};
            ret = launch_kernel(trt_model->kernel_nchw_to_hwc4[dtype_idx], trt_model->stream,
                               out_width, out_height, args, ctx);
            if (ret != 0) goto err;

            if (p_cuStreamSynchronize(trt_model->stream) != CUDA_SUCCESS) {
                av_log(ctx, AV_LOG_ERROR, "CUDA stream sync failed\n");
                goto err;
            }

            task->out_frame->width = out_width;
            task->out_frame->height = out_height;
            task->inference_done++;
            goto done;
        }

        // For 0RGB/ARGB formats (alpha first)
        if (hw_frames->sw_format == AV_PIX_FMT_0RGB || hw_frames->sw_format == AV_PIX_FMT_0BGR ||
            hw_frames->sw_format == AV_PIX_FMT_ARGB || hw_frames->sw_format == AV_PIX_FMT_ABGR) {
            int r_off = 1, g_off = 2, b_off = 3, a_off = 0;
            if (hw_frames->sw_format == AV_PIX_FMT_0BGR || hw_frames->sw_format == AV_PIX_FMT_ABGR) {
                r_off = 3; b_off = 1;
            }
            void *args[] = {&trt_model->output_buffer, &cuda_out, &out_height, &out_width, &out_linesize,
                           &r_off, &g_off, &b_off, &a_off};
            ret = launch_kernel(trt_model->kernel_nchw_to_hwc4[dtype_idx], trt_model->stream,
                               out_width, out_height, args, ctx);
            if (ret != 0) goto err;

            if (p_cuStreamSynchronize(trt_model->stream) != CUDA_SUCCESS) {
                av_log(ctx, AV_LOG_ERROR, "CUDA stream sync failed\n");
                goto err;
            }

            task->out_frame->width = out_width;
            task->out_frame->height = out_height;
            task->inference_done++;
            goto done;
        }

        av_log(ctx, AV_LOG_WARNING, "CUDA output sw_format %s not supported for zero-copy, using CPU path\n",
               av_get_pix_fmt_name(hw_frames->sw_format));
    }

    // Standard CPU path - only supports FP32 engines
    // For FP16/BF16, use CUDA hw frames for zero-copy path
    if (trt_model->output_dtype != TRT_DT_FLOAT32) {
        av_log(ctx, AV_LOG_ERROR, "CPU output path only supports FP32 engines, got %s. "
               "Use hwupload to provide CUDA frames for FP16/BF16 zero-copy.\n",
               trt_dtype_name(trt_model->output_dtype));
        goto err;
    }

    output_elements = outputs.dims[0] * outputs.dims[1] * outputs.dims[2] * outputs.dims[3];
    output_data = (float *)av_malloc(output_elements * sizeof(float));
    if (!output_data) {
        av_log(ctx, AV_LOG_ERROR, "Failed to allocate output buffer\n");
        goto err;
    }

    // Sync stream before copying (inference runs async on stream)
    if (p_cudaStreamSynchronize_rt) {
        cudaError_t sync_err = p_cudaStreamSynchronize_rt((cudaStream_t)trt_model->stream);
        if (sync_err != cudaSuccess) {
            av_log(ctx, AV_LOG_ERROR, "Stream sync failed before output copy: %d\n", sync_err);
            av_freep(&output_data);
            goto err;
        }
    } else {
        // Fallback to Driver API sync
        CUresult err = p_cuStreamSynchronize(trt_model->stream);
        if (err != CUDA_SUCCESS) {
            av_log(ctx, AV_LOG_ERROR, "Stream sync failed: %s\n", cuda_error_string(err));
            av_freep(&output_data);
            goto err;
        }
    }

    // Copy output from GPU using CUDA Runtime API
    {
        cudaError_t cuda_err = p_cudaMemcpy(output_data, (void*)trt_model->output_buffer,
                                             trt_model->output_size, cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) {
            av_log(ctx, AV_LOG_ERROR, "cudaMemcpy failed for output: %d\n", cuda_err);
            av_freep(&output_data);
            goto err;
        }
    }

    switch (trt_model->model.func_type) {
    case DFT_PROCESS_FRAME:
        if (task->do_ioproc) {
            outputs.scale = 255;
            outputs.data = output_data;
            if (trt_model->model.frame_post_proc != NULL) {
                trt_model->model.frame_post_proc(task->out_frame, &outputs, trt_model->model.filter_ctx);
            } else {
                ff_proc_from_dnn_to_frame(task->out_frame, &outputs, ctx);
            }
        } else {
            task->out_frame->width = out_width;
            task->out_frame->height = out_height;
        }
        break;
    default:
        av_log(ctx, AV_LOG_ERROR, "Unsupported model function type %d\n", trt_model->model.func_type);
        av_freep(&output_data);
        goto err;
    }

    av_freep(&output_data);
    task->inference_done++;
    goto done;

err:
    // Increment inference_done even on error so task completion tracking works
    // The caller can detect failure through other means (e.g., frame validation)
    task->inference_done++;

done:
    av_freep(&request->lltask);
    if (ff_safe_queue_push_back(trt_model->request_queue, request) < 0) {
        destroy_request_item(&request);
        av_log(ctx, AV_LOG_ERROR, "Unable to push back request_queue.\n");
    }
}

static int execute_model_trt(TRTRequestItem *request, Queue *lltask_queue)
{
    TRTModel *trt_model = NULL;
    LastLevelTaskItem *lltask;
    TaskItem *task = NULL;
    int ret = 0;

    if (ff_queue_size(lltask_queue) == 0) {
        destroy_request_item(&request);
        return 0;
    }

    lltask = (LastLevelTaskItem *)ff_queue_peek_front(lltask_queue);
    if (lltask == NULL) {
        av_log(NULL, AV_LOG_ERROR, "Failed to get LastLevelTaskItem\n");
        ret = AVERROR(EINVAL);
        goto err;
    }
    task = lltask->task;
    trt_model = (TRTModel *)task->model;

    ret = fill_model_input_trt(trt_model, request);
    if (ret != 0) {
        goto err;
    }

    // Synchronous execution (TensorRT is fast, async adds complexity)
    ret = trt_start_inference((void *)request);
    if (ret != 0) {
        goto err;
    }
    infer_completion_callback(request);
    return (task->inference_done == task->inference_todo) ? 0 : DNN_GENERIC_ERROR;

err:
    trt_free_request(request->infer_request);
    av_freep(&request->lltask);  // Free lltask that was popped from queue
    // Clean up the task that was left in task_queue to prevent memory leak
    // The task is at the back since we just pushed it in dnn_execute_model_trt
    if (trt_model && task) {
        // Remove task from queue - it should be the last one we added
        // Iterate to find and remove it (safer than assuming position)
        Queue *tq = trt_model->task_queue;
        size_t queue_size = ff_queue_size(tq);
        for (size_t i = 0; i < queue_size; i++) {
            TaskItem *queued_task = (TaskItem *)ff_queue_peek_front(tq);
            if (queued_task == task) {
                ff_queue_pop_front(tq);
                av_frame_free(&task->in_frame);
                av_frame_free(&task->out_frame);
                av_freep(&task);
                break;
            }
            // Move to next by popping and re-pushing (rotate queue)
            ff_queue_pop_front(tq);
            ff_queue_push_back(tq, queued_task);
        }
    }
    if (!trt_model || ff_safe_queue_push_back(trt_model->request_queue, request) < 0) {
        destroy_request_item(&request);
    }
    return ret;
}

static int get_output_trt(DNNModel *model, const char *input_name, int input_width, int input_height,
                          const char *output_name, int *output_width, int *output_height)
{
    TRTModel *trt_model = (TRTModel *)model;

    // Get from engine's output dimensions
    *output_width = trt_model->output_dims.d[3];
    *output_height = trt_model->output_dims.d[2];

    return 0;
}

static TRTInferRequest *trt_create_inference_request(void)
{
    TRTInferRequest *request = (TRTInferRequest *)av_mallocz(sizeof(TRTInferRequest));
    return request;
}

static DNNModel *dnn_load_model_trt(DnnContext *ctx, DNNFunctionType func_type, AVFilterContext *filter_ctx)
{
    TRTModel *trt_model = NULL;
    TRTRequestItem *item = NULL;
    CUresult err;

    trt_model = (TRTModel *)av_mallocz(sizeof(TRTModel));
    if (!trt_model)
        return NULL;

    trt_model->ctx = ctx;

    // Load CUDA and TensorRT libraries via dlopen
    if (load_libs(ctx) < 0) {
        goto fail;
    }

    log_gpu_memory(ctx, "after load_libs");

    // Set CUDA device using Runtime API for TensorRT compatibility
    if (p_cudaSetDevice) {
        int device_id = ctx->trt_option.device_id;
        cudaError_t cuda_err = p_cudaSetDevice(device_id);
        if (cuda_err != cudaSuccess) {
            av_log(ctx, AV_LOG_ERROR, "cudaSetDevice(%d) failed: %d\n", device_id, cuda_err);
            goto fail;
        }
        av_log(ctx, AV_LOG_DEBUG, "Set CUDA device %d for TensorRT\n", device_id);
    }

    // Create TensorRT logger (cleaned up by dnn_free_model_trt on any failure path)
    trt_model->logger = new TRTLogger(ctx);

    // Check engine cache first (avoid reloading same engine file)
    trt_model->engine_path = av_strdup(ctx->model_filename);
    if (!trt_model->engine_path) {
        av_log(ctx, AV_LOG_ERROR, "Failed to allocate engine path\n");
        goto fail;
    }
    {
        std::lock_guard<std::mutex> lock(g_engine_cache_mutex);
        std::string path_key(trt_model->engine_path);
        av_log(ctx, AV_LOG_DEBUG, "Checking engine cache for: %s (cache size=%zu)\n",
               trt_model->engine_path, g_engine_cache.size());
        auto it = g_engine_cache.find(path_key);
        if (it != g_engine_cache.end()) {
            // Found in cache - reuse existing engine
            trt_model->cached_engine = it->second;
            if (!trt_model->cached_engine->engine || !trt_model->cached_engine->runtime) {
                av_log(ctx, AV_LOG_ERROR, "BUG: Cached engine has NULL pointers! Removing stale entry.\n");
                g_engine_cache.erase(it);
                trt_model->cached_engine = nullptr;
            } else {
                // Use atomic fetch_add to avoid race condition
                int new_refcount = trt_model->cached_engine->refcount.fetch_add(1, std::memory_order_acq_rel) + 1;
                trt_model->engine = trt_model->cached_engine->engine;
                trt_model->runtime = trt_model->cached_engine->runtime;
                av_log(ctx, AV_LOG_INFO, "Reusing cached TensorRT engine (refcount=%d, engine=%p)\n",
                       new_refcount, (void*)trt_model->engine);
            }
        }
        if (!trt_model->cached_engine) {
            av_log(ctx, AV_LOG_DEBUG, "Engine not in cache, will load from file\n");
        }
    }

    // If not in cache, load engine from file
    if (!trt_model->engine) {
        // Create runtime using dynamically loaded function
        trt_model->runtime = p_createInferRuntime(*trt_model->logger);
        if (!trt_model->runtime) {
            av_log(ctx, AV_LOG_ERROR, "Failed to create TensorRT runtime\n");
            goto fail;
        }

        // Load engine from file
        {
            std::ifstream file(ctx->model_filename, std::ios::binary | std::ios::ate);
            if (!file.is_open()) {
                av_log(ctx, AV_LOG_ERROR, "Failed to open engine file: %s\n", ctx->model_filename);
                goto fail;
            }

            std::streampos pos = file.tellg();
            if (pos == std::streampos(-1) || pos <= 0) {
                av_log(ctx, AV_LOG_ERROR, "Engine file is empty or unreadable: %s\n", ctx->model_filename);
                goto fail;
            }
            size_t size = static_cast<size_t>(pos);
            file.seekg(0, std::ios::beg);

            std::vector<char> buffer(size);
            if (!file.read(buffer.data(), size)) {
                av_log(ctx, AV_LOG_ERROR, "Failed to read engine file\n");
                goto fail;
            }

            trt_model->engine = trt_model->runtime->deserializeCudaEngine(buffer.data(), size);
            if (!trt_model->engine) {
                av_log(ctx, AV_LOG_ERROR, "Failed to deserialize CUDA engine\n");
                goto fail;
            }
            log_gpu_memory(ctx, "after engine deserialize");
        }

        // Add to cache
        {
            std::lock_guard<std::mutex> lock(g_engine_cache_mutex);
            std::string path_key(trt_model->engine_path);
            // Double-check another thread didn't add it while we were loading
            auto it = g_engine_cache.find(path_key);
            if (it == g_engine_cache.end()) {
                trt_model->cached_engine = new CachedEngine(trt_model->engine, trt_model->runtime);
                g_engine_cache[path_key] = trt_model->cached_engine;
                av_log(ctx, AV_LOG_INFO, "Added TensorRT engine to cache\n");
            } else {
                // Another thread added it - use theirs, discard ours
                delete trt_model->engine;
                delete trt_model->runtime;
                trt_model->cached_engine = it->second;
                // Use atomic fetch_add to avoid race condition
                int new_refcount = trt_model->cached_engine->refcount.fetch_add(1, std::memory_order_acq_rel) + 1;
                trt_model->engine = trt_model->cached_engine->engine;
                trt_model->runtime = trt_model->cached_engine->runtime;
                av_log(ctx, AV_LOG_INFO, "Using engine added by another thread (refcount=%d)\n",
                       new_refcount);
            }
        }
    }

    // NOTE: Execution context is created lazily on first inference (saves ~720MB for probe instances)
    // FFmpeg creates two filter instances: one for probing (never runs inference) and one for execution
    trt_model->context = nullptr;

    // CUDA Graph state (captured lazily on first inference)
    trt_model->cuda_graph = NULL;
    trt_model->cuda_graph_exec = NULL;
    trt_model->cuda_graph_captured = 0;
    trt_model->cuda_graph_failed = 0;

    // Create CUDA stream for TensorRT operations
    err = p_cuStreamCreate(&trt_model->stream, 0);
    if (err != CUDA_SUCCESS) {
        av_log(ctx, AV_LOG_ERROR, "Failed to create CUDA stream: %s\n", cuda_error_string(err));
        goto fail;
    }

    // Load CUDA kernels from embedded PTX
    if (load_cuda_kernels(trt_model, ctx) < 0) {
        goto fail;
    }
    log_gpu_memory(ctx, "after load_cuda_kernels");

    // Get I/O tensor info (TensorRT 10.x API)
    {
        int nb_io_tensors = trt_model->engine->getNbIOTensors();
        if (nb_io_tensors < 2) {
            av_log(ctx, AV_LOG_ERROR, "Engine must have at least 2 tensors (input and output), got %d\n", nb_io_tensors);
            goto fail;
        }

        // Find input and output tensors
        for (int i = 0; i < nb_io_tensors; i++) {
            const char *name = trt_model->engine->getIOTensorName(i);
            nvinfer1::TensorIOMode mode = trt_model->engine->getTensorIOMode(name);

            if (mode == nvinfer1::TensorIOMode::kINPUT && !trt_model->input_name) {
                trt_model->input_name = av_strdup(name);
                trt_model->input_dims = trt_model->engine->getTensorShape(name);
                trt_model->input_dtype = nvinfer_to_trt_dtype(trt_model->engine->getTensorDataType(name));
            } else if (mode == nvinfer1::TensorIOMode::kOUTPUT && !trt_model->output_name) {
                trt_model->output_name = av_strdup(name);
                trt_model->output_dims = trt_model->engine->getTensorShape(name);
                trt_model->output_dtype = nvinfer_to_trt_dtype(trt_model->engine->getTensorDataType(name));
            }
        }

        if (!trt_model->input_name || !trt_model->output_name) {
            av_log(ctx, AV_LOG_ERROR, "Could not find input/output tensors\n");
            goto fail;
        }

        // Validate dtypes are supported
        if (trt_model->input_dtype == TRT_DT_UNKNOWN) {
            av_log(ctx, AV_LOG_ERROR, "Unsupported input tensor data type\n");
            goto fail;
        }
        if (trt_model->output_dtype == TRT_DT_UNKNOWN) {
            av_log(ctx, AV_LOG_ERROR, "Unsupported output tensor data type\n");
            goto fail;
        }
        // For now, we only support FP32/FP16/BF16 for zero-copy kernels
        if (trt_model->input_dtype > TRT_DT_BFLOAT16 || trt_model->output_dtype > TRT_DT_BFLOAT16) {
            av_log(ctx, AV_LOG_ERROR, "Only FP32/FP16/BF16 tensors supported for zero-copy, got input=%s output=%s\n",
                   trt_dtype_name(trt_model->input_dtype), trt_dtype_name(trt_model->output_dtype));
            goto fail;
        }

        // Validate tensor dimensions (must be 4D for NCHW format)
        if (trt_model->input_dims.nbDims != 4) {
            av_log(ctx, AV_LOG_ERROR, "Input tensor must be 4D (NCHW), got %d dimensions\n",
                   trt_model->input_dims.nbDims);
            goto fail;
        }
        if (trt_model->output_dims.nbDims != 4) {
            av_log(ctx, AV_LOG_ERROR, "Output tensor must be 4D (NCHW), got %d dimensions\n",
                   trt_model->output_dims.nbDims);
            goto fail;
        }

        // Validate all dimensions are positive
        for (int i = 0; i < 4; i++) {
            if (trt_model->input_dims.d[i] <= 0) {
                av_log(ctx, AV_LOG_ERROR, "Invalid input dimension[%d] = %ld\n",
                       i, (long)trt_model->input_dims.d[i]);
                goto fail;
            }
            if (trt_model->output_dims.d[i] <= 0) {
                av_log(ctx, AV_LOG_ERROR, "Invalid output dimension[%d] = %ld\n",
                       i, (long)trt_model->output_dims.d[i]);
                goto fail;
            }
        }

        // Log dimensions and dtypes
        av_log(ctx, AV_LOG_INFO, "TensorRT engine loaded:\n");
        av_log(ctx, AV_LOG_INFO, "  Input '%s': %ldx%ldx%ldx%ld (%s)\n",
               trt_model->input_name,
               (long)trt_model->input_dims.d[0], (long)trt_model->input_dims.d[1],
               (long)trt_model->input_dims.d[2], (long)trt_model->input_dims.d[3],
               trt_dtype_name(trt_model->input_dtype));
        av_log(ctx, AV_LOG_INFO, "  Output '%s': %ldx%ldx%ldx%ld (%s)\n",
               trt_model->output_name,
               (long)trt_model->output_dims.d[0], (long)trt_model->output_dims.d[1],
               (long)trt_model->output_dims.d[2], (long)trt_model->output_dims.d[3],
               trt_dtype_name(trt_model->output_dtype));

        // Calculate buffer sizes (allocation deferred to first inference via ensure_execution_context)
        {
            // Cast each factor to int64_t to prevent overflow during multiplication
            int64_t in_elems = (int64_t)trt_model->input_dims.d[0] * (int64_t)trt_model->input_dims.d[1] *
                               (int64_t)trt_model->input_dims.d[2] * (int64_t)trt_model->input_dims.d[3];
            int64_t out_elems = (int64_t)trt_model->output_dims.d[0] * (int64_t)trt_model->output_dims.d[1] *
                                (int64_t)trt_model->output_dims.d[2] * (int64_t)trt_model->output_dims.d[3];

            size_t in_elem_size = trt_dtype_size(trt_model->input_dtype);
            size_t out_elem_size = trt_dtype_size(trt_model->output_dtype);

            // Check for overflow (max reasonable GPU buffer ~16GB)
            const int64_t max_bytes = (int64_t)16 * 1024 * 1024 * 1024;
            if (in_elems * (int64_t)in_elem_size > max_bytes || out_elems * (int64_t)out_elem_size > max_bytes) {
                av_log(ctx, AV_LOG_ERROR, "Tensor size exceeds maximum supported (16GB)\n");
                goto fail;
            }

            trt_model->input_size = (size_t)in_elems * in_elem_size;
            trt_model->output_size = (size_t)out_elems * out_elem_size;

            av_log(ctx, AV_LOG_INFO, "  Buffer sizes (deferred): input=%zuMB output=%zuMB\n",
                   trt_model->input_size / (1024 * 1024), trt_model->output_size / (1024 * 1024));
        }

        // NOTE: GPU buffers and execution context are allocated lazily on first inference
        // This saves ~720MB+ for FFmpeg's probe filter instance that never runs inference
    }

    // Initialize queues
    trt_model->request_queue = ff_safe_queue_create();
    if (!trt_model->request_queue)
        goto fail;

    item = (TRTRequestItem *)av_mallocz(sizeof(TRTRequestItem));
    if (!item)
        goto fail;

    item->infer_request = trt_create_inference_request();
    if (!item->infer_request)
        goto fail;

    item->exec_module.start_inference = &trt_start_inference;
    item->exec_module.callback = &infer_completion_callback;
    item->exec_module.args = item;

    if (ff_safe_queue_push_back(trt_model->request_queue, item) < 0)
        goto fail;
    item = NULL;

    trt_model->task_queue = ff_queue_create();
    if (!trt_model->task_queue)
        goto fail;

    trt_model->lltask_queue = ff_queue_create();
    if (!trt_model->lltask_queue)
        goto fail;

    // Set up model interface
    trt_model->model.get_input = &get_input_trt;
    trt_model->model.get_output = &get_output_trt;
    trt_model->model.filter_ctx = filter_ctx;
    trt_model->model.func_type = func_type;

    return &trt_model->model;

fail:
    if (item) {
        destroy_request_item(&item);
    }
    dnn_free_model_trt((DNNModel **)&trt_model);
    return NULL;
}

static int dnn_execute_model_trt(const DNNModel *model, DNNExecBaseParams *exec_params)
{
    TRTModel *trt_model = (TRTModel *)model;
    DnnContext *ctx = trt_model->ctx;
    TaskItem *task;
    TRTRequestItem *request;
    int ret = 0;

    ret = ff_check_exec_params(ctx, DNN_TRT, model->func_type, exec_params);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "exec parameter checking fail.\n");
        return ret;
    }

    task = (TaskItem *)av_malloc(sizeof(TaskItem));
    if (!task) {
        av_log(ctx, AV_LOG_ERROR, "unable to alloc memory for task item.\n");
        return AVERROR(ENOMEM);
    }

    ret = ff_dnn_fill_task(task, exec_params, trt_model, 0, 1);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "unable to fill task.\n");
        av_frame_free(&task->in_frame);
        av_frame_free(&task->out_frame);
        av_freep(&task);
        return ret;
    }

    ret = ff_queue_push_back(trt_model->task_queue, task);
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "unable to push back task_queue.\n");
        av_frame_free(&task->in_frame);
        av_frame_free(&task->out_frame);
        av_freep(&task);
        return ret;
    }

    ret = extract_lltask_from_task(task, trt_model->lltask_queue);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "unable to extract last level task from task.\n");
        // Remove task from queue since extraction failed
        ff_queue_pop_back(trt_model->task_queue);
        av_frame_free(&task->in_frame);
        av_frame_free(&task->out_frame);
        av_freep(&task);
        return ret;
    }

    request = (TRTRequestItem *)ff_safe_queue_pop_front(trt_model->request_queue);
    if (!request) {
        av_log(ctx, AV_LOG_ERROR, "unable to get infer request.\n");
        // Clean up: remove lltask and task we just added
        LastLevelTaskItem *lltask = (LastLevelTaskItem *)ff_queue_pop_back(trt_model->lltask_queue);
        av_freep(&lltask);
        ff_queue_pop_back(trt_model->task_queue);
        av_frame_free(&task->in_frame);
        av_frame_free(&task->out_frame);
        av_freep(&task);
        return AVERROR(EINVAL);
    }

    return execute_model_trt(request, trt_model->lltask_queue);
}

static DNNAsyncStatusType dnn_get_result_trt(const DNNModel *model, AVFrame **in, AVFrame **out)
{
    TRTModel *trt_model = (TRTModel *)model;
    return ff_dnn_get_result_common(trt_model->task_queue, in, out);
}

static int dnn_flush_trt(const DNNModel *model)
{
    TRTModel *trt_model = (TRTModel *)model;
    TRTRequestItem *request;

    if (ff_queue_size(trt_model->lltask_queue) == 0)
        return 0;

    request = (TRTRequestItem *)ff_safe_queue_pop_front(trt_model->request_queue);
    if (!request) {
        av_log(trt_model->ctx, AV_LOG_ERROR, "unable to get infer request.\n");
        return AVERROR(EINVAL);
    }

    return execute_model_trt(request, trt_model->lltask_queue);
}

extern const DNNModule ff_dnn_backend_tensorrt = {
    .clazz          = DNN_DEFINE_CLASS(dnn_trt),
    .type           = DNN_TRT,
    .load_model     = dnn_load_model_trt,
    .execute_model  = dnn_execute_model_trt,
    .get_result     = dnn_get_result_trt,
    .flush          = dnn_flush_trt,
    .free_model     = dnn_free_model_trt,
};
