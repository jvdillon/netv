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
 *
 * CUDA kernels for DNN backend format conversion.
 * Compiled to PTX at build time, loaded dynamically at runtime.
 * No cudart dependency - uses CUDA Driver API via FFmpeg's dynlink.
 *
 * WHY CUSTOM CUDA KERNELS FOR SUCH A PRIMITIVE OPERATION?
 * ========================================================
 * These kernels convert FFmpeg frames (HWC uint8) to neural network input (NCHW float).
 * This seems like it should be a one-liner, but no standard library handles our needs:
 *
 * What we need (all in one pass, zero-copy from FFmpeg CUDA frames):
 *   1. Handle FFmpeg's linesize padding (row stride != width * channels)
 *   2. Convert uint8 [0,255] to float [0,1] (type conversion + scaling)
 *   3. Transpose HWC to NCHW (layout conversion)
 *   4. Support multiple pixel formats (RGB24, BGR24, RGBA, BGRA, ARGB, etc.)
 *   5. Support multiple tensor types (FP32, FP16, BF16)
 *
 * Why existing libraries don't work:
 *   - cuDNN cudnnTransformTensor: float-to-float layout changes only, no uint8
 *   - NPP (nppiConvert_8u32f): type conversion but no transpose, separate calls
 *   - TensorRT IReformatLayer: layout changes for float, not uint8 ingestion
 *   - Chaining these: multiple kernel launches + intermediate allocations
 *
 * Alternatives considered:
 *   - Preprocessing in ONNX model: Can't handle variable linesize padding
 *   - TensorRT custom plugin: Adds export complexity, less flexible
 *   - NPP + cuBLAS transpose: 2 passes, intermediate buffer, slower
 *
 * The fused kernel approach: one read, one write, no intermediate buffers.
 * It's unfortunate that we need 400 lines of CUDA for "pixels to floats",
 * but this is the reality of bridging FFmpeg's frame formats to ML frameworks.
 *
 * MEMORY ACCESS PATTERN NOTES:
 * ============================
 * The HWC->NCHW conversion writes R,G,B to memory locations separated by H*W elements.
 * This looks like poor cache behavior, but:
 *   - Writes within each channel plane ARE coalesced (adjacent threads write adjacent addresses)
 *   - Modern GPU L2 caches (4-6MB) can hold multiple planes for typical frame sizes
 *   - The strided HWC reads (3 bytes apart) are actually the slower part
 *   - Shared memory staging for better write coalescing was tested but didn't help
 *
 * An elementwise approach (3 separate kernels, one per channel) would:
 *   - Triple kernel launch overhead
 *   - Read the input 3x instead of 1x
 *   - Not improve coalescing (still stride-3 reads)
 */

#include <cuda_fp16.h>
#include <cuda_bf16.h>

extern "C" {

// Precomputed reciprocal for [0,255] -> [0,1] conversion
// Using multiplication is faster than division
__device__ __constant__ float kScale255Inv = 1.0f / 255.0f;

// Kernel: HWC uint8 [0,255] -> NCHW float32 [0,1]
// Input: uint8 buffer in HWC format (height, width, 3) with possible row padding
// Output: float32 buffer in NCHW format (1, 3, height, width)
__global__ void hwc_uint8_to_nchw_float32_kernel(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int height, int width, int input_linesize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Input: HWC with potential row padding
    const unsigned char* row = input + y * input_linesize;
    unsigned char r = row[x * 3 + 0];
    unsigned char g = row[x * 3 + 1];
    unsigned char b = row[x * 3 + 2];

    // Output: NCHW (batch=1), scale to [0,1] using multiplication (faster than division)
    int hw = height * width;
    int offset = y * width + x;
    output[0 * hw + offset] = r * kScale255Inv;  // R channel
    output[1 * hw + offset] = g * kScale255Inv;  // G channel
    output[2 * hw + offset] = b * kScale255Inv;  // B channel
}

// Helper: Clamp float to [0,255] with NaN handling and proper rounding
__device__ __forceinline__ unsigned char float_to_uint8_safe(float val) {
    // Handle NaN and Inf: NaN comparisons return false, so we check explicitly
    // isfinite() returns false for NaN and Inf
    if (!isfinite(val)) {
        return 0;  // Default to black for corrupted values
    }
    // Scale, clamp, and round to nearest integer
    val = val * 255.0f + 0.5f;  // Add 0.5 for proper rounding
    val = fminf(fmaxf(val, 0.0f), 255.0f);
    return (unsigned char)val;
}

// Kernel: NCHW float32 [0,1] -> HWC uint8 [0,255]
// Input: float32 buffer in NCHW format (1, 3, height, width)
// Output: uint8 buffer in HWC format (height, width, 3) with possible row padding
__global__ void nchw_float32_to_hwc_uint8_kernel(
    const float* __restrict__ input,
    unsigned char* __restrict__ output,
    int height, int width, int output_linesize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int hw = height * width;
    int offset = y * width + x;

    // Input: NCHW (batch=1), values in [0,1]
    float r = input[0 * hw + offset];
    float g = input[1 * hw + offset];
    float b = input[2 * hw + offset];

    // Output: HWC with potential row padding
    // Using safe conversion with NaN handling and proper rounding
    unsigned char* row = output + y * output_linesize;
    row[x * 3 + 0] = float_to_uint8_safe(r);
    row[x * 3 + 1] = float_to_uint8_safe(g);
    row[x * 3 + 2] = float_to_uint8_safe(b);
}

// Kernel: 4-channel HWC uint8 -> NCHW float32 (extract RGB, ignore alpha)
// NOTE: r_offset, g_offset, b_offset must be validated by host before launch (range [0,3]).
// Device-side bounds checking would add branching overhead to every pixel - not worth it.
__global__ void hwc4_uint8_to_nchw_float32_kernel(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    int height, int width, int input_linesize,
    int r_offset, int g_offset, int b_offset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Host is responsible for validating offsets are in [0,3]
    const unsigned char* row = input + y * input_linesize;
    unsigned char r = row[x * 4 + r_offset];
    unsigned char g = row[x * 4 + g_offset];
    unsigned char b = row[x * 4 + b_offset];

    int hw = height * width;
    int offset = y * width + x;
    output[0 * hw + offset] = r * kScale255Inv;
    output[1 * hw + offset] = g * kScale255Inv;
    output[2 * hw + offset] = b * kScale255Inv;
}

// Kernel: NCHW float32 -> 4-channel HWC uint8 (add alpha=255)
// NOTE: r_offset, g_offset, b_offset, a_offset must be validated by host before launch (range [0,3]).
// Device-side bounds checking would add branching overhead to every pixel - not worth it.
__global__ void nchw_float32_to_hwc4_uint8_kernel(
    const float* __restrict__ input,
    unsigned char* __restrict__ output,
    int height, int width, int output_linesize,
    int r_offset, int g_offset, int b_offset, int a_offset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Host is responsible for validating offsets are in [0,3]
    int hw = height * width;
    int offset = y * width + x;
    float r = input[0 * hw + offset];
    float g = input[1 * hw + offset];
    float b = input[2 * hw + offset];

    // Using safe conversion with NaN handling and proper rounding
    unsigned char* row = output + y * output_linesize;
    row[x * 4 + r_offset] = float_to_uint8_safe(r);
    row[x * 4 + g_offset] = float_to_uint8_safe(g);
    row[x * 4 + b_offset] = float_to_uint8_safe(b);
    row[x * 4 + a_offset] = 255;  // Alpha = opaque
}

// ============================================================================
// FP16 (half precision) variants
// ============================================================================

// Kernel: HWC uint8 [0,255] -> NCHW float16 [0,1]
__global__ void hwc_uint8_to_nchw_float16_kernel(
    const unsigned char* __restrict__ input,
    __half* __restrict__ output,
    int height, int width, int input_linesize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const unsigned char* row = input + y * input_linesize;
    unsigned char r = row[x * 3 + 0];
    unsigned char g = row[x * 3 + 1];
    unsigned char b = row[x * 3 + 2];

    int hw = height * width;
    int offset = y * width + x;
    output[0 * hw + offset] = __float2half(r * kScale255Inv);
    output[1 * hw + offset] = __float2half(g * kScale255Inv);
    output[2 * hw + offset] = __float2half(b * kScale255Inv);
}

// Helper: Convert half to uint8 safely
__device__ __forceinline__ unsigned char half_to_uint8_safe(__half val) {
    float f = __half2float(val);
    if (!isfinite(f)) return 0;
    f = f * 255.0f + 0.5f;
    f = fminf(fmaxf(f, 0.0f), 255.0f);
    return (unsigned char)f;
}

// Kernel: NCHW float16 [0,1] -> HWC uint8 [0,255]
__global__ void nchw_float16_to_hwc_uint8_kernel(
    const __half* __restrict__ input,
    unsigned char* __restrict__ output,
    int height, int width, int output_linesize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int hw = height * width;
    int offset = y * width + x;
    __half r = input[0 * hw + offset];
    __half g = input[1 * hw + offset];
    __half b = input[2 * hw + offset];

    unsigned char* row = output + y * output_linesize;
    row[x * 3 + 0] = half_to_uint8_safe(r);
    row[x * 3 + 1] = half_to_uint8_safe(g);
    row[x * 3 + 2] = half_to_uint8_safe(b);
}

// Kernel: 4-channel HWC uint8 -> NCHW float16 (extract RGB, ignore alpha)
// NOTE: r_offset, g_offset, b_offset must be validated by host before launch (range [0,3]).
// Device-side bounds checking would add branching overhead to every pixel - not worth it.
__global__ void hwc4_uint8_to_nchw_float16_kernel(
    const unsigned char* __restrict__ input,
    __half* __restrict__ output,
    int height, int width, int input_linesize,
    int r_offset, int g_offset, int b_offset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Host is responsible for validating offsets are in [0,3]
    const unsigned char* row = input + y * input_linesize;
    unsigned char r = row[x * 4 + r_offset];
    unsigned char g = row[x * 4 + g_offset];
    unsigned char b = row[x * 4 + b_offset];

    int hw = height * width;
    int offset = y * width + x;
    output[0 * hw + offset] = __float2half(r * kScale255Inv);
    output[1 * hw + offset] = __float2half(g * kScale255Inv);
    output[2 * hw + offset] = __float2half(b * kScale255Inv);
}

// Kernel: NCHW float16 -> 4-channel HWC uint8 (set alpha to 255)
// NOTE: r_offset, g_offset, b_offset, a_offset must be validated by host before launch (range [0,3]).
// Device-side bounds checking would add branching overhead to every pixel - not worth it.
__global__ void nchw_float16_to_hwc4_uint8_kernel(
    const __half* __restrict__ input,
    unsigned char* __restrict__ output,
    int height, int width, int output_linesize,
    int r_offset, int g_offset, int b_offset, int a_offset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Host is responsible for validating offsets are in [0,3]
    int hw = height * width;
    int offset = y * width + x;
    __half r = input[0 * hw + offset];
    __half g = input[1 * hw + offset];
    __half b = input[2 * hw + offset];

    unsigned char* row = output + y * output_linesize;
    row[x * 4 + r_offset] = half_to_uint8_safe(r);
    row[x * 4 + g_offset] = half_to_uint8_safe(g);
    row[x * 4 + b_offset] = half_to_uint8_safe(b);
    row[x * 4 + a_offset] = 255;
}

// ============================================================================
// BF16 (bfloat16) variants
// ============================================================================

// Kernel: HWC uint8 [0,255] -> NCHW bfloat16 [0,1]
__global__ void hwc_uint8_to_nchw_bfloat16_kernel(
    const unsigned char* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int height, int width, int input_linesize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    const unsigned char* row = input + y * input_linesize;
    unsigned char r = row[x * 3 + 0];
    unsigned char g = row[x * 3 + 1];
    unsigned char b = row[x * 3 + 2];

    int hw = height * width;
    int offset = y * width + x;
    output[0 * hw + offset] = __float2bfloat16(r * kScale255Inv);
    output[1 * hw + offset] = __float2bfloat16(g * kScale255Inv);
    output[2 * hw + offset] = __float2bfloat16(b * kScale255Inv);
}

// Helper: Convert bfloat16 to uint8 safely
__device__ __forceinline__ unsigned char bfloat16_to_uint8_safe(__nv_bfloat16 val) {
    float f = __bfloat162float(val);
    if (!isfinite(f)) return 0;
    f = f * 255.0f + 0.5f;
    f = fminf(fmaxf(f, 0.0f), 255.0f);
    return (unsigned char)f;
}

// Kernel: NCHW bfloat16 [0,1] -> HWC uint8 [0,255]
__global__ void nchw_bfloat16_to_hwc_uint8_kernel(
    const __nv_bfloat16* __restrict__ input,
    unsigned char* __restrict__ output,
    int height, int width, int output_linesize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int hw = height * width;
    int offset = y * width + x;
    __nv_bfloat16 r = input[0 * hw + offset];
    __nv_bfloat16 g = input[1 * hw + offset];
    __nv_bfloat16 b = input[2 * hw + offset];

    unsigned char* row = output + y * output_linesize;
    row[x * 3 + 0] = bfloat16_to_uint8_safe(r);
    row[x * 3 + 1] = bfloat16_to_uint8_safe(g);
    row[x * 3 + 2] = bfloat16_to_uint8_safe(b);
}

// Kernel: 4-channel HWC uint8 -> NCHW bfloat16 (extract RGB, ignore alpha)
// NOTE: r_offset, g_offset, b_offset must be validated by host before launch (range [0,3]).
// Device-side bounds checking would add branching overhead to every pixel - not worth it.
__global__ void hwc4_uint8_to_nchw_bfloat16_kernel(
    const unsigned char* __restrict__ input,
    __nv_bfloat16* __restrict__ output,
    int height, int width, int input_linesize,
    int r_offset, int g_offset, int b_offset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Host is responsible for validating offsets are in [0,3]
    const unsigned char* row = input + y * input_linesize;
    unsigned char r = row[x * 4 + r_offset];
    unsigned char g = row[x * 4 + g_offset];
    unsigned char b = row[x * 4 + b_offset];

    int hw = height * width;
    int offset = y * width + x;
    output[0 * hw + offset] = __float2bfloat16(r * kScale255Inv);
    output[1 * hw + offset] = __float2bfloat16(g * kScale255Inv);
    output[2 * hw + offset] = __float2bfloat16(b * kScale255Inv);
}

// Kernel: NCHW bfloat16 -> 4-channel HWC uint8 (set alpha to 255)
// NOTE: r_offset, g_offset, b_offset, a_offset must be validated by host before launch (range [0,3]).
// Device-side bounds checking would add branching overhead to every pixel - not worth it.
__global__ void nchw_bfloat16_to_hwc4_uint8_kernel(
    const __nv_bfloat16* __restrict__ input,
    unsigned char* __restrict__ output,
    int height, int width, int output_linesize,
    int r_offset, int g_offset, int b_offset, int a_offset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Host is responsible for validating offsets are in [0,3]
    int hw = height * width;
    int offset = y * width + x;
    __nv_bfloat16 r = input[0 * hw + offset];
    __nv_bfloat16 g = input[1 * hw + offset];
    __nv_bfloat16 b = input[2 * hw + offset];

    unsigned char* row = output + y * output_linesize;
    row[x * 4 + r_offset] = bfloat16_to_uint8_safe(r);
    row[x * 4 + g_offset] = bfloat16_to_uint8_safe(g);
    row[x * 4 + b_offset] = bfloat16_to_uint8_safe(b);
    row[x * 4 + a_offset] = 255;
}

}  // extern "C"
