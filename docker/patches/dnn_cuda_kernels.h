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
 * CUDA kernel PTX declarations for DNN backend format conversion.
 * Kernels are compiled to PTX at build time and loaded via Driver API at runtime.
 * This avoids any CUDA runtime (cudart) dependency.
 */

#ifndef AVFILTER_DNN_CUDA_KERNELS_H
#define AVFILTER_DNN_CUDA_KERNELS_H

#include <stddef.h>

/* PTX bytecode embedded at compile time via bin2c */
extern const unsigned char ff_dnn_cuda_kernels_ptx[];
extern const unsigned int ff_dnn_cuda_kernels_ptx_len;

/* Kernel names within the PTX module */
/* FP32 variants */
#define DNN_CUDA_KERNEL_HWC_UINT8_TO_NCHW_FLOAT32     "hwc_uint8_to_nchw_float32_kernel"
#define DNN_CUDA_KERNEL_NCHW_FLOAT32_TO_HWC_UINT8     "nchw_float32_to_hwc_uint8_kernel"
#define DNN_CUDA_KERNEL_HWC4_UINT8_TO_NCHW_FLOAT32    "hwc4_uint8_to_nchw_float32_kernel"
#define DNN_CUDA_KERNEL_NCHW_FLOAT32_TO_HWC4_UINT8    "nchw_float32_to_hwc4_uint8_kernel"

/* FP16 variants */
#define DNN_CUDA_KERNEL_HWC_UINT8_TO_NCHW_FLOAT16     "hwc_uint8_to_nchw_float16_kernel"
#define DNN_CUDA_KERNEL_NCHW_FLOAT16_TO_HWC_UINT8     "nchw_float16_to_hwc_uint8_kernel"
#define DNN_CUDA_KERNEL_HWC4_UINT8_TO_NCHW_FLOAT16    "hwc4_uint8_to_nchw_float16_kernel"
#define DNN_CUDA_KERNEL_NCHW_FLOAT16_TO_HWC4_UINT8    "nchw_float16_to_hwc4_uint8_kernel"

/* BF16 variants */
#define DNN_CUDA_KERNEL_HWC_UINT8_TO_NCHW_BFLOAT16    "hwc_uint8_to_nchw_bfloat16_kernel"
#define DNN_CUDA_KERNEL_NCHW_BFLOAT16_TO_HWC_UINT8    "nchw_bfloat16_to_hwc_uint8_kernel"
#define DNN_CUDA_KERNEL_HWC4_UINT8_TO_NCHW_BFLOAT16   "hwc4_uint8_to_nchw_bfloat16_kernel"
#define DNN_CUDA_KERNEL_NCHW_BFLOAT16_TO_HWC4_UINT8   "nchw_bfloat16_to_hwc4_uint8_kernel"

#endif  /* AVFILTER_DNN_CUDA_KERNELS_H */
