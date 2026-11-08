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
 * Based on FFmpeg's dnn_backend_torch.cpp with extensive modifications
 * for CUDA zero-copy support and hardware frame integration.
 */

/**
 * @file
 * DNN Torch backend implementation.
 */

#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/script.h>
#include <dlfcn.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

extern "C" {
#include "dnn_io_proc.h"
#include "dnn_backend_common.h"
#include "libavutil/opt.h"
#include "libavutil/mem.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_cuda.h"
#include "libavutil/pixfmt.h"
#include "libavutil/pixdesc.h"
#include "queue.h"
#include "safe_queue.h"
}

#include <cuda_runtime.h>

typedef struct THModel {
    DNNModel model;
    DnnContext *ctx;
    torch::jit::Module *jit_model;
    SafeQueue *request_queue;
    Queue *task_queue;
    Queue *lltask_queue;
    SafeQueue *pending_queue;       ///< requests waiting for inference
    std::thread *worker_thread;     ///< background worker thread
    std::mutex *mutex;              ///< mutex for the condition variable
    std::condition_variable *cond;  ///< condition variable for worker wakeup
    std::atomic<bool> worker_stop;  ///< signal for thread exit
} THModel;

typedef struct THInferRequest {
    torch::Tensor *output;
    torch::Tensor *input_tensor;
} THInferRequest;

typedef struct THRequestItem {
    THInferRequest *infer_request;
    LastLevelTaskItem *lltask;
    DNNAsyncExecModule exec_module;
} THRequestItem;


#define OFFSET(x) offsetof(THOptions, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM
static const AVOption dnn_th_options[] = {
    { "optimize", "turn on graph executor optimization", OFFSET(optimize), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 1, FLAGS},
    { NULL }
};

static int extract_lltask_from_task(TaskItem *task, Queue *lltask_queue)
{
    THModel *th_model = (THModel *)task->model;
    DnnContext *ctx = th_model->ctx;
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

static void th_free_request(THInferRequest *request)
{
    if (!request)
        return;
    if (request->output) {
        delete(request->output);
        request->output = NULL;
    }
    if (request->input_tensor) {
        delete(request->input_tensor);
        request->input_tensor = NULL;
    }
    return;
}

static inline void destroy_request_item(THRequestItem **arg)
{
    THRequestItem *item;
    if (!arg || !*arg) {
        return;
    }
    item = *arg;
    th_free_request(item->infer_request);
    av_freep(&item->infer_request);
    av_freep(&item->lltask);
    ff_dnn_async_module_cleanup(&item->exec_module);
    av_freep(arg);
}

static void dnn_free_model_th(DNNModel **model)
{
    THModel *th_model;
    if (!model || !*model)
        return;

    th_model = (THModel *)(*model);

    /* 1. Stop and join the worker thread if it exists */
    if (th_model->worker_thread) {
        {
            std::lock_guard<std::mutex> lock(*th_model->mutex);
            th_model->worker_stop = true;
        }
        th_model->cond->notify_all();
        th_model->worker_thread->join();
        delete th_model->worker_thread;
        th_model->worker_thread = NULL;
    }

    /* 2. Safely delete C++ synchronization objects */
    if (th_model->mutex) {
        delete th_model->mutex;
        th_model->mutex = NULL;
    }
    if (th_model->cond) {
        delete th_model->cond;
        th_model->cond = NULL;
    }

    /* 3. Clean up the pending queue */
    if (th_model->pending_queue) {
        while (ff_safe_queue_size(th_model->pending_queue) > 0) {
            THRequestItem *item = (THRequestItem *)ff_safe_queue_pop_front(th_model->pending_queue);
            destroy_request_item(&item);
        }
        ff_safe_queue_destroy(th_model->pending_queue);
    }

    /* 4. Clean up standard backend queues */
    if (th_model->request_queue) {
        while (ff_safe_queue_size(th_model->request_queue) != 0) {
            THRequestItem *item = (THRequestItem *)ff_safe_queue_pop_front(th_model->request_queue);
            destroy_request_item(&item);
        }
        ff_safe_queue_destroy(th_model->request_queue);
    }

    if (th_model->lltask_queue) {
        while (ff_queue_size(th_model->lltask_queue) != 0) {
            LastLevelTaskItem *item = (LastLevelTaskItem *)ff_queue_pop_front(th_model->lltask_queue);
            av_freep(&item);
        }
        ff_queue_destroy(th_model->lltask_queue);
    }

    if (th_model->task_queue) {
        while (ff_queue_size(th_model->task_queue) != 0) {
            TaskItem *item = (TaskItem *)ff_queue_pop_front(th_model->task_queue);
            av_frame_free(&item->in_frame);
            av_frame_free(&item->out_frame);
            av_freep(&item);
        }
        ff_queue_destroy(th_model->task_queue);
    }

    /* 5. Final model cleanup */
    if (th_model->jit_model)
        delete th_model->jit_model;

    av_freep(&th_model);
    *model = NULL;
}

static int get_input_th(DNNModel *model, DNNData *input, const char *input_name)
{
    input->dt = DNN_FLOAT;
    input->order = DCO_RGB;
    input->layout = DL_NCHW;
    input->dims[0] = 1;
    input->dims[1] = 3;
    input->dims[2] = -1;
    input->dims[3] = -1;
    return 0;
}

static void deleter(void *arg)
{
    av_freep(&arg);
}

static int fill_model_input_th(THModel *th_model, THRequestItem *request)
{
    LastLevelTaskItem *lltask = NULL;
    TaskItem *task = NULL;
    THInferRequest *infer_request = NULL;
    DNNData input = { 0 };
    DnnContext *ctx = th_model->ctx;
    int ret, width_idx, height_idx, channel_idx;

    lltask = (LastLevelTaskItem *)ff_queue_pop_front(th_model->lltask_queue);
    if (!lltask) {
        ret = AVERROR(EINVAL);
        goto err;
    }
    request->lltask = lltask;
    task = lltask->task;
    infer_request = request->infer_request;

    ret = get_input_th(&th_model->model, &input, NULL);
    if (ret != 0) {
        goto err;
    }
    width_idx = dnn_get_width_idx_by_layout(input.layout);
    height_idx = dnn_get_height_idx_by_layout(input.layout);
    channel_idx = dnn_get_channel_idx_by_layout(input.layout);
    input.dims[height_idx] = task->in_frame->height;
    input.dims[width_idx] = task->in_frame->width;

    // Allocate tensors. Note: th_create_inference_request() NULL-initializes both pointers,
    // so th_free_request() in the err path safely handles partial allocation if second new throws.
    try {
        infer_request->input_tensor = new torch::Tensor();
        infer_request->output = new torch::Tensor();
    } catch (const std::exception& e) {
        av_log(ctx, AV_LOG_ERROR, "Failed to allocate torch tensors: %s\n", e.what());
        ret = AVERROR(ENOMEM);
        goto err;
    }

    // Check for CUDA hardware frames (zero-copy input path)
    if (task->in_frame->format == AV_PIX_FMT_CUDA && task->in_frame->hw_frames_ctx) {
        AVHWFramesContext *hw_frames = (AVHWFramesContext *)task->in_frame->hw_frames_ctx->data;
        int width = task->in_frame->width;
        int height = task->in_frame->height;
        int linesize = task->in_frame->linesize[0];
        uint8_t *cuda_data = task->in_frame->data[0];

        av_log(ctx, AV_LOG_DEBUG, "CUDA frame input: %dx%d, sw_format=%s, linesize=%d\n",
               width, height, av_get_pix_fmt_name(hw_frames->sw_format), linesize);

        try {
            // Handle RGB24/BGR24 sw_format - zero-copy path (3 bytes per pixel)
            if (hw_frames->sw_format == AV_PIX_FMT_RGB24 || hw_frames->sw_format == AV_PIX_FMT_BGR24) {
                auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);

                // Create tensor from CUDA memory (HWC format, uint8)
                torch::Tensor input_hwc = torch::from_blob(
                    cuda_data,
                    {height, width, 3},
                    {linesize, 3, 1},  // strides for row-major with padding
                    options
                );

                // Convert: HWC uint8 [0,255] -> NCHW float32 [0,1]
                *infer_request->input_tensor = input_hwc.permute({2, 0, 1})  // HWC -> CHW
                                                        .unsqueeze(0)        // CHW -> NCHW
                                                        .to(torch::kFloat32)
                                                        .div(255.0f)
                                                        .contiguous();

                av_log(ctx, AV_LOG_DEBUG, "Zero-copy CUDA input created (RGB24/BGR24)\n");
                return 0;
            }

            // Handle RGB0/BGR0/0RGB/0BGR sw_format - zero-copy path (4 bytes per pixel, ignore alpha)
            if (hw_frames->sw_format == AV_PIX_FMT_RGB0 || hw_frames->sw_format == AV_PIX_FMT_BGR0 ||
                hw_frames->sw_format == AV_PIX_FMT_0RGB || hw_frames->sw_format == AV_PIX_FMT_0BGR ||
                hw_frames->sw_format == AV_PIX_FMT_RGBA || hw_frames->sw_format == AV_PIX_FMT_BGRA ||
                hw_frames->sw_format == AV_PIX_FMT_ARGB || hw_frames->sw_format == AV_PIX_FMT_ABGR) {
                auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);

                // Create tensor from CUDA memory (4 channels, uint8)
                torch::Tensor input_hwc4 = torch::from_blob(
                    cuda_data,
                    {height, width, 4},
                    {linesize, 4, 1},  // strides for row-major with padding
                    options
                );

                // Extract RGB channels based on format
                torch::Tensor input_hwc;
                if (hw_frames->sw_format == AV_PIX_FMT_RGB0 || hw_frames->sw_format == AV_PIX_FMT_BGR0 ||
                    hw_frames->sw_format == AV_PIX_FMT_RGBA || hw_frames->sw_format == AV_PIX_FMT_BGRA) {
                    // RGB(A) format: first 3 channels are R, G, B
                    input_hwc = input_hwc4.slice(2, 0, 3);  // slice along channel dim
                } else {
                    // (A)RGB format: last 3 channels are R, G, B
                    input_hwc = input_hwc4.slice(2, 1, 4);  // slice along channel dim
                }

                // Convert: HWC uint8 [0,255] -> NCHW float32 [0,1]
                *infer_request->input_tensor = input_hwc.permute({2, 0, 1})  // HWC -> CHW
                                                        .unsqueeze(0)        // CHW -> NCHW
                                                        .to(torch::kFloat32)
                                                        .div(255.0f)
                                                        .contiguous();

                av_log(ctx, AV_LOG_DEBUG, "Zero-copy CUDA input created (4-channel format)\n");
                return 0;
            }
        } catch (const std::exception& e) {
            av_log(ctx, AV_LOG_ERROR, "Torch exception in zero-copy input: %s\n", e.what());
            ret = AVERROR(ENOSYS);
            goto err;
        }

        av_log(ctx, AV_LOG_WARNING, "CUDA sw_format %s not supported for zero-copy, falling back to CPU\n",
               av_get_pix_fmt_name(hw_frames->sw_format));
    }

    // Standard CPU path - allocate memory for input data
    input.data = av_malloc(input.dims[height_idx] * input.dims[width_idx] *
                           input.dims[channel_idx] * sizeof(float));
    if (!input.data) {
        ret = AVERROR(ENOMEM);
        goto err;
    }

    switch (th_model->model.func_type) {
    case DFT_PROCESS_FRAME:
        input.scale = 255;
        if (task->do_ioproc) {
            if (th_model->model.frame_pre_proc != NULL) {
                th_model->model.frame_pre_proc(task->in_frame, &input, th_model->model.filter_ctx);
            } else {
                ff_proc_from_frame_to_dnn(task->in_frame, &input, ctx);
            }
        }
        break;
    default:
        avpriv_report_missing_feature(NULL, "model function type %d", th_model->model.func_type);
        av_freep(&input.data);
        ret = AVERROR(ENOSYS);
        goto err;
    }
    *infer_request->input_tensor = torch::from_blob(input.data,
        {1, input.dims[channel_idx], input.dims[height_idx], input.dims[width_idx]},
        deleter, torch::kFloat32);
    return 0;

err:
    th_free_request(infer_request);
    return ret;
}

static int th_start_inference(void *args)
{
    THRequestItem *request = (THRequestItem *)args;
    THInferRequest *infer_request = NULL;
    LastLevelTaskItem *lltask = NULL;
    TaskItem *task = NULL;
    THModel *th_model = NULL;
    DnnContext *ctx = NULL;
    std::vector<torch::jit::IValue> inputs;
    torch::NoGradGuard no_grad;

    if (!request) {
        av_log(NULL, AV_LOG_ERROR, "THRequestItem is NULL\n");
        return AVERROR(EINVAL);
    }
    infer_request = request->infer_request;
    lltask = request->lltask;
    if (!lltask) {
        av_log(NULL, AV_LOG_ERROR, "THRequestItem lltask is NULL\n");
        return AVERROR(EINVAL);
    }
    task = lltask->task;
    th_model = (THModel *)task->model;
    ctx = th_model->ctx;

    if (!infer_request->input_tensor || !infer_request->output) {
        av_log(ctx, AV_LOG_ERROR, "input or output tensor is NULL\n");
        return DNN_GENERIC_ERROR;
    }

    try {
        // Transfer tensor to the same device as model
        c10::Device device = torch::kCUDA;
        auto params = th_model->jit_model->parameters();
        if (params.begin() != params.end()) {
            device = (*params.begin()).device();
        }
        if (infer_request->input_tensor->device() != device)
            *infer_request->input_tensor = infer_request->input_tensor->to(device);
        inputs.push_back(*infer_request->input_tensor);

        auto _fwd_out = th_model->jit_model->forward(inputs);
        if (_fwd_out.isTuple()) {
            *infer_request->output = _fwd_out.toTuple()->elements()[0].toTensor();
        } else {
            *infer_request->output = _fwd_out.toTensor();
        }
    } catch (const std::exception& e) {
        av_log(ctx, AV_LOG_ERROR, "Torch inference failed: %s\n", e.what());
        return DNN_GENERIC_ERROR;
    }

    return 0;
}

static void infer_completion_callback(void *args) {
    THRequestItem *request = (THRequestItem*)args;
    LastLevelTaskItem *lltask = request->lltask;
    TaskItem *task = lltask->task;
    DNNData outputs = { 0 };
    THInferRequest *infer_request = request->infer_request;
    THModel *th_model = (THModel *)task->model;
    torch::Tensor *output = infer_request->output;
    c10::IntArrayRef sizes;

    try {
        sizes = output->sizes();
    outputs.order = DCO_RGB;
    outputs.layout = DL_NCHW;
    outputs.dt = DNN_FLOAT;
    if (sizes.size() == 4) {
        // 4 dimensions: [batch_size, channel, height, width]
        // this format of data is normally used for video frame SR
        outputs.dims[0] = sizes.at(0); // N
        outputs.dims[1] = sizes.at(1); // C
        outputs.dims[2] = sizes.at(2); // H
        outputs.dims[3] = sizes.at(3); // W
    } else {
        avpriv_report_missing_feature(th_model->ctx, "Support of this kind of model");
        goto err;
    }

    switch (th_model->model.func_type) {
    case DFT_PROCESS_FRAME:
        // Check for CUDA output frames (zero-copy output path)
        if (task->out_frame->format == AV_PIX_FMT_CUDA && task->out_frame->hw_frames_ctx) {
            AVHWFramesContext *hw_frames = (AVHWFramesContext *)task->out_frame->hw_frames_ctx->data;
            int out_height = outputs.dims[dnn_get_height_idx_by_layout(outputs.layout)];
            int out_width = outputs.dims[dnn_get_width_idx_by_layout(outputs.layout)];
            int out_linesize = task->out_frame->linesize[0];
            uint8_t *cuda_out = task->out_frame->data[0];

            av_log(th_model->ctx, AV_LOG_DEBUG, "CUDA frame output: %dx%d, sw_format=%s\n",
                   out_width, out_height, av_get_pix_fmt_name(hw_frames->sw_format));

            if (hw_frames->sw_format == AV_PIX_FMT_RGB24 || hw_frames->sw_format == AV_PIX_FMT_BGR24) {
                // Ensure output is on CUDA
                if (!output->is_cuda()) {
                    *output = output->to(torch::kCUDA);
                }

                // Convert: NCHW float32 [0,1] -> HWC uint8 [0,255]
                torch::Tensor output_hwc = output->squeeze(0)           // NCHW -> CHW
                                                  .permute({1, 2, 0})   // CHW -> HWC
                                                  .mul(255.0f)
                                                  .clamp(0.0f, 255.0f)
                                                  .to(torch::kUInt8)
                                                  .contiguous();

                // Copy to output CUDA frame
                cudaError_t cuda_err;
                if (out_linesize == out_width * 3) {
                    // Contiguous - single copy
                    cuda_err = cudaMemcpy(cuda_out, output_hwc.data_ptr(),
                                          out_height * out_width * 3, cudaMemcpyDeviceToDevice);
                    if (cuda_err != cudaSuccess) {
                        av_log(th_model->ctx, AV_LOG_ERROR, "cudaMemcpy failed: %s\n",
                               cudaGetErrorString(cuda_err));
                        goto err;
                    }
                } else {
                    // Padded rows - copy row by row
                    for (int y = 0; y < out_height; y++) {
                        cuda_err = cudaMemcpy(cuda_out + y * out_linesize,
                                              (uint8_t*)output_hwc.data_ptr() + y * out_width * 3,
                                              out_width * 3, cudaMemcpyDeviceToDevice);
                        if (cuda_err != cudaSuccess) {
                            av_log(th_model->ctx, AV_LOG_ERROR, "cudaMemcpy row %d failed: %s\n",
                                   y, cudaGetErrorString(cuda_err));
                            goto err;
                        }
                    }
                }

                task->out_frame->width = out_width;
                task->out_frame->height = out_height;

                av_log(th_model->ctx, AV_LOG_DEBUG, "Zero-copy CUDA output done (RGB24/BGR24)\n");
                break;
            }

            // Handle 4-channel output formats (RGB0, BGR0, RGBA, etc.)
            if (hw_frames->sw_format == AV_PIX_FMT_RGB0 || hw_frames->sw_format == AV_PIX_FMT_BGR0 ||
                hw_frames->sw_format == AV_PIX_FMT_0RGB || hw_frames->sw_format == AV_PIX_FMT_0BGR ||
                hw_frames->sw_format == AV_PIX_FMT_RGBA || hw_frames->sw_format == AV_PIX_FMT_BGRA ||
                hw_frames->sw_format == AV_PIX_FMT_ARGB || hw_frames->sw_format == AV_PIX_FMT_ABGR) {
                // Ensure output is on CUDA
                if (!output->is_cuda()) {
                    *output = output->to(torch::kCUDA);
                }

                // Convert: NCHW float32 [0,1] -> HWC uint8 [0,255]
                torch::Tensor output_hwc = output->squeeze(0)           // NCHW -> CHW
                                                  .permute({1, 2, 0})   // CHW -> HWC
                                                  .mul(255.0f)
                                                  .clamp(0.0f, 255.0f)
                                                  .to(torch::kUInt8)
                                                  .contiguous();

                // Create 4-channel output with alpha=255
                auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
                torch::Tensor alpha = torch::full({out_height, out_width, 1}, 255, options);
                torch::Tensor output_hwc4;

                if (hw_frames->sw_format == AV_PIX_FMT_RGB0 || hw_frames->sw_format == AV_PIX_FMT_BGR0 ||
                    hw_frames->sw_format == AV_PIX_FMT_RGBA || hw_frames->sw_format == AV_PIX_FMT_BGRA) {
                    // RGB(A) format: R, G, B, A
                    output_hwc4 = torch::cat({output_hwc, alpha}, 2).contiguous();
                } else {
                    // (A)RGB format: A, R, G, B
                    output_hwc4 = torch::cat({alpha, output_hwc}, 2).contiguous();
                }

                // Copy to output CUDA frame
                cudaError_t cuda_err4;
                if (out_linesize == out_width * 4) {
                    // Contiguous - single copy
                    cuda_err4 = cudaMemcpy(cuda_out, output_hwc4.data_ptr(),
                                           out_height * out_width * 4, cudaMemcpyDeviceToDevice);
                    if (cuda_err4 != cudaSuccess) {
                        av_log(th_model->ctx, AV_LOG_ERROR, "cudaMemcpy failed: %s\n",
                               cudaGetErrorString(cuda_err4));
                        goto err;
                    }
                } else {
                    // Padded rows - copy row by row
                    for (int y = 0; y < out_height; y++) {
                        cuda_err4 = cudaMemcpy(cuda_out + y * out_linesize,
                                               (uint8_t*)output_hwc4.data_ptr() + y * out_width * 4,
                                               out_width * 4, cudaMemcpyDeviceToDevice);
                        if (cuda_err4 != cudaSuccess) {
                            av_log(th_model->ctx, AV_LOG_ERROR, "cudaMemcpy row %d failed: %s\n",
                                   y, cudaGetErrorString(cuda_err4));
                            goto err;
                        }
                    }
                }

                task->out_frame->width = out_width;
                task->out_frame->height = out_height;

                av_log(th_model->ctx, AV_LOG_DEBUG, "Zero-copy CUDA output done (4-channel format)\n");
                break;
            }

            av_log(th_model->ctx, AV_LOG_WARNING, "CUDA output sw_format %s not supported, falling back to CPU\n",
                   av_get_pix_fmt_name(hw_frames->sw_format));
        }

        // Standard CPU output path
        if (task->do_ioproc) {
            // Post process can only deal with CPU memory.
            if (output->device() != torch::kCPU)
                *output = output->to(torch::kCPU);  // Expensive GPU->CPU copy!
            outputs.scale = 255;
            outputs.data = output->data_ptr();
            if (th_model->model.frame_post_proc != NULL) {
                th_model->model.frame_post_proc(task->out_frame, &outputs, th_model->model.filter_ctx);
            } else {
                ff_proc_from_dnn_to_frame(task->out_frame, &outputs, th_model->ctx);
            }
        } else {
            task->out_frame->width = outputs.dims[dnn_get_width_idx_by_layout(outputs.layout)];
            task->out_frame->height = outputs.dims[dnn_get_height_idx_by_layout(outputs.layout)];
        }
        break;
    default:
        avpriv_report_missing_feature(th_model->ctx, "model function type %d", th_model->model.func_type);
        goto err;
    }
    task->inference_done++;
    goto done;
    } catch (const std::exception& e) {
        av_log(th_model->ctx, AV_LOG_ERROR, "Torch exception in completion callback: %s\n", e.what());
    }
err:
    // Increment inference_done even on error so task completion tracking works
    task->inference_done++;
done:
    // Free lltask - it was popped from the queue in fill_model_input_th
    av_freep(&request->lltask);

    // Free the inference request data (tensors)
    th_free_request(infer_request);

    // Don't free infer_request struct here - it's reused when pushed back to request_queue
    // The struct will be freed when the model is destroyed

    if (ff_safe_queue_push_back(th_model->request_queue, request) < 0) {
        // Only destroy if we can't push back - this will free the struct
        av_freep(&request->infer_request);
        av_freep(&request);
        av_log(th_model->ctx, AV_LOG_ERROR, "Unable to push back request_queue when failed to start inference.\n");
    }
}

static void th_worker_thread(THModel *th_model) {
    while (true) {
        THRequestItem *request = NULL;
        {
            std::unique_lock<std::mutex> lock(*th_model->mutex);
            th_model->cond->wait(lock, [&]{
                return th_model->worker_stop || ff_safe_queue_size(th_model->pending_queue) > 0;
            });

            if (th_model->worker_stop && ff_safe_queue_size(th_model->pending_queue) == 0)
                break;

            request = (THRequestItem *)ff_safe_queue_pop_front(th_model->pending_queue);
        }

        if (request) {
            int ret = th_start_inference(request);
            if (ret < 0) {
                av_log(NULL, AV_LOG_ERROR, "Async inference failed: %d\n", ret);
            }
            infer_completion_callback(request);
        }
    }
}

static int execute_model_th(THRequestItem *request, Queue *lltask_queue)
{
    THModel *th_model = NULL;
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
    th_model = (THModel *)task->model;

    ret = fill_model_input_th(th_model, request);
    if ( ret != 0) {
        goto err;
    }
    if (task->async) {
        std::lock_guard<std::mutex> lock(*th_model->mutex);
        if (ff_safe_queue_push_back(th_model->pending_queue, request) < 0) {
            th_free_request(request->infer_request);
            av_freep(&request->lltask);
            if (ff_safe_queue_push_back(th_model->request_queue, request) < 0) {
                destroy_request_item(&request);
            }
            return AVERROR(ENOMEM);
        }
        th_model->cond->notify_one();
        return 0;
    } else {
        // Synchronous execution path
        ret = th_start_inference((void *)request);
        if (ret != 0) {
            goto err;
        }
        infer_completion_callback(request);
        return (task->inference_done == task->inference_todo) ? 0 : DNN_GENERIC_ERROR;
    }

err:
    th_free_request(request->infer_request);
    av_freep(&request->lltask);  // Free lltask to avoid leak and dangling pointer
    if (!th_model || ff_safe_queue_push_back(th_model->request_queue, request) < 0) {
        destroy_request_item(&request);
    }
    return ret;
}

static int get_output_th(DNNModel *model, const char *input_name, int input_width, int input_height,
                                   const char *output_name, int *output_width, int *output_height)
{
    int ret = 0;
    THModel *th_model = (THModel*) model;
    DnnContext *ctx = th_model->ctx;
    TaskItem task = { 0 };
    THRequestItem *request = NULL;
    DNNExecBaseParams exec_params = {
        .input_name     = input_name,
        .output_names   = &output_name,
        .nb_output      = 1,
        .in_frame       = NULL,
        .out_frame      = NULL,
    };
    ret = ff_dnn_fill_gettingoutput_task(&task, &exec_params, th_model, input_height, input_width, ctx);
    if ( ret != 0) {
        goto err;
    }

    ret = extract_lltask_from_task(&task, th_model->lltask_queue);
    if ( ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "unable to extract last level task from task.\n");
        goto err;
    }

    request = (THRequestItem*) ff_safe_queue_pop_front(th_model->request_queue);
    if (!request) {
        av_log(ctx, AV_LOG_ERROR, "unable to get infer request.\n");
        ret = AVERROR(EINVAL);
        // Clean up lltask that was pushed by extract_lltask_from_task
        LastLevelTaskItem *lltask = (LastLevelTaskItem *)ff_queue_pop_back(th_model->lltask_queue);
        av_freep(&lltask);
        goto err;
    }

    ret = execute_model_th(request, th_model->lltask_queue);
    *output_width = task.out_frame->width;
    *output_height = task.out_frame->height;

err:
    av_frame_free(&task.out_frame);
    av_frame_free(&task.in_frame);
    return ret;
}

static THInferRequest *th_create_inference_request(void)
{
    THInferRequest *request = (THInferRequest *)av_malloc(sizeof(THInferRequest));
    if (!request) {
        return NULL;
    }
    request->input_tensor = NULL;
    request->output = NULL;
    return request;
}

static DNNModel *dnn_load_model_th(DnnContext *ctx, DNNFunctionType func_type, AVFilterContext *filter_ctx)
{
    DNNModel *model = NULL;
    THModel *th_model = NULL;
    THRequestItem *item = NULL;
    const char *device_name = ctx->device ? ctx->device : "cpu";

    th_model = (THModel *)av_mallocz(sizeof(THModel));
    if (!th_model)
        return NULL;
    model = &th_model->model;
    th_model->ctx = ctx;

    c10::Device device = c10::Device(device_name);
    if (device.is_xpu()) {
        if (!at::hasXPU()) {
            av_log(ctx, AV_LOG_ERROR, "No XPU device found\n");
            goto fail;
        }
        at::detail::getXPUHooks().init();
    } else if (device.is_cuda()) {
        if (!at::cuda::is_available()) {
            av_log(ctx, AV_LOG_ERROR, "No CUDA device found\n");
            goto fail;
        }
        // Load CUDA kernels - required for libtorch CUDA ops
        // Thread-safe initialization using call_once
        // NOTE: These handles are intentionally never dlclose'd. CUDA/TensorRT libraries
        // have complex cleanup requirements and calling dlclose can cause crashes.
        // The OS reclaims resources on process exit.
        static std::once_flag cuda_lib_once;
        static void *cuda_lib_handle = NULL;
        std::call_once(cuda_lib_once, [ctx]() {
            cuda_lib_handle = dlopen("libtorch_cuda.so", RTLD_NOW | RTLD_GLOBAL);
            if (cuda_lib_handle) {
                av_log(ctx, AV_LOG_DEBUG, "libtorch_cuda.so loaded\n");
            } else {
                av_log(ctx, AV_LOG_WARNING, "Failed to load libtorch_cuda.so: %s\n", dlerror());
            }
        });
    } else if (!device.is_cpu()) {
        av_log(ctx, AV_LOG_ERROR, "Not supported device:\"%s\"\n", device_name);
        goto fail;
    }

    try {
        th_model->jit_model = new torch::jit::Module;
        // Load TensorRT runtime if available (enables TRT-compiled models)
        // Thread-safe initialization using call_once
        static std::once_flag trt_once;
        static void *trt_lib_handle = NULL;
        std::call_once(trt_once, [ctx]() {
            trt_lib_handle = dlopen("libtorchtrt_runtime.so", RTLD_NOW | RTLD_GLOBAL);
            if (trt_lib_handle) {
                av_log(ctx, AV_LOG_INFO, "TensorRT runtime loaded\n");
            }
        });
        (*th_model->jit_model) = torch::jit::load(ctx->model_filename);
        th_model->jit_model->to(device);
        // Set JIT optimization once at model load time (thread-safe)
        torch::jit::setGraphExecutorOptimize(ctx->torch_option.optimize ? true : false);
        av_log(ctx, AV_LOG_INFO, "Model loaded to device: %s (JIT optimize=%d)\n",
               device_name, ctx->torch_option.optimize);
        if (device.is_cuda()) {
            av_log(ctx, AV_LOG_INFO, "CUDA available: %s, device count: %d\n",
                   at::cuda::is_available() ? "yes" : "no",
                   at::cuda::device_count());
        }
    } catch (const c10::Error& e) {
        av_log(ctx, AV_LOG_ERROR, "Failed to load torch model: %s\n", e.what());
        goto fail;
    } catch (const std::exception& e) {
        av_log(ctx, AV_LOG_ERROR, "Failed to load torch model: %s\n", e.what());
        goto fail;
    }

    th_model->request_queue = ff_safe_queue_create();
    if (!th_model->request_queue) {
        goto fail;
    }

    item = (THRequestItem *)av_mallocz(sizeof(THRequestItem));
    if (!item) {
        goto fail;
    }
    item->lltask = NULL;
    item->infer_request = th_create_inference_request();
    if (!item->infer_request) {
        av_log(NULL, AV_LOG_ERROR, "Failed to allocate memory for Torch inference request\n");
        goto fail;
    }
    item->exec_module.start_inference = &th_start_inference;
    item->exec_module.callback = &infer_completion_callback;
    item->exec_module.args = item;

    if (ff_safe_queue_push_back(th_model->request_queue, item) < 0) {
        goto fail;
    }
    item = NULL;

    th_model->task_queue = ff_queue_create();
    if (!th_model->task_queue) {
        goto fail;
    }

    th_model->lltask_queue = ff_queue_create();
    if (!th_model->lltask_queue) {
        goto fail;
    }

    th_model->pending_queue = ff_safe_queue_create();
    if (!th_model->pending_queue) {
        goto fail;
    }

    try {
        th_model->mutex = new std::mutex();
        th_model->cond = new std::condition_variable();
        th_model->worker_stop = false;
        th_model->worker_thread = new std::thread(th_worker_thread, th_model);
    } catch (const std::exception& e) {
        av_log(ctx, AV_LOG_ERROR, "Failed to create worker thread: %s\n", e.what());
        goto fail;
    }

    model->get_input = &get_input_th;
    model->get_output = &get_output_th;
    model->filter_ctx = filter_ctx;
    model->func_type = func_type;
    return model;

fail:
    if (item) {
        destroy_request_item(&item);
        // Note: destroy_request_item already calls av_freep(arg), so item is now NULL
    }
    dnn_free_model_th(&model);
    return NULL;
}

static int dnn_execute_model_th(const DNNModel *model, DNNExecBaseParams *exec_params)
{
    THModel *th_model = (THModel *)model;
    DnnContext *ctx = th_model->ctx;
    TaskItem *task;
    THRequestItem *request;
    int ret = 0;

    ret = ff_check_exec_params(ctx, DNN_TH, model->func_type, exec_params);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "exec parameter checking fail.\n");
        return ret;
    }

    task = (TaskItem *)av_malloc(sizeof(TaskItem));
    if (!task) {
        av_log(ctx, AV_LOG_ERROR, "unable to alloc memory for task item.\n");
        return AVERROR(ENOMEM);
    }

    ret = ff_dnn_fill_task(task, exec_params, th_model, 0, 1);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "unable to fill task.\n");
        av_frame_free(&task->in_frame);
        av_frame_free(&task->out_frame);
        av_freep(&task);
        return ret;
    }

    ret = ff_queue_push_back(th_model->task_queue, task);
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "unable to push back task_queue.\n");
        av_frame_free(&task->in_frame);
        av_frame_free(&task->out_frame);
        av_freep(&task);
        return ret;
    }

    ret = extract_lltask_from_task(task, th_model->lltask_queue);
    if (ret != 0) {
        av_log(ctx, AV_LOG_ERROR, "unable to extract last level task from task.\n");
        ff_queue_pop_back(th_model->task_queue);
        av_frame_free(&task->in_frame);
        av_frame_free(&task->out_frame);
        av_freep(&task);
        return ret;
    }

    request = (THRequestItem *)ff_safe_queue_pop_front(th_model->request_queue);
    if (!request) {
        av_log(ctx, AV_LOG_ERROR, "unable to get infer request.\n");
        LastLevelTaskItem *lltask = (LastLevelTaskItem *)ff_queue_pop_back(th_model->lltask_queue);
        av_freep(&lltask);
        ff_queue_pop_back(th_model->task_queue);
        av_frame_free(&task->in_frame);
        av_frame_free(&task->out_frame);
        av_freep(&task);
        return AVERROR(EINVAL);
    }

    return execute_model_th(request, th_model->lltask_queue);
}

static DNNAsyncStatusType dnn_get_result_th(const DNNModel *model, AVFrame **in, AVFrame **out)
{
    THModel *th_model = (THModel *)model;
    return ff_dnn_get_result_common(th_model->task_queue, in, out);
}

static int dnn_flush_th(const DNNModel *model)
{
    THModel *th_model = (THModel *)model;
    THRequestItem *request;

    if (ff_queue_size(th_model->lltask_queue) == 0)
        // no pending task need to flush
        return 0;

    request = (THRequestItem *)ff_safe_queue_pop_front(th_model->request_queue);
    if (!request) {
        av_log(th_model->ctx, AV_LOG_ERROR, "unable to get infer request.\n");
        return AVERROR(EINVAL);
    }

    return execute_model_th(request, th_model->lltask_queue);
}

extern const DNNModule ff_dnn_backend_torch = {
    .clazz          = DNN_DEFINE_CLASS(dnn_th),
    .type           = DNN_TH,
    .load_model     = dnn_load_model_th,
    .execute_model  = dnn_execute_model_th,
    .get_result     = dnn_get_result_th,
    .flush          = dnn_flush_th,
    .free_model     = dnn_free_model_th,
};
