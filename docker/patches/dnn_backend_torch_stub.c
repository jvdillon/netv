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
 * DNN LibTorch backend, loaded on first use.
 *
 * Linking libtorch into ffmpeg runs its static initializers on every launch
 * (~0.35 s), whether or not a torch filter runs. ffmpeg links this stub
 * instead: it carries the backend's options, and the first torch model load
 * dlopens the real backend (dnn_backend_torch.cpp built as
 * libffmpeg_dnn_torch.so, which links libtorch) and forwards to it.
 */

#include <dlfcn.h>
#include <stdio.h>

#include "libavutil/opt.h"
#include "libavutil/thread.h"
#include "../dnn_interface.h"
#include "dnn_backend_common.h"
#include "dnn_backend_torch_options.h"

#define TORCH_PLUGIN "libffmpeg_dnn_torch.so"

static const DNNModule *torch_impl;
static char torch_load_error[512];
static AVOnce torch_load_once = AV_ONCE_INIT;

static void load_torch_impl(void)
{
    void *handle = dlopen(TORCH_PLUGIN, RTLD_NOW | RTLD_GLOBAL);
    const char *error;

    if (handle)
        torch_impl = (const DNNModule *)dlsym(handle, "ff_dnn_backend_torch_impl");
    if (!torch_impl) {
        error = dlerror();
        snprintf(torch_load_error, sizeof(torch_load_error), "%s", error ? error : "unknown error");
    }
}

static DNNModel *dnn_load_model_th(DnnContext *ctx, DNNFunctionType func_type, AVFilterContext *filter_ctx)
{
    ff_thread_once(&torch_load_once, load_torch_impl);
    if (!torch_impl) {
        av_log(ctx, AV_LOG_ERROR, "Could not load the LibTorch backend (" TORCH_PLUGIN "): %s\n",
               torch_load_error);
        return NULL;
    }
    return torch_impl->load_model(ctx, func_type, filter_ctx);
}

// A model exists only after the backend loaded, so these always have torch_impl.
static int dnn_execute_model_th(const DNNModel *model, DNNExecBaseParams *exec_params)
{
    return torch_impl->execute_model(model, exec_params);
}

static DNNAsyncStatusType dnn_get_result_th(const DNNModel *model, AVFrame **in, AVFrame **out)
{
    return torch_impl->get_result(model, in, out);
}

static int dnn_flush_th(const DNNModel *model)
{
    return torch_impl->flush(model);
}

static void dnn_free_model_th(DNNModel **model)
{
    // Filter teardown frees the model even when loading never succeeded.
    if (torch_impl)
        torch_impl->free_model(model);
}

const DNNModule ff_dnn_backend_torch = {
    .clazz          = DNN_DEFINE_CLASS(dnn_th),
    .type           = DNN_TH,
    .load_model     = dnn_load_model_th,
    .execute_model  = dnn_execute_model_th,
    .get_result     = dnn_get_result_th,
    .flush          = dnn_flush_th,
    .free_model     = dnn_free_model_th,
};
