// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "postprocess.hpp"
#include "yolov6.h"
#include "common.h"
#include "file_utils.hpp"
#include "image_utils.h"
#include "utils.hpp"
#include <iostream>

// static void dump_tensor_attr(rknn_tensor_attr *attr)
// {
//     printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
//            "zp=%d, scale=%f\n",
//            attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
//            attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
//            get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
// }

int read_data_from_file(const char *path, char **out_data)
{
    FILE *fp = fopen(path, "rb");
    if(fp == NULL) {
        printf("fopen %s fail!\n", path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    int file_size = ftell(fp);
    char *data = (char *)malloc(file_size+1);
    data[file_size] = 0;
    fseek(fp, 0, SEEK_SET);
    if(file_size != fread(data, 1, file_size, fp)) {
        printf("fread %s fail!\n", path);
        free(data);
        fclose(fp);
        return -1;
    }
    if(fp) {
        fclose(fp);
    }
    *out_data = data;
    return file_size;
}



int init_yolov6_model(const char *model_path, rknn_app_context_t *app_ctx)
{
    rknn_context ctx = 0;
    int            model_len = 0;
    unsigned char* model     = load_model(model_path, &model_len);
    int            ret       = rknn_init(&ctx, model, model_len, 0 , NULL);
    rknn_core_mask core_mask = RKNN_NPU_CORE_0_1_2;
    int ret1 = rknn_set_core_mask(ctx, core_mask);
    std::cout << "NPU connected: " << ret << std::endl;
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }



    // Get Model Input Output Info
    
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &app_ctx->io_num, sizeof(app_ctx->io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    printf("model input num: %d, output num: %d\n", app_ctx->io_num.n_input, app_ctx->io_num.n_output);

    printf("input tensors:\n");
    
    rknn_tensor_attr input_attrs[app_ctx->io_num.n_input];
    app_ctx->input_attrs = input_attrs;
    memset(app_ctx->input_attrs, 0, app_ctx->io_num.n_input * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < app_ctx->io_num.n_input; i++) {
        app_ctx->input_attrs[i].index = i;
        // query info
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(app_ctx->input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
        printf("rknn_init error! ret=%d\n", ret);
        return -1;
        }
        dump_tensor_attr(&app_ctx->input_attrs[i]);
    }

    switch (input_attrs[0].fmt) {
    case RKNN_TENSOR_NHWC:
        app_ctx->model_height = app_ctx->input_attrs[0].dims[1];
        app_ctx->model_width  = app_ctx->input_attrs[0].dims[2];
        app_ctx->model_channel     = app_ctx->input_attrs[0].dims[3];
        break;
    case RKNN_TENSOR_NCHW:
        app_ctx->model_height = app_ctx->input_attrs[0].dims[2];
        app_ctx->model_width  = app_ctx->input_attrs[0].dims[3];
        app_ctx->model_channel     = app_ctx->input_attrs[0].dims[1];
        break;
    default:
        printf("meet unsupported layout\n");
    }
    int wstride = app_ctx->model_width + (8 - app_ctx->model_height % 8) % 8;
    int hstride = app_ctx->model_height;


    
    printf("output tensors:\n");
    rknn_tensor_attr output_attrs[app_ctx->io_num.n_output];
    app_ctx->output_attrs = output_attrs;
    memset(app_ctx->output_attrs, 0, app_ctx->io_num.n_output * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < app_ctx->io_num.n_output; i++) {
        app_ctx->output_attrs[i].index = i;
        // query info
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(app_ctx->output_attrs[i]), sizeof(rknn_tensor_attr));
        std::cout << "INIT size " << app_ctx->output_attrs[i].size << std::endl;
        if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
        }
        dump_tensor_attr(&app_ctx->output_attrs[i]);
    }
    app_ctx->rknn_ctx = ctx;
    return 0;
}

int release_yolov6_model(rknn_app_context_t *app_ctx)
{
    if (app_ctx->input_attrs != NULL)
    {
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL)
    {
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
    if (app_ctx->rknn_ctx != 0)
    {
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    return 0;
}
