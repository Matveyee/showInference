#include "../include/rockChipNN.hpp"
#include "../include/rockChipPostprocess.hpp"

RockChipNN::RockChipNN(std::string model_path) {
    init(model_path);
}

void RockChipNN::init(std::string model_path) {

    int            model_len = 0;
    unsigned char* model     = load_model(model_path.c_str(), &model_len);
    int            ret       = rknn_init(&ctx, model, model_len, 0 , NULL);
    rknn_core_mask core_mask = RKNN_NPU_CORE_0_1_2;
    int ret1 = rknn_set_core_mask(ctx, core_mask);
    std::cout << "NPU connected: " << ret << std::endl;
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return;
    }



    // Get Model Input Output Info
    
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return;
    }
    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    printf("input tensors:\n");
    // rknn_tensor_attr input_attrs[io_num.n_input];
    input_attrs = (rknn_tensor_attr*)calloc(io_num.n_input, sizeof(rknn_tensor_attr));
    memset(input_attrs, 0, io_num.n_input * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        // query info
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
        printf("rknn_init error! ret=%d\n", ret);
        return;
        }
        dump_tensor_attr(&input_attrs[i]);
    }

    switch (input_attrs[0].fmt) {
    case RKNN_TENSOR_NHWC:
        model_in_height = input_attrs[0].dims[1];
        model_in_width  = input_attrs[0].dims[2];
        req_channel     = input_attrs[0].dims[3];
        break;
    case RKNN_TENSOR_NCHW:
        model_in_height = input_attrs[0].dims[2];
        model_in_width  = input_attrs[0].dims[3];
        req_channel     = input_attrs[0].dims[1];
        break;
    default:
        printf("meet unsupported layout\n");
    }
    wstride = model_in_width + (8 - model_in_width % 8) % 8;
    hstride = model_in_height;


    
    printf("output tensors:\n");
    // rknn_tensor_attr output_attrs[io_num.n_output];
    output_attrs = (rknn_tensor_attr*)calloc(io_num.n_output, sizeof(rknn_tensor_attr));
    memset(output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        // query info
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return;
        }
        dump_tensor_attr(&output_attrs[i]);
    }

}

void RockChipNN::inference(standart_inference_ctx* app_ctx) {

    rknn_input input;
    input.index = 0;
    input.pass_through = 0;
    input.type = (rknn_tensor_type)RKNN_TENSOR_UINT8;
    input.fmt = (rknn_tensor_format)RKNN_TENSOR_NHWC;
    input.buf =  app_ctx->input_buffer;
    input.size = 640 * 640 * 3;
    int ret = rknn_inputs_set(ctx, io_num.n_input, &input);



    int ret = rknn_run(ctx, NULL);

    rknn_output* outputs = (rknn_output*)calloc(io_num.n_output, sizeof(rknn_output));
  
    for (uint32_t i = 0; i < io_num.n_output; ++i) {
    
        outputs[i].want_float  = 1;
        outputs[i].index       = i;
        
    }

    rknn_outputs_get(ctx, io_num.n_output, outputs, NULL);

    app_ctx->output_buffer = outputs;

    if (ret < 0) {
        printf("rknn run error %d\n", ret);
    }

}
