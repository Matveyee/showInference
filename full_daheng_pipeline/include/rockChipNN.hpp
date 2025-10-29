#include "../include/neuralNetwork.hpp"

//Example for RKNN

class RockChipNN : public AbstractNN<standart_inference_ctx> {

    public:

        rknn_context ctx;

        rknn_input_output_num io_num;

        rknn_tensor_attr* input_attrs;

        rknn_tensor_attr* output_attrs;

        int model_in_height;

        int model_in_width;

        int req_channel;

        int wstride;

        int hstride;

        RockChipNN();

        RockChipNN(std::string path);

        void init(std::string path) override;

        void inference(standart_inference_ctx* ctx) override;


};