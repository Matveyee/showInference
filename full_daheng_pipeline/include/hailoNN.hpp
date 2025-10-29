#include "../include/neuralNetwork.hpp"

class HailoNN : public AbstractNN<standart_inference_ctx> {

    public:

        std::unique_ptr<hailort::VDevice> vdevice;
        std::shared_ptr<hailort::InferModel> infer_model;
        hailort::ConfiguredInferModel configured_infer_model;
        

        HailoNN();

        HailoNN(std::string path);

        void init(std::string path) override;

        void inference(standart_inference_ctx* ctx) override;

};