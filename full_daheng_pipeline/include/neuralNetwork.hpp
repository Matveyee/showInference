
#ifndef NEURALNETWORK.HPP
#define NEURALNETWORK.HPP

#include "hailo/hailort.hpp"
#include "../include/utils.hpp"



class AbstractNNBase {
    
    public:

        virtual void inference(void* ctx) = 0;

};
//Abstract class for NeuralNetwork
template<typename InferenceContext>
class AbstractNN : public AbstractNNBase{

    public:

        virtual void inference(InferenceContext* ctx) = 0;

        virtual void init(std::string path) = 0;

        void inference(void* ctx) {
            inference(static_cast<InferenceContext*>(ctx));
        }

};




typedef struct {


    void* input_buffer;
    void* output_buffer;
    Projection proj;

    

} standart_inference_ctx;




#endif