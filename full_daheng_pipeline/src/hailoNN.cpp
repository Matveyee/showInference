#include "../include/hailoNN.hpp"
#include "../include/hailoPostprocess.hpp"

HailoNN::HailoNN() {}

HailoNN::HailoNN(std::string path) {

    init(path);

}

void HailoNN::init(std::string path) {

    vdevice = hailort::VDevice::create().expect("Failed to create vdevice");
    infer_model = vdevice->create_infer_model(path).expect("Failed to create infer model");
    configured_infer_model = infer_model->configure().expect("Failed to configure model");

}

static std::shared_ptr<uint8_t> page_aligned_alloc(size_t size)
{
#if defined(__unix__)
    auto addr = mmap(NULL, size, PROT_WRITE | PROT_READ, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (MAP_FAILED == addr) throw std::bad_alloc();
    return std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(addr), [size](void *addr) { munmap(addr, size); });
#elif defined(_MSC_VER)
    auto addr = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!addr) throw std::bad_alloc();
    return std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(addr), [](void *addr){ VirtualFree(addr, 0, MEM_RELEASE); });
#else
#pragma error("Aligned alloc not supported")
#endif
}


void HailoNN::inference(standart_inference_ctx* ctx) {
    log("Inference entered");
    auto &infer_model1 = configured_infer_model;
    auto bindings = infer_model1.create_bindings().expect("Failed to create bindings");
    
    int i = 0;
    for (const auto &input_name : infer_model->get_input_names()) {
        size_t input_frame_size = infer_model->input(input_name)->get_frame_size();
      
        bindings.input(input_name)->set_buffer(hailort::MemoryView(ctx->input_buffer, input_frame_size));
        i++;
    }

    log("Inputs set");                          

    int k = 0;
    for (const auto &output_name : infer_model->get_output_names()) {
        size_t output_frame_size = infer_model->output(output_name)->get_frame_size();
        ctx->output_buffer = page_aligned_alloc(output_frame_size);
        bindings.output(output_name)->set_buffer(hailort::MemoryView(ctx->output_buffer.get(), output_frame_size));

        k++;
    }


    auto start = std::chrono::high_resolution_clock::now();
    log("Trying to inference");
    auto job = infer_model1.run_async(bindings,[&](const hailort::AsyncInferCompletionInfo & info){
        log("Inferenced");
        free(ctx->input_buffer);
        auto bboxes = parse_nms_data(ctx->output_buffer.get(), 80);

        draw_bounding_boxes(map, bboxes, 640, 640, pitch, ctx->proj);
   
    }).expect("Failed to start async infer job");
    

}


