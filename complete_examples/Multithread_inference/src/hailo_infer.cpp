#include "../include/hailo_infer.hpp"
#if defined(__unix__)
#include <sys/mman.h>
#endif



struct NamedBbox {
    hailo_bbox_float32_t bbox;
    size_t class_id;
};

std::vector<NamedBbox> parse_nms_data(uint8_t* data, size_t max_class_count) {
    std::vector<NamedBbox> bboxes;
    size_t offset = 0;

    for (size_t class_id = 0; class_id < max_class_count; class_id++) {
        auto det_count = static_cast<uint32_t>(*reinterpret_cast<float32_t*>(data + offset));
        offset += sizeof(float32_t);

        for (size_t j = 0; j < det_count; j++) {
            hailo_bbox_float32_t bbox_data = *reinterpret_cast<hailo_bbox_float32_t*>(data + offset);
            offset += sizeof(hailo_bbox_float32_t);

            NamedBbox named_bbox;
            named_bbox.bbox = bbox_data;
            named_bbox.class_id = class_id + 1;
            bboxes.push_back(named_bbox);
        }
    }
    return bboxes;
}

void draw_line(uint8_t* map, int x, int y, int x1, int y1, uint32_t pitch) {
    uint32_t pixel = (0xFF << 24) | (255 << 16) | (0 << 8) | 0;
    if (x == x1) {
        for (int i = y; i <= y1; i++) {
            ((uint32_t*)(map + i * pitch))[x] = pixel;
        }
    } else if (y == y1) {
        for (int i = x; i <= x1; i++) {
            ((uint32_t*)(map + y * pitch))[i] = pixel;
        }
    }
    
}

void draw_rect(uint8_t* map, int x, int y, int x1, int y1, uint32_t pitch) {
    draw_line(map,x,y,x1,y, pitch);
    draw_line(map,x,y1,x1,y1, pitch);
    draw_line(map,x,y,x,y1, pitch);
    draw_line(map,x1,y,x1,y1, pitch);
}


void draw_bounding_boxes(uint8_t* map, const std::vector<NamedBbox>& bboxes, int width, int height, uint32_t pitch) {


    for (const auto& named_bbox : bboxes) {
        hailo_bbox_float32_t bbox = named_bbox.bbox;
        int x = static_cast<int>(bbox.x_min * width);
        int y = static_cast<int>(bbox.y_min * height);
        int x1 = x + static_cast<int>((bbox.x_max - bbox.x_min) * width);
        int y1 = y + static_cast<int>((bbox.y_max - bbox.y_min) * height);
//        std::cout << "x = " << x << ", y = " << y << ", x1 = " << x1 << ", y1 = " << y1 << std::endl;
        if (x1 > 0 && x > 0 && y > 0 && y1 > 0) {
            draw_rect(map, x , y, x1 ,y1, pitch);
        }
    }
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
void inference(
    hailort::ConfiguredInferModel &configured_model,
    std::shared_ptr<hailort::InferModel> infer_model,
    Queue<int>& queue,
    int& index,
    std::vector<double>& times,
    uint8_t* map,
    uint32_t pitch
    )
{
    auto &infer_model1 = configured_model;

    // создаём bindings
    auto bindings_exp = infer_model1.create_bindings();
    if (!bindings_exp) {
        std::cerr << "Failed to create bindings: " << bindings_exp.status() << std::endl;
        return;
    }
    auto bindings = std::move(bindings_exp.value());

    // -----------------------------------------------------
    //                  В Х О Д   (DMA)
    // -----------------------------------------------------
    for (const auto &input_name : infer_model->get_input_names()) {

        // получаем входной stream
        auto input_stream_exp = bindings.input(input_name);
        if (!input_stream_exp) {
            std::cerr << "Failed to get input stream: "
                    << input_stream_exp.status() << std::endl;
            return;
        }
        auto input_stream = input_stream_exp.value();

        // узнаём размер входного кадра
        size_t input_frame_size = infer_model->input(input_name)->get_frame_size();

        // готовим dma buffer
        hailo_dma_buffer_t dma_buf{};
        dma_buf.fd   = queue.read();             // <<< вот он — fd от RGA
        dma_buf.size = input_frame_size;   // 640*640*3

        auto status = input_stream.set_dma_buffer(dma_buf);
        if (status != HAILO_SUCCESS) {
            std::cerr << "set_dma_buffer failed: " << status << std::endl;
            return;
        }
    }

    // -----------------------------------------------------
    //                 В Ы Х О Д Ы  (CPU buffer)
    // -----------------------------------------------------

    std::vector<std::shared_ptr<uint8_t>> output_buffers;

    for (const auto &output_name : infer_model->get_output_names()) {

        size_t output_frame_size = infer_model->output(output_name)->get_frame_size();

        // CPU (page aligned) buffer
        auto out_buf = page_aligned_alloc(output_frame_size);
        output_buffers.push_back(out_buf);

        // привязываем MemoryView к bindings
        auto output_stream = bindings.output(output_name).value();
        auto status = output_stream.set_buffer(
            hailort::MemoryView(out_buf.get(), output_frame_size)
        );

        if (status != HAILO_SUCCESS) {
            std::cerr << "failed to set output buffer: " << status << std::endl;
            return;
        }
    }

    // -----------------------------------------------------
    //                 З А П У С К   I N F E R
    // -----------------------------------------------------

    auto start = std::chrono::high_resolution_clock::now();

    auto job_exp = infer_model1.run_async(
        bindings,
        [&, start](const hailort::AsyncInferCompletionInfo &info) {
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> duration = end - start;
            times.push_back(duration.count());
            // std::cout << "FPS = " << 1 / ( duration.count() / 1000) << std::endl;
            //std::cout << "Inferenced" << std::endl;
            index++;
            close(queue.queue_.front());
            queue.pop();
            // auto end = std::chrono::high_resolution_clock::now();
            // std::chrono::duration<double, std::milli> duration = end - start;
            // times.push_back(duration.count() / 1000);

            // processed_index++;

            // // Потом ты сможешь распарсить:
            // auto bboxes = parse_nms_data(output_buffers[0].get(), 80);

            // draw_bounding_boxes(map, bboxes, 640, 640, pitch);

            // // очищаем очередь
            // std::unique_lock<std::mutex> lock(queue_mutex);
            // queue.pop_front();
        }
    );

    if (!job_exp) {
        std::cerr << "Failed to start async infer job: " << job_exp.status() << std::endl;
        return;
    }

    // auto job = job_exp.value();

}
