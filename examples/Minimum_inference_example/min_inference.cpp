#include "hailo/hailort.hpp"
#include <iostream>

#if defined(__unix__)
#include <sys/mman.h>
#endif

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

using namespace hailort;

#define HEF_FILE "path_to_hef_model"

int main() {
    // Создание виртуального девайса
    auto vdevice = VDevice::create().expect("Failed create vdevice");
    // Создание импортирование модели из .hef файла
    auto infer_model = vdevice->create_infer_model(HEF_FILE).expect("Failed to create infer model");
    // Конфигурирование модели
    auto configured_infer_model = infer_model->configure().expect("Failed to create configured infer model");
    // Создание соответвий между входными и выходными слоями модели
    auto bindings = configured_infer_model.create_bindings().expect("Failed to create bindings");

    for (const auto &input_name : infer_model->get_input_names()) {
        size_t input_frame_size = infer_model->input(input_name)->get_frame_size();
        std::shared_ptr<uint8_t> input_buffer;
        // Аллоцирование буферов. Не обязетально через page_aligned_alloc, но рекомендуемо 
        input_buffer = page_aligned_alloc(input_frame_size);
        bindings.input(input_name)->set_buffer(MemoryView(input_buffer.get(), input_frame_size));
    }

    std::map<std::string, std::shared_ptr<uint8_t>> output_buffers;
    for (const auto &output_name : infer_model->get_output_names()) {
        size_t output_frame_size = infer_model->output(output_name)->get_frame_size();
        output_buffers[output_name] = page_aligned_alloc(output_frame_size);
        bindings.output(output_name)->set_buffer(MemoryView(output_buffers[output_name].get(), output_frame_size));
    }
    // Запуск инференса
    auto job = configured_infer_model.run_async(bindings,[&output_buffers](const AsyncInferCompletionInfo & info){
        std::cout << "Inference ended successfuly! " << std::endl;
        // Здесь можно обрабатывать результат работы нейросети, который находится в output_buffers
    }).expect("Failed to start async infer job");
    //  Можно использовать configured_infer_model.run с указаением времени ожидания результата инференса

}
