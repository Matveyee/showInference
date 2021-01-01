#include "opencv2/opencv.hpp"
#include "hailo/hailort.hpp"
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>
#if defined(__unix__)
#include <sys/mman.h>
#endif

using namespace hailort;

static std::shared_ptr<uint8_t> page_aligned_alloc(size_t size) {
#if defined(__unix__)
    auto addr = mmap(NULL, size, PROT_WRITE | PROT_READ, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (MAP_FAILED == addr) throw std::bad_alloc();
    return std::shared_ptr<uint8_t>(reinterpret_cast<uint8_t*>(addr), [size](void *addr) { munmap(addr, size); });
#else
#pragma error("Aligned alloc not supported")
#endif
}
int completed = 0;
int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cout << "Usage:\n" << argv[0] << " model.hef video.mp4 FPS FRAME_COUNT\n";
        return 1;
    }

    std::string hef_path = argv[1];
    std::string video_path = argv[2];
    int FPS = std::stoi(argv[3]);
    int COUNT = std::stoi(argv[4]);

    // Load video
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video\n";
        return 1;
    }

    // Load model
    auto vdevice = VDevice::create().expect("Failed to create vdevice");
    auto infer_model = vdevice->create_infer_model(hef_path).expect("Failed to create infer model");
    auto configured_infer_model = infer_model->configure().expect("Failed to configure model");

    // Read all frames into RAM
    std::vector<cv::Mat> input_frames;
    for (int i = 0; i < COUNT; i++) {
        cv::Mat frame, resized;
        cap >> frame;
        if (frame.empty()) break;
        cv::resize(frame, resized, cv::Size(640, 640));  // адаптировать под размер модели
        input_frames.push_back(resized.clone());
    }

    // Статистика времени
    std::vector<double> times(input_frames.size());
    
    std::mutex mtx;
    std::condition_variable cv;

    auto start_all = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < input_frames.size(); i++) {
        auto bindings = configured_infer_model.create_bindings().expect("Failed to create bindings");

        for (const auto &input_name : infer_model->get_input_names()) {
            size_t input_size = infer_model->input(input_name)->get_frame_size();
            bindings.input(input_name)->set_buffer(MemoryView(input_frames[i].data, input_size));
        }

        std::shared_ptr<uint8_t> output_buffer;
        for (const auto &output_name : infer_model->get_output_names()) {
            size_t output_size = infer_model->output(output_name)->get_frame_size();
            output_buffer = page_aligned_alloc(output_size);
            bindings.output(output_name)->set_buffer(MemoryView(output_buffer.get(), output_size));
        }

        // Время запуска для этого кадра
        auto start = std::chrono::high_resolution_clock::now();

        auto job = configured_infer_model.run_async(bindings, [&, i, start](const AsyncInferCompletionInfo &) {
            auto end = std::chrono::high_resolution_clock::now();
            times[i] = std::chrono::duration<double, std::milli>(end - start).count();

            if (++completed == (int)input_frames.size()) {
                cv.notify_one();
            }
        });

        if (!job) {
            std::cerr << "Failed to run async job\n";
            return 1;
        }
    }

    // Ждем завершения всех инференсов
    {
        std::unique_lock<std::mutex> lock(mtx);
        cv.wait(lock, [&]() { return completed == (int)input_frames.size(); });
    }

    auto end_all = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_time = end_all - start_all;

    // Расчет FPS
    double sum = 0;
    for (auto t : times) sum += t;
    double avg_inf_time = sum / times.size();
    double avg_inf_fps = 1000.0 / avg_inf_time;
    double total_fps = times.size() / total_time.count();

    std::cout << "Average inference time per frame: " << avg_inf_time << " ms\n";
    std::cout << "Average inference FPS (per frame): " << avg_inf_fps << "\n";
    std::cout << "Total FPS (pipeline): " << total_fps << "\n";

    return 0;
}