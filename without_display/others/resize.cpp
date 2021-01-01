#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Использование: " << argv[0] << " input_video.mp4\n";
        return 1;
    }

    std::string input_path = argv[1];
    std::string output_path = "resized_640x640.mp4";

    // Открытие входного видео
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) {
        std::cerr << "Не удалось открыть видеофайл: " << input_path << "\n";
        return 1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v'); // можно заменить на 'XVID', 'H264' и т.п.

    // Создание объекта для записи
    cv::VideoWriter out(output_path, fourcc, fps, cv::Size(640, 640));
    if (!out.isOpened()) {
        std::cerr << "Не удалось открыть выходной файл для записи\n";
        return 1;
    }

    cv::Mat frame, resized;
    while (cap.read(frame)) {
        cv::resize(frame, resized, cv::Size(640, 640));
        out.write(resized);
    }

    cap.release();
    out.release();

    std::cout << "Готово: сохранено в " << output_path << "\n";
    return 0;
}
