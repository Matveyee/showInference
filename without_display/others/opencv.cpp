#include <opencv2/opencv.hpp>
#include <chrono>
#include <iostream>

int main() {
    cv::VideoCapture cap("full_mov_slow.mp4");
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 300; i++) {
        cv::Mat img;
        cap >> img;
        cv::imshow("Video",img);
        cv::waitKey(1);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "FPS = "<< 300 / (duration.count() / 1000);
}