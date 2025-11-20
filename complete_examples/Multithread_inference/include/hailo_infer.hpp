#include "hailo/hailort.hpp"
#include "queue.hpp"
#include <iostream>
#include <vector>

#pragma once


void inference(
    hailort::ConfiguredInferModel& configured_model,
    std::shared_ptr<hailort::InferModel> infer_model,
    Queue<int>& queue,
    int& index,
    std::vector<double>& times,
    uint8_t* map,
    uint32_t pitch
);