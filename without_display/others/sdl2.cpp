#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <SDL2/SDL.h>
#include <iostream>

#define WINDOW_WIDTH 640
#define WINDOW_HEIGHT 640

// Callback для получения новых кадров
GstFlowReturn on_new_sample(GstAppSink *appsink, gpointer user_data) {
    auto *context = reinterpret_cast<YourContextStruct *>(user_data);

    GstSample *sample = gst_app_sink_pull_sample(appsink);
    if (!sample) return GST_FLOW_OK;

    GstBuffer *buffer = gst_sample_get_buffer(sample);
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }

    // ----------- HAILO INFERENCE ------------
    auto &infer_model = context->configured_infer_model;
    auto bindings = infer_model.create_bindings().expect("Failed to create bindings");

    for (const auto &input_name : context->infer_model->get_input_names()) {
        size_t input_frame_size = context->infer_model->input(input_name)->get_frame_size();
        bindings.input(input_name)->set_buffer(MemoryView(map.data, input_frame_size));
    }

    std::shared_ptr<uint8_t> output_buffer;
    for (const auto &output_name : context->infer_model->get_output_names()) {
        size_t output_frame_size = context->infer_model->output(output_name)->get_frame_size();
        output_buffer = page_aligned_alloc(output_frame_size);
        bindings.output(output_name)->set_buffer(MemoryView(output_buffer.get(), output_frame_size));
    }

    context->configured_infer_model.run(bindings).expect("Failed to run inference");

    // ----------- SDL DRAW -------------------
    SDL_UpdateYUVTexture(context->texture, nullptr,
                         map.data, context->width,
                         map.data + context->width * context->height, context->width / 2,
                         map.data + context->width * context->height * 5 / 4, context->width / 2);

    SDL_RenderClear(context->renderer);
    SDL_RenderCopy(context->renderer, context->texture, nullptr, nullptr);
    SDL_RenderPresent(context->renderer);

    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);

    return GST_FLOW_OK;
}

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);
    SDL_Init(SDL_INIT_VIDEO);

    SDL_Window *window = SDL_CreateWindow("GStreamer + SDL2",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_Texture *texture = SDL_CreateTexture(renderer,
        SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING,
        WINDOW_WIDTH, WINDOW_HEIGHT);

    void *userdata[2] = { renderer, texture };

    std::string pipeline_str =
        "filesrc location=test.mp4 ! decodebin ! videoconvert ! video/x-raw,format=RGB,width=640,height=640 ! appsink name=sink";

    GError *error = nullptr;
    GstElement *pipeline = gst_parse_launch(pipeline_str.c_str(), &error);
    if (!pipeline) {
        std::cerr << "Failed to create pipeline: " << error->message << std::endl;
        return -1;
    }

    GstElement *appsink = gst_bin_get_by_name(GST_BIN(pipeline), "sink");
    g_object_set(appsink, "emit-signals", TRUE, "sync", FALSE, NULL);
    g_signal_connect(appsink, "new-sample", G_CALLBACK(on_new_sample), userdata);

    gst_element_set_state(pipeline, GST_STATE_PLAYING);

    SDL_Event event;
    bool quit = false;
    while (!quit) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                quit = true;
        }
        SDL_Delay(10);
    }

    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}
