#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <queue>
#include "../include/queue.hpp"
#include "../include/inc/rk_mpi.h"
#include <thread>
#define PACKET_SIZE 8192
#include <sys/mman.h>

#include "../include/rga/RgaUtils.h"
#include "../include/rga/im2d.h"
#include "../include/rga/rga.h"
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/dma-heap.h>
#include "../include/hailo_infer.hpp"


#include <xf86drm.h>
#include <xf86drmMode.h>
#include <libdrm/drm_mode.h>

using namespace hailort;


struct drm_mode_fb_cmd fb = {};
int drm_fd;
uint8_t* map = nullptr;
uint32_t fb_id;
uint32_t handle;
uint32_t pitch;
uint64_t size;
drmModeCrtc *old_crtc = nullptr;
int SIZE;
Queue<int> queue;
Queue<MppFrame> queue_raw;
std::vector<double> times;

hailort::ConfiguredInferModel configured_model;
std::shared_ptr<hailort::InferModel> infer_model;

int processed_index = 0;
int decoded_index = 0;
int converted_index = 0;
std::chrono::_V2::system_clock::time_point now() {
    return std::chrono::high_resolution_clock::now();
}

typedef std::chrono::duration<double, std::milli> duration;
    

MPP_RET handle_info_change(MppCtx ctx, MppApi *mpi,
                           MppFrame frame,
                           MppBufferGroup &ext_group,
                           bool &ext_inited)
{
    int w  = mpp_frame_get_width(frame);
    int h  = mpp_frame_get_height(frame);
    int hs = mpp_frame_get_hor_stride(frame);
    int vs = mpp_frame_get_ver_stride(frame);
    size_t buf_size = mpp_frame_get_buf_size(frame);

    std::cout << "INFO_CHANGE: " << w << "x" << h
              << " stride " << hs << "x" << vs
              << " buf_size " << buf_size << std::endl;

    if (!ext_inited) {
        // 1) создаём внешнюю группу
        MPP_RET ret = mpp_buffer_group_get_external(&ext_group, MPP_BUFFER_TYPE_DMA_HEAP);
        if (ret) {
            std::cout << "mpp_buffer_group_get_external failed: " << ret << std::endl;
            return ret;
        }

        int buffer_count = 50;
        int heap_fd = open("/dev/dma_heap/system", O_RDWR);
        if (heap_fd < 0) {
            perror("open /dev/dma_heap/system");
            return MPP_NOK;
        }

        for (int i = 0; i < buffer_count; i++) {
            struct dma_heap_allocation_data req;
            memset(&req, 0, sizeof(req));
            req.len       = buf_size;
            req.fd_flags  = O_RDWR | O_CLOEXEC;

            if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &req) < 0) {
                perror("DMA_HEAP_IOCTL_ALLOC");
                close(heap_fd);
                return MPP_NOK;
            }

            int fd = req.fd;

            MppBufferInfo info;
            memset(&info, 0, sizeof(info));
            info.type  = MPP_BUFFER_TYPE_DMA_HEAP;
            info.size  = buf_size;
            info.fd    = fd;
            info.index = i;

            MppBuffer mbuf;
            ret = mpp_buffer_import(&mbuf, &info);
            if (ret) {
                std::cout << "mpp_buffer_import failed: " << ret << std::endl;
                close(heap_fd);
                return ret;
            }

            ret = mpp_buffer_commit(ext_group, &info);
            if (ret) {
                std::cout << "mpp_buffer_commit failed: " << ret << std::endl;
                close(heap_fd);
                return ret;
            }
        }

        close(heap_fd);

        // 3) Сообщаем декодеру
        MPP_RET ret2 = mpi->control(ctx, MPP_DEC_SET_EXT_BUF_GROUP, ext_group);
        if (ret2) {
            std::cout << "MPP_DEC_SET_EXT_BUF_GROUP failed: " << ret2 << std::endl;
            return ret2;
        }

        ext_inited = true;
    }

    // 4) Подтверждаем info_change
    mpi->control(ctx, MPP_DEC_SET_INFO_CHANGE_READY, NULL);
    return MPP_OK;
}

int nv12_to_rgb_with_rga(MppFrame frame) {
    rga_buffer_t src;
    rga_buffer_t dst;
    rga_buffer_t final_dst;

    MppBuffer mpp_buffer = mpp_frame_get_buffer(frame);
    int src_fd = mpp_buffer_get_fd(mpp_buffer);

    // NV12 WIDTHxHEIGHT dma-buffer Image

    src = wrapbuffer_fd_t(
        src_fd,
        mpp_frame_get_width(frame), 
        mpp_frame_get_height(frame), 
        mpp_frame_get_hor_stride(frame), 
        mpp_frame_get_ver_stride(frame), 
        RK_FORMAT_YCbCr_420_SP);
    

    int heap_fd = open("/dev/dma_heap/system", O_RDWR);
    if (heap_fd < 0) {
        perror("open /dev/dma_heap/system");
        mpp_frame_deinit(&frame);
    }

    // creating dma-buffer

    struct dma_heap_allocation_data req;
    memset(&req, 0, sizeof(req));
    req.len       = SIZE*SIZE*3/2;
    req.fd_flags  = O_RDWR | O_CLOEXEC;
    req.heap_flags = 0;

    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &req) < 0) {
        perror("DMA_HEAP_IOCTL_ALLOC");
        close(heap_fd);
        mpp_frame_deinit(&frame);
    }

    int dst_fd = req.fd;

    //

    // NV12 640x640 dma-buffer Image
    dst = wrapbuffer_fd(dst_fd, SIZE, SIZE, RK_FORMAT_YCbCr_420_SP);

    IM_STATUS status = imresize(src, dst);

    mpp_frame_deinit(&frame);
    queue_raw.pop();

    //releasing mpp buffers

 

    // creating rgb dma buffer

    memset(&req, 0, sizeof(req));
    req.len       = SIZE*SIZE*3;
    req.fd_flags  = O_RDWR | O_CLOEXEC;
    req.heap_flags = 0;

    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &req) < 0) {
        perror("DMA_HEAP_IOCTL_ALLOC");
        close(heap_fd);
        mpp_frame_deinit(&frame);
    }

    int rgb_fd = req.fd;

    final_dst = wrapbuffer_fd(rgb_fd, SIZE, SIZE, RK_FORMAT_RGB_888);

    status = imcvtcolor(dst, final_dst, RK_FORMAT_YCbCr_420_SP, RK_FORMAT_RGB_888);

    close(dst_fd);
    close(heap_fd);

    return rgb_fd;


}


int stop = 0;


int REALTIME;
int FRAMES;
void inference_thread() {
    while (true) {
        if (stop == 1 && REALTIME == 1) {
            break;
        }
        if (processed_index == FRAMES) {
            break;
        }
        if (queue.size == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        inference(configured_model, infer_model, queue, processed_index, times, map, pitch);

    }
}

void convertation_thread() {
    while (true) {
        if (stop == 1 && REALTIME == 1) {
            break;
        }
        if (converted_index == FRAMES) {
            break;
        }
        if (queue_raw.size == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        int fd = nv12_to_rgb_with_rga(queue_raw.read());
        queue.push(fd);
        converted_index++;

    }
}

int main(int argc, char **argv) {

    if (argc < 6) {
        printf("Usage: %s input.h264 model.hef fps frames realtime size\n", argv[0]);
        return -1;
    }

    SIZE = std::stoi(argv[6]);
    std::cout << SIZE << std::endl;

    drm_fd = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
    if (drm_fd < 0) {
        perror("open");
        return 1;
    }

    drmModeRes* res = drmModeGetResources(drm_fd);
    drmModeConnector* conn = nullptr;
    drmModeModeInfo mode;
    uint32_t conn_id = 0;

    for (int i = 0; i < res->count_connectors; ++i) {
        conn = drmModeGetConnector(drm_fd, res->connectors[i]);
        if (conn->connection == DRM_MODE_CONNECTED && conn->count_modes > 0) {
            mode = conn->modes[0];
            conn_id = conn->connector_id;
            break;
        }
        drmModeFreeConnector(conn);
    }

    if (!conn_id) {
        std::cerr << "No connected display found\n";
        return 1;
    }

    drmModeEncoder* enc = drmModeGetEncoder(drm_fd, conn->encoder_id);
    uint32_t crtc_id = enc->crtc_id;
    old_crtc = drmModeGetCrtc(drm_fd, crtc_id);

    struct drm_mode_create_dumb create = {};
    create.width = mode.hdisplay;
    create.height = mode.vdisplay;
    create.bpp = 32;
    ioctl(drm_fd, DRM_IOCTL_MODE_CREATE_DUMB, &create);
    handle = create.handle;
    pitch = create.pitch;
    size = create.size;

    struct drm_mode_map_dumb map_dumb = {};
    map_dumb.handle = handle;
    ioctl(drm_fd, DRM_IOCTL_MODE_MAP_DUMB, &map_dumb);
    map = (uint8_t*)mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, drm_fd, map_dumb.offset);

    
    fb.width = mode.hdisplay;
    fb.height = mode.vdisplay;
    fb.pitch = pitch;
    fb.bpp = 32;
    fb.depth = 24;
    fb.handle = handle;
    drmModeAddFB(drm_fd, fb.width, fb.height, fb.depth, fb.bpp, pitch, handle, &fb_id);

    drmModeSetCrtc(drm_fd, crtc_id, fb_id, 0, 0, &conn_id, 1, &mode);

    int FPS = std::stoi(argv[3]);
    FRAMES = std::stoi(argv[4]);
    REALTIME = std::stoi(argv[5]);
    auto vdevice_exp = VDevice::create();
    if (!vdevice_exp) {
        std::cerr << "Failed to create VDevice: " << vdevice_exp.status() << std::endl;
        return vdevice_exp.status();
    }
    std::unique_ptr<VDevice> vdevice = std::move(vdevice_exp.value());

    auto hef_exp = Hef::create(argv[2]);
    if (!hef_exp) {
        std::cerr << "Failed to create Hef: " << hef_exp.status() << std::endl;
        return hef_exp.status();
    }
    Hef hef = std::move(hef_exp.value());


    auto infer_model_exp = vdevice->create_infer_model(hef);
    if (!infer_model_exp) {
        std::cerr << "Failed to create InferModel: " << infer_model_exp.status() << std::endl;
        return infer_model_exp.status();
    }
    infer_model = infer_model_exp.value();


    auto configured_exp = infer_model->configure();
    if (!configured_exp) {
        std::cerr << "Failed to configure InferModel: " << configured_exp.status() << std::endl;
        return configured_exp.status();
    }
    configured_model = std::move(configured_exp.value());


    const char *filename = argv[1];
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("fopen");
        return -1;
    }

    MppCtx ctx   = nullptr;
    MppApi *mpi  = nullptr;
    MPP_RET ret  = MPP_OK;

    // Создаём контекст
    ret = mpp_create(&ctx, &mpi);
    if (ret) {
        std::cout << "mpp_create failed: " << ret << std::endl;
        return -1;
    }

    // Включаем split mode парсера (не обязательно, но пусть будет)
    {
        int param = 1;
        ret = mpi->control(ctx, MPP_DEC_SET_PARSER_SPLIT_MODE, &param);
        if (ret) {
            std::cout << "set split mode failed: " << ret << std::endl;
            // не критично, можно продолжать
        }
    }

    // Инициализация декодера для H.264
    ret = mpp_init(ctx, MPP_CTX_DEC, MPP_VIDEO_CodingAVC);
    if (ret) {
        std::cout << "mpp_init failed: " << ret << std::endl;
        return -1;
    }

    // Буфер для чтения кусков bitstream
    unsigned char *buf = (unsigned char *)malloc(PACKET_SIZE);
    if (!buf) {
        std::cout << "malloc failed" << std::endl;
        return -1;
    }

    bool eos = false;
    bool ext_group_inited = false;

    MppBufferGroup ext_group = nullptr;
    std::thread inf;
    std::thread conv;
    if (REALTIME == 1 ) {
        inf = std::thread(inference_thread);
        conv = std::thread(convertation_thread);
    }
    // std::thread inf(inference_thread);
    auto start = now();
    int started = 0;
    auto gen_start = now();
    while (!eos) {
        
        int read_size = fread(buf, 1, PACKET_SIZE, fp);
        if (read_size <= 0) {
            // достигли конца файла – поставим eos-флаг
            eos = true;
        }

        // Если ещё есть данные – отправляем их в декодер
        if (!eos) {
            MppPacket packet = nullptr;

            // ВАЖНО: третий аргумент – размер буфера, а не длина данных
            ret = mpp_packet_init(&packet, buf, PACKET_SIZE);
            if (ret) {
                std::cout << "mpp_packet_init failed: " << ret << std::endl;
                break;
            }

            mpp_packet_set_pos(packet, buf);
            mpp_packet_set_length(packet, read_size);

        put_again:
            ret = mpi->decode_put_packet(ctx, packet);

            if (ret == MPP_OK) {
                // пакет успешно принят
            } else if (ret == MPP_ERR_BUFFER_FULL) {
                // Очередь входных пакетов переполнена – нужно вытащить кадры
                while (1) {
                    MppFrame frm = nullptr;
                    MPP_RET r2 = mpi->decode_get_frame(ctx, &frm);
                    if (r2 != MPP_OK || !frm)
                        break;

                    if (mpp_frame_get_info_change(frm)) {
                        handle_info_change(ctx, mpi, frm, ext_group, ext_group_inited);
                        mpp_frame_deinit(&frm);
                        continue;
                    }

                    // Здесь можно обработать кадр (но сейчас просто освобождаем)
                    mpp_frame_deinit(&frm);
                }

                usleep(1000);
                goto put_again;
            } else {
                std::cout << "decode_put_packet error: " << ret << std::endl;
                mpp_packet_deinit(&packet);
                break;
            }

            mpp_packet_deinit(&packet);
        } else {
            // EOF: можно отправить пустой пакет с EOS-флагом,
            // чтобы вытащить хвост B-кадров
            MppPacket packet = nullptr;
            ret = mpp_packet_init(&packet, NULL, 0);
            if (!ret) {
                mpp_packet_set_eos(packet);
                mpi->decode_put_packet(ctx, packet);
                mpp_packet_deinit(&packet);
            }
        }

        // После КАЖДОГО успешного put – читаем все доступные кадры
        int br = 0;
        while (1) {
            MppFrame frame = nullptr;
            ret = mpi->decode_get_frame(ctx, &frame);
            if (ret != MPP_OK)
                break;
            if (!frame)
                break;

            if (mpp_frame_get_info_change(frame)) {
                handle_info_change(ctx, mpi, frame, ext_group, ext_group_inited);
                mpp_frame_deinit(&frame);
                continue;
            }

            
            if (!mpp_frame_get_errinfo(frame)) {

                decoded_index++;
                // int w  = mpp_frame_get_width(frame);
                // int h  = mpp_frame_get_height(frame);

                //std::cout << "Decoded frame: " << w << "x" << h << std::endl;
                auto end = now();
                duration dur = end - start;
            //    std::cout << "Long " << dur.count() << std::endl;
            //    std::cout << "Should " << (1.0/FPS * 1000) << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds( (int)(1.0/FPS * 1000) - (int)dur.count()) );
                if (decoded_index % 20 == 0) {
                    std::cout << "Decoded, Converted, Processed, Delay : " << decoded_index << ";" << converted_index << ";" << processed_index << ";"<<converted_index - processed_index <<std::endl;
                }
            //    std::cout << "Decoded : " << decoded_index << " ; Processed : " << processed_index << std::endl;
                // MppBuffer buffer = mpp_frame_get_buffer(frame);
                // std::cout << "fd = " << mpp_buffer_get_fd(buffer) << std::endl;
                queue_raw.push(frame);  
                // int rgb_fd = nv12_to_rgb_with_rga(frame);
                // queue.push(rgb_fd);
                // started = 1;
                start = now();
                if (decoded_index == FRAMES) {
                    br = 1;
                }

                

            }

            if (mpp_frame_get_eos(frame)) {
                std::cout << "Got EOS frame" << std::endl;
                mpp_frame_deinit(&frame);
                eos = true;
                break;
            }

           // mpp_frame_deinit(&frame);
        }
        if (br == 1) {
            break;
        }
    }

    // Финальный дренаж (на случай, если ещё есть кадры)
    // while (1) {
    //     MppFrame frame = nullptr;
    //     ret = mpi->decode_get_frame(ctx, &frame);
    //     if (ret != MPP_OK || !frame)
    //         break;

    //     if (!mpp_frame_get_errinfo(frame)) {
    //         int w = mpp_frame_get_width(frame);
    //         int h = mpp_frame_get_height(frame);
    //         std::cout << "Drain frame: " << w << "x" << h << std::endl;
    //     }
    //     mpp_frame_deinit(&frame);
    // }
   // stop = 1;
    if (REALTIME == 0) {
        inf = std::thread(inference_thread);
    }
    conv.join();
    inf.join();

    auto gen_end= now();
    int size = times.size();
    double sum = 0;
    for (int i = 0; i < size; i++) {

        sum += times[i];

    }
    
    std::chrono::duration<double, std::milli> duration = gen_end - gen_start;
    std::cout << "Average Inference FPS = " << size / (sum / 1000) << std::endl;
    std::cout << "Average Full FPS = " << processed_index / (duration.count() / 1000) << std::endl;
    free(buf);
    fclose(fp);
    mpp_destroy(ctx);

    if (old_crtc) {
        drmModeSetCrtc(drm_fd, old_crtc->crtc_id, old_crtc->buffer_id,
                       old_crtc->x, old_crtc->y,
                       &conn_id, 1, &old_crtc->mode);
        drmModeFreeCrtc(old_crtc);
    }

    munmap(map, size);
    drmModeRmFB(drm_fd, fb_id);

    struct drm_mode_destroy_dumb destroy = {0};
    destroy.handle = handle;
    drmIoctl(drm_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy);

    drmModeFreeConnector(conn);
    drmModeFreeResources(res);
    close(drm_fd);

    return 0;
}
