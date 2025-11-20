#include "../include/inc/rk_mpi.h"
#include <iostream>
#include <thread>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/dma-heap.h>
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#define PACKET_SIZE 8192

void paste_frame_into_file(MppFrame frame, FILE* file_fd) {


    int w  = mpp_frame_get_width(frame);
    int h  = mpp_frame_get_height(frame);

    int stride = mpp_frame_get_hor_stride(frame);
    int vstride = mpp_frame_get_ver_stride(frame);

    MppBuffer buffer = mpp_frame_get_buffer(frame);
    int fd = mpp_buffer_get_fd(buffer);
    size_t buf_size = mpp_buffer_get_size(buffer);

    uint8_t *ptr = (uint8_t *)mmap(NULL, buf_size, PROT_READ, MAP_SHARED, fd, 0);


    // Y
    uint8_t *src_y = ptr;
    for (int y = 0; y < h; y++) {
        fwrite(src_y + y * stride, 1, w, file_fd);
    }

    // UV
    uint8_t *src_uv = ptr + vstride * stride;
    for (int y = 0; y < h / 2; y++) {
        fwrite(src_uv + y * stride, 1, w, file_fd);
    }

    munmap(ptr, buf_size);

    std::cout << "Saved NV12 frame: " << w << "x" << h << std::endl;

    mpp_frame_deinit(&frame);

}

// Функция для создания группы dma буферов
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
        // создаём внешнюю группу
        MPP_RET ret = mpp_buffer_group_get_external(&ext_group, MPP_BUFFER_TYPE_DMA_HEAP);
        if (ret) {
            std::cout << "mpp_buffer_group_get_external failed: " << ret << std::endl;
            return ret;
        }

        int buffer_count = 20;
        int heap_fd = open("/dev/dma_heap/system", O_RDWR);
        if (heap_fd < 0) {
            perror("open /dev/dma_heap/system");
            return MPP_NOK;
        }

        // Выделяем dma буферы
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
            // Кладем буферы в mpp
            MppBuffer mbuf;
            ret = mpp_buffer_import(&mbuf, &info);
            if (ret) {
                std::cout << "mpp_buffer_import failed: " << ret << std::endl;
                close(heap_fd);
                return ret;
            }
            // Кладем буферы в группу
            ret = mpp_buffer_commit(ext_group, &info);
            if (ret) {
                std::cout << "mpp_buffer_commit failed: " << ret << std::endl;
                close(heap_fd);
                return ret;
            }
        }

        close(heap_fd);

        
        MPP_RET ret2 = mpi->control(ctx, MPP_DEC_SET_EXT_BUF_GROUP, ext_group);
        if (ret2) {
            std::cout << "MPP_DEC_SET_EXT_BUF_GROUP failed: " << ret2 << std::endl;
            return ret2;
        }

        ext_inited = true;
    }


    mpi->control(ctx, MPP_DEC_SET_INFO_CHANGE_READY, NULL);
    return MPP_OK;
}

int main(int argc, char **argv) {

    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " input_filename.h264 output_filename.nv12" << std::endl;
        return 1;

    }
    const char *filename = argv[1];
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("fopen");
        return -1;
    }
    FILE *nv12_out = fopen(argv[2], "wb");
    if (!nv12_out) {
        perror("fopen output.nv12");
        return -1;
    }


    MppCtx ctx   = nullptr;
    MppApi *mpi  = nullptr;
    MPP_RET ret  = MPP_OK;

    
    ret = mpp_create(&ctx, &mpi);
    if (ret) {
        std::cout << "mpp_create failed: " << ret << std::endl;
        return -1;
    }

    
    int param = 1;
    // Данные передаем декодеру не целыми кадрами, а кусками
    ret = mpi->control(ctx, MPP_DEC_SET_PARSER_SPLIT_MODE, &param);
        

    
    ret = mpp_init(ctx, MPP_CTX_DEC, MPP_VIDEO_CodingAVC);
    if (ret) {
        std::cout << "mpp_init failed: " << ret << std::endl;
        return -1;
    }

    
    unsigned char *buf = (unsigned char *)malloc(PACKET_SIZE);

    
    bool eos = false;
    bool ext_group_inited = false;

    // Инициализация группы dma буферов
    MppBufferGroup ext_group = nullptr;
   
    while (!eos) {
        
        int read_size = fread(buf, 1, PACKET_SIZE, fp);
        if (read_size <= 0) {
            // Конец файла
            eos = true;
        }

        
        if (!eos) {
            MppPacket packet = nullptr;

            // Инициализируем пакет
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
                
            } else if (ret == MPP_ERR_BUFFER_FULL) {
                // Очистка переполненной очереди
                while (1) {
                    MppFrame frm = nullptr;
                    MPP_RET r2 = mpi->decode_get_frame(ctx, &frm);
                    if (r2 != MPP_OK || !frm)
                        break;

                    if (mpp_frame_get_info_change(frm)) {
                        // Получаем первый кадр, и создаем группу буфером с соотвутствующими параметрами
                        handle_info_change(ctx, mpi, frm, ext_group, ext_group_inited);
                        mpp_frame_deinit(&frm);
                        continue;
                    }

                    
                    mpp_frame_deinit(&frm);
                }

                usleep(1000);
                // Пытаемся отправить тот же самый пакет еще раз
                goto put_again;
            } else {
                std::cout << "decode_put_packet error: " << ret << std::endl;
                mpp_packet_deinit(&packet);
                break;
            }

            mpp_packet_deinit(&packet);
        } 

        
        // Получение декодированного кадра в формате NV12
        while (1) {
            MppFrame frame = nullptr;
            ret = mpi->decode_get_frame(ctx, &frame);
            if (ret != MPP_OK)
                break;
            if (!frame)
                break;

            if (mpp_frame_get_info_change(frame)) {
                // Это на всякий случай, если первый кадр попадется в этой части кода
                handle_info_change(ctx, mpi, frame, ext_group, ext_group_inited);
                mpp_frame_deinit(&frame);
                continue;
            }

            
            if (!mpp_frame_get_errinfo(frame)) {

                paste_frame_into_file(frame, nv12_out);
               
            }

            if (mpp_frame_get_eos(frame)) {
                std::cout << "Got EOS frame" << std::endl;
                mpp_frame_deinit(&frame);
                eos = true;
                break;
            }

           mpp_frame_deinit(&frame);
        }

    }

    // Выход
    free(buf);
    fclose(fp);
    mpp_destroy(ctx);
}