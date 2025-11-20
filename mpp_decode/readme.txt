Сборка

mkdir build
cd build
cmake ..
make

Использование

./main ../Tennis1080p.h264 out.nv12

Для просмотра результата

ffplay -f rawvideo -pixel_format nv12 -video_size 1920x1080 out.nv12
