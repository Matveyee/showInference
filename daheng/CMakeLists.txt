project(daheng)

include_directories("./include")
include_directories("./include/rga")

find_package(HailoRT REQUIRED)

link_directories("/usr/local/lib")
link_directories("/usr/lib")
link_directories(./libs )

add_executable(main src/main.cpp)

target_link_libraries(main -lgxiapi -lrga HailoRT::libhailort -pthread

)