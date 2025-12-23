cmake_minimum_required(VERSION 3.18.1)
project(rockenhancer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(
        rockenhancer
        SHARED
        native-lib.cpp)

find_library(
        log-lib
        log)

target_compile_definitions(rockenhancer PRIVATE STB_IMAGE_IMPLEMENTATION STB_IMAGE_WRITE_IMPLEMENTATION)

target_compile_options(rockenhancer PRIVATE -O2)

target_link_libraries(
        rockenhancer
        ${log-lib})

find_library(log-lib log)
find_library(jnigraphics-lib jnigraphics)

target_link_libraries(rockenhancer
        ${log-lib}
        ${jnigraphics-lib}
)
