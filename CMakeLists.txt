cmake_minimum_required(VERSION 3.10)

project(ragify)

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)

FetchContent_Declare(
        llama_cpp
        GIT_REPOSITORY https://github.com/ggml-org/llama.cpp
        GIT_TAG b6060
)

option(LLAMA_STANDALONE "" OFF)
option(LLAMA_BUILD_TESTS "" OFF)
option(GGML_LLAMAFILE "" OFF)
option(LLAMA_BUILD_COMMON "" ON)

FetchContent_MakeAvailable(llama_cpp)

set_target_properties(
        build_info llama common PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "$<BUILD_INTERFACE:${llama_cpp_SOURCE_DIR}/include;${llama_cpp_SOURCE_DIR}/src;${llama_cpp_SOURCE_DIR}/common;${llama_cpp_SOURCE_DIR}/>"
        CXX_STANDARD 17
)

aux_source_directory(src SRC_LIST)

add_library(ragify ${SRC_LIST})

target_link_libraries(ragify PUBLIC llama common)

add_subdirectory(test)
