cmake_minimum_required(VERSION 3.25)

file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" DNP_VERSION)
string(STRIP "${DNP_VERSION}" DNP_VERSION)

project(
    definitely_not_pytorch
    VERSION ${DNP_VERSION}
    LANGUAGES CXX
)

message(STATUS "Definitely Not PyTorch version: ${PROJECT_VERSION}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(
    DNP_ENABLE_CLANG_FORMAT
    "Format DNP sources automatically during builds"
    ON
)

if(DNP_ENABLE_CLANG_FORMAT)
    find_program(CLANG_FORMAT_EXECUTABLE
        NAMES clang-format
    )

    if(CLANG_FORMAT_EXECUTABLE)
        file(GLOB_RECURSE DNP_FORMAT_SOURCES
            CONFIGURE_DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/*.tpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/*.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cxx"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cuh"
            "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.tpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.tpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cxx"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cuh"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cu"
            "${CMAKE_CURRENT_SOURCE_DIR}/sandbox/*.h"
            "${CMAKE_CURRENT_SOURCE_DIR}/sandbox/*.hpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/sandbox/*.tpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/sandbox/*.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/sandbox/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/sandbox/*.cc"
            "${CMAKE_CURRENT_SOURCE_DIR}/sandbox/*.cxx"
            "${CMAKE_CURRENT_SOURCE_DIR}/sandbox/*.cuh"
            "${CMAKE_CURRENT_SOURCE_DIR}/sandbox/*.cu"
        )

        add_custom_target(dnp_format
            COMMAND "${CLANG_FORMAT_EXECUTABLE}" -i ${DNP_FORMAT_SOURCES}
            COMMENT "Formatting C, C++, and CUDA sources"
            VERBATIM
        )
    else()
        message(WARNING "clang-format not found - automatic formatting disabled")
    endif()
endif()

add_library(dnp INTERFACE)

if(DNP_ENABLE_CLANG_FORMAT AND CLANG_FORMAT_EXECUTABLE)
    add_dependencies(dnp dnp_format)
endif()

target_include_directories(
    dnp
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(dnp_sandbox
    sandbox/main.cpp
)

target_link_libraries(
    dnp_sandbox
    PRIVATE
        dnp
)

add_executable(dnp_tests
    tests/main.cpp
    tests/tensor.cpp
    tests/activation.cpp
    tests/neural_network.cpp
    tests/training.cpp
    tests/serialization.cpp
)

target_link_libraries(
    dnp_tests
    PRIVATE
        dnp
)

enable_testing()

add_test(
    NAME dnp_tests
    COMMAND dnp_tests
)
