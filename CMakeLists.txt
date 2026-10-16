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

file(GLOB_RECURSE FLUID_SIMULATOR_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_library(dnp
    ${FLUID_SIMULATOR_SOURCES}
)

target_include_directories(
    dnp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(dnp_test
    tests/test.cpp
)

target_link_libraries(
    dnp_test
    PRIVATE
        dnp
)