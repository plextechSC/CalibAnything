cmake_minimum_required(VERSION 3.18)  # Increased for CUDA support
project(LidarToCamera LANGUAGES CXX)

# Build configuration
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pthread")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall")

# CUDA support (optional)
option(USE_CUDA "Enable CUDA GPU acceleration" ON)

find_package(jsoncpp REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
link_directories(${OpenCV_LIBRARY_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# CUDA configuration
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit)

    if(CUDAToolkit_FOUND)
        set(CMAKE_CUDA_STANDARD 14)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        # Support modern NVIDIA GPUs (Pascal, Volta, Turing, Ampere, Ada)
        set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86 89)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -pthread")
        set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
        set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")

        add_definitions(-DUSE_CUDA)
        message(STATUS "CUDA support enabled")
        message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
        message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    else()
        set(USE_CUDA OFF)
        message(WARNING "CUDA Toolkit not found. Building CPU-only version.")
        message(WARNING "To enable GPU acceleration, install CUDA Toolkit and reconfigure.")
    endif()
else()
    message(STATUS "CUDA support disabled (USE_CUDA=OFF)")
endif()

include_directories(${EIGEN_ROOT})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PCL_INCLUDE_DIRS})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Collect source files
file(GLOB_RECURSE CPP_SOURCES src/*.cpp)

# Add CUDA sources if enabled
if(USE_CUDA AND CUDAToolkit_FOUND)
    file(GLOB_RECURSE CUDA_SOURCES src/*.cu)
    add_library(${PROJECT_NAME} STATIC ${CPP_SOURCES} ${CUDA_SOURCES})
    target_link_libraries(${PROJECT_NAME} jsoncpp_lib ${OpenCV_LIBS} CUDA::cudart)
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    add_library(${PROJECT_NAME} STATIC ${CPP_SOURCES})
    target_link_libraries(${PROJECT_NAME} jsoncpp_lib ${OpenCV_LIBS})
endif()

add_executable(run_lidar2camera src/run_lidar2camera.cpp)
target_link_libraries(run_lidar2camera ${PROJECT_NAME})
target_link_libraries(run_lidar2camera ${PCL_LIBRARIES})