/*
 * CUDA Utility Functions and Macros
 * Error checking, memory management, and helper functions for CUDA operations
 */

#pragma once

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <iostream>
#include <sstream>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            std::cerr << "Failed call: " << #call << std::endl; \
            return false; \
        } \
    } while(0)

// CUDA kernel launch error checking
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA kernel launch error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            return false; \
        } \
    } while(0)

// CUDA synchronization with error checking
#define CUDA_SYNC_CHECK() \
    do { \
        cudaError_t err = cudaDeviceSynchronize(); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA synchronization error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            return false; \
        } \
    } while(0)

// Version without return for use in constructors/destructors
#define CUDA_CHECK_NO_RETURN(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            std::cerr << "Failed call: " << #call << std::endl; \
        } \
    } while(0)

namespace cuda_utils {

// Get CUDA device properties and print info
inline bool PrintDeviceInfo(int device_id = 0) {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        std::cerr << "No CUDA-capable devices found" << std::endl;
        return false;
    }

    if (device_id >= device_count) {
        std::cerr << "Device ID " << device_id << " not available. "
                  << "Only " << device_count << " device(s) available." << std::endl;
        return false;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    std::cout << "=== CUDA Device Info ===" << std::endl;
    std::cout << "Device " << device_id << ": " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total global memory: " << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)
              << " GB" << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024.0
              << " KB" << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Multiprocessors: " << prop.multiProcessorCount << std::endl;
    std::cout << "Warp size: " << prop.warpSize << std::endl;
    std::cout << "========================" << std::endl;

    return true;
}

// Check if sufficient GPU memory is available
inline bool CheckAvailableMemory(size_t required_bytes, int device_id = 0) {
    CUDA_CHECK(cudaSetDevice(device_id));

    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));

    double free_gb = free_mem / (1024.0 * 1024.0 * 1024.0);
    double required_gb = required_bytes / (1024.0 * 1024.0 * 1024.0);

    std::cout << "GPU memory - Free: " << free_gb << " GB, Required: "
              << required_gb << " GB" << std::endl;

    if (required_bytes > free_mem) {
        std::cerr << "Insufficient GPU memory!" << std::endl;
        std::cerr << "Required: " << required_gb << " GB" << std::endl;
        std::cerr << "Available: " << free_gb << " GB" << std::endl;
        return false;
    }

    return true;
}

// Calculate optimal grid and block dimensions
inline void CalculateLaunchConfig(int num_elements, int& grid_size, int& block_size,
                                   int max_block_size = 256) {
    block_size = (num_elements < max_block_size) ? num_elements : max_block_size;
    grid_size = (num_elements + block_size - 1) / block_size;
}

// Calculate optimal grid and block dimensions for 2D
inline void CalculateLaunchConfig2D(int width, int height,
                                     dim3& grid, dim3& block,
                                     int block_x = 16, int block_y = 16) {
    block = dim3(block_x, block_y);
    grid = dim3((width + block_x - 1) / block_x, (height + block_y - 1) / block_y);
}

// Allocate pinned (page-locked) host memory for faster transfers
template<typename T>
inline bool AllocatePinnedMemory(T** ptr, size_t num_elements) {
    CUDA_CHECK(cudaMallocHost((void**)ptr, num_elements * sizeof(T)));
    return true;
}

// Free pinned host memory
template<typename T>
inline void FreePinnedMemory(T* ptr) {
    if (ptr != nullptr) {
        CUDA_CHECK_NO_RETURN(cudaFreeHost(ptr));
    }
}

// Allocate device memory
template<typename T>
inline bool AllocateDeviceMemory(T** ptr, size_t num_elements) {
    CUDA_CHECK(cudaMalloc((void**)ptr, num_elements * sizeof(T)));
    return true;
}

// Free device memory
template<typename T>
inline void FreeDeviceMemory(T* ptr) {
    if (ptr != nullptr) {
        CUDA_CHECK_NO_RETURN(cudaFree(ptr));
    }
}

// Copy from host to device
template<typename T>
inline bool CopyHostToDevice(T* d_dest, const T* h_src, size_t num_elements) {
    CUDA_CHECK(cudaMemcpy(d_dest, h_src, num_elements * sizeof(T),
                          cudaMemcpyHostToDevice));
    return true;
}

// Copy from device to host
template<typename T>
inline bool CopyDeviceToHost(T* h_dest, const T* d_src, size_t num_elements) {
    CUDA_CHECK(cudaMemcpy(h_dest, d_src, num_elements * sizeof(T),
                          cudaMemcpyDeviceToHost));
    return true;
}

// Initialize device memory to zero
template<typename T>
inline bool ZeroDeviceMemory(T* d_ptr, size_t num_elements) {
    CUDA_CHECK(cudaMemset(d_ptr, 0, num_elements * sizeof(T)));
    return true;
}

// RAII wrapper for device memory
template<typename T>
class DeviceMemory {
public:
    DeviceMemory() : ptr_(nullptr), size_(0) {}

    explicit DeviceMemory(size_t num_elements) : ptr_(nullptr), size_(0) {
        Allocate(num_elements);
    }

    ~DeviceMemory() {
        Free();
    }

    // Delete copy constructor and assignment
    DeviceMemory(const DeviceMemory&) = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;

    // Move constructor and assignment
    DeviceMemory(DeviceMemory&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_) {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceMemory& operator=(DeviceMemory&& other) noexcept {
        if (this != &other) {
            Free();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    bool Allocate(size_t num_elements) {
        Free();
        size_ = num_elements;
        return AllocateDeviceMemory(&ptr_, num_elements);
    }

    void Free() {
        if (ptr_) {
            FreeDeviceMemory(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    bool Zero() {
        return ZeroDeviceMemory(ptr_, size_);
    }

    bool CopyFromHost(const T* h_src) {
        return CopyHostToDevice(ptr_, h_src, size_);
    }

    bool CopyToHost(T* h_dest) const {
        return CopyDeviceToHost(h_dest, ptr_, size_);
    }

    T* Get() { return ptr_; }
    const T* Get() const { return ptr_; }
    size_t Size() const { return size_; }
    size_t SizeInBytes() const { return size_ * sizeof(T); }

private:
    T* ptr_;
    size_t size_;
};

} // namespace cuda_utils

#endif // USE_CUDA
