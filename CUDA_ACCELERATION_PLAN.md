# CUDA Acceleration Plan for CalibAnything

## Executive Summary

This plan outlines a strategy to add CUDA GPU acceleration to CalibAnything for lidar-camera calibration. The main bottleneck is the `CalScore()` function called thousands of times during optimization. GPU acceleration of this function alone can provide **50-100x speedup**.

## Performance Analysis

### Computational Bottlenecks (by impact)

#### 1. CalScore() - HIGHEST PRIORITY ⭐⭐⭐
**Current behavior:**
- Called 1000s of times during random search optimization
- For each call, iterates over ALL point cloud points (10k-100k points)
- Each point: matrix multiplication + projection + mask lookup + accumulation
- Matrix operations for consistency score calculation

**Why it's slow:**
- Sequential processing of large point clouds
- Repeated matrix multiplications
- No data reuse between iterations

**GPU opportunity:**
- Massive parallelism (one thread per point)
- Shared memory for accumulation
- Keep data on GPU across iterations
- **Expected speedup: 50-100x**

#### 2. ProjectOnImage() - HIGH PRIORITY ⭐⭐⭐
**Current behavior:**
- 4x4 matrix × 4x1 vector multiplication
- Perspective projection
- Called for EVERY point in EVERY CalScore call

**GPU opportunity:**
- Embarrassingly parallel operation
- Coalesced memory access patterns
- **Expected speedup: 20-50x**

#### 3. ProcessPointcloud() - MEDIUM PRIORITY ⭐⭐
**Current behavior:**
- Point filtering based on projection bounds
- Octree-based voxel downsampling
- Normal estimation (KNN search + PCA)
- Plane segmentation (RANSAC)
- Euclidean clustering

**GPU opportunity:**
- Parallel point filtering
- GPU voxel downsampling
- cuPCL for normal estimation
- Keep CPU for complex algorithms (RANSAC, clustering)
- **Expected speedup: 5-10x**

#### 4. RandomSearch() - LOWER PRIORITY ⭐
**Current behavior:**
- Already uses multi-threading
- Generates random transformation samples
- Evaluates each with CalScore

**GPU opportunity:**
- Generate samples on GPU (cuRAND)
- Evaluate multiple candidates in parallel
- GPU reduction to find best score
- **Expected speedup: 2-5x** (if CalScore is GPU-accelerated)

## Implementation Strategy

### Phase 1: Core GPU Acceleration (CalScore) 🎯

**Goal:** Accelerate the main bottleneck for maximum impact with minimal code changes

**Components to implement:**

1. **CUDA kernel: `ProjectPointsKernel`**
   ```cuda
   __global__ void ProjectPointsKernel(
       const float4* points,        // Point cloud (x,y,z,w=1)
       const float* normals,        // Normal vectors (nx,ny,nz)
       const float* intensities,    // Intensity values
       const int* segments,         // Segment IDs
       const float* transform,      // 4x4 transformation matrix
       const float* intrinsic,      // 3x3 or 3x4 intrinsic matrix
       int num_points,
       int img_width, int img_height,
       // Output arrays
       int* projected_x,            // Projected pixel coordinates
       int* projected_y,
       bool* valid                  // Projection validity flags
   )
   ```

2. **CUDA kernel: `AccumulateMaskStatsKernel`**
   ```cuda
   __global__ void AccumulateMaskStatsKernel(
       const int* projected_x,
       const int* projected_y,
       const bool* valid,
       const float* normals,
       const float* intensities,
       const int* segments,
       const uchar4* masks,         // 4-channel mask image
       int num_points,
       int img_width, int img_height,
       int num_masks,
       // Output: per-mask accumulators (use atomics)
       float* mask_normals,         // Accumulated normals per mask
       float* mask_intensities,     // Accumulated intensities per mask
       int* mask_segments,          // Segment histogram per mask
       int* mask_point_counts       // Point count per mask
   )
   ```

3. **CUDA kernel: `ComputeConsistencyScoresKernel`**
   ```cuda
   __global__ void ComputeConsistencyScoresKernel(
       const float* mask_normals,
       const float* mask_intensities,
       const int* mask_segments,
       const int* mask_point_counts,
       const int* mask_pixel_counts,
       int num_masks,
       float point_per_pixel,
       // Output scores
       float* normal_scores,
       float* intensity_scores,
       float* segment_scores,
       bool* mask_valid
   )
   ```

4. **Host interface: `CalScoreGPU` class**
   ```cpp
   class CalScoreGPU {
   public:
       void UploadData(
           const std::vector<pcl::PointCloud<PointXYZINS>::Ptr>& pcs,
           const std::vector<cv::Mat>& masks,
           const Eigen::MatrixXf& intrinsic
       );

       double CalScore(
           const Eigen::Matrix4f& transform,
           int file_index
       );

       void FreeGPUMemory();

   private:
       // Device memory pointers
       float4* d_points_;
       float* d_normals_;
       float* d_intensities_;
       int* d_segments_;
       uchar4* d_masks_;
       float* d_intrinsic_;

       // Workspace memory
       int* d_projected_x_;
       int* d_projected_y_;
       bool* d_valid_;
       float* d_mask_accumulators_;

       // Pinned host memory for transfers
       float* h_transform_pinned_;
   };
   ```

**Integration points:**
- Modify `Calibrator::CalScore()` to use `CalScoreGPU::CalScore()`
- Upload data in `Calibrator` constructor
- Free GPU memory in destructor
- Keep CPU implementation as fallback

**Estimated effort:** 3-5 days
**Expected speedup:** 50-100x on CalScore, 10-50x overall calibration time

### Phase 2: Point Cloud Processing Acceleration

**Goal:** Accelerate preprocessing for additional speedup

**Components to implement:**

1. **CUDA kernel: `FilterPointsKernel`**
   - Parallel point filtering based on projection bounds
   - Compact valid points using stream compaction

2. **CUDA kernel: `VoxelDownsampleKernel`**
   - Hash-based voxel grid
   - Parallel centroid computation per voxel

3. **Integration with cuPCL (optional)**
   - Use `pcl::gpu::NormalEstimation` for GPU normal computation
   - Requires PCL compiled with CUDA support

**Estimated effort:** 2-3 days
**Expected additional speedup:** 2-5x preprocessing time

### Phase 3: Random Search Optimization

**Goal:** Fully GPU-accelerated optimization pipeline

**Components to implement:**

1. **CUDA random number generation**
   - Use cuRAND to generate transformation samples on GPU

2. **Batch evaluation**
   - Evaluate multiple candidates in parallel
   - GPU reduction to find best score

3. **Stream-based processing**
   - Overlap computation and data transfer
   - Multiple CUDA streams for concurrency

**Estimated effort:** 2-3 days
**Expected additional speedup:** 1.5-2x search time

## Technical Design Decisions

### 1. Memory Management Strategy

**Approach:** Minimize host-device transfers
- Upload point clouds, masks, intrinsic once at start
- Keep on GPU throughout optimization
- Only transfer transformation matrices (16 floats) per iteration
- Use pinned memory for host-device transfers

**Memory layout:**
- Structure of Arrays (SoA) for coalesced access
- Example: separate arrays for x, y, z vs. array of structs

### 2. Accumulation Strategy for Mask Statistics

**Challenge:** Multiple threads may write to same mask accumulator (race condition)

**Options:**
- **Option A: Atomic operations** (Recommended)
  - Use `atomicAdd` for float accumulators
  - Slower but simpler implementation

- **Option B: Parallel reduction**
  - Each block accumulates to shared memory
  - Reduce across blocks in second pass
  - Faster but more complex

- **Option C: Warp-level primitives**
  - Use `__shfl_down_sync` for warp reduction
  - Fastest but requires careful implementation

**Recommendation:** Start with Option A (atomics), optimize with Option B if needed

### 3. Build System Integration

**CMake configuration:**
```cmake
option(USE_CUDA "Build with CUDA support" ON)

if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)

    add_definitions(-DUSE_CUDA)

    file(GLOB_RECURSE CUDA_SOURCES src/*.cu)
    set_source_files_properties(${CUDA_SOURCES} PROPERTIES LANGUAGE CUDA)

    target_link_libraries(${PROJECT_NAME} CUDA::cudart)
endif()
```

**Conditional compilation:**
```cpp
#ifdef USE_CUDA
    double CalScore(Eigen::Matrix4f T) {
        return cal_score_gpu_.CalScore(T, current_file_idx_);
    }
#else
    double CalScore(Eigen::Matrix4f T) {
        // Existing CPU implementation
    }
#endif
```

### 4. Error Handling

**Strategy:**
- Check CUDA errors after every kernel launch and API call
- Provide informative error messages
- Graceful fallback to CPU if GPU unavailable or memory insufficient

**Helper macro:**
```cpp
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            std::cerr << "Falling back to CPU implementation" << std::endl; \
            use_gpu_ = false; \
        } \
    } while(0)
```

### 5. Testing Strategy

**Validation:**
- Compare GPU results with CPU implementation (numerical accuracy)
- Threshold: < 1e-5 difference in scores
- Test on multiple datasets

**Performance benchmarking:**
- Measure CPU vs GPU time for CalScore
- Profile with Nsight Systems
- Report speedup factors

**Correctness checks:**
- Verify final calibration accuracy matches CPU version
- Check for memory leaks with cuda-memcheck
- Validate on different GPU architectures

## File Structure

```
CalibAnything/
├── CMakeLists.txt                   # Modified: add CUDA support
├── include/
│   ├── calibration.hpp              # Modified: add GPU class members
│   ├── calibration_gpu.cuh          # NEW: GPU declarations
│   └── cuda_utils.cuh               # NEW: CUDA helper functions
├── src/
│   ├── calibration.cpp              # Modified: integrate GPU calls
│   ├── calibration_gpu.cu           # NEW: GPU implementations
│   ├── kernels/
│   │   ├── projection.cu            # NEW: projection kernels
│   │   ├── accumulation.cu          # NEW: accumulation kernels
│   │   └── score.cu                 # NEW: scoring kernels
│   └── run_lidar2camera.cpp         # Modified: add GPU info output
└── README_CUDA.md                   # NEW: GPU usage documentation
```

## Implementation Checklist

### Phase 1: Core GPU Acceleration
- [ ] Update CMakeLists.txt with CUDA support
- [ ] Create cuda_utils.cuh with error checking macros
- [ ] Implement ProjectPointsKernel
- [ ] Implement AccumulateMaskStatsKernel
- [ ] Implement ComputeConsistencyScoresKernel
- [ ] Create CalScoreGPU class with memory management
- [ ] Integrate CalScoreGPU into Calibrator class
- [ ] Add CPU fallback logic
- [ ] Test numerical accuracy vs CPU version
- [ ] Benchmark performance
- [ ] Handle edge cases (empty masks, invalid projections)

### Phase 2: Point Cloud Processing
- [ ] Implement FilterPointsKernel
- [ ] Implement VoxelDownsampleKernel
- [ ] Integrate into ProcessPointcloud
- [ ] Test and benchmark

### Phase 3: Random Search Optimization
- [ ] Implement GPU random sample generation
- [ ] Implement batch evaluation
- [ ] Add CUDA streams for concurrency
- [ ] Test and benchmark

### Documentation & Testing
- [ ] Write README_CUDA.md with usage instructions
- [ ] Add GPU memory requirements to documentation
- [ ] Create performance comparison table
- [ ] Test on different GPU architectures (Pascal, Volta, Ampere)
- [ ] Profile with Nsight Systems
- [ ] Add GPU info output (device name, memory, compute capability)

## Performance Expectations

### Conservative Estimates (Phase 1 only)

| Component | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| CalScore (single call) | 50 ms | 0.5-1 ms | 50-100x |
| Random search (1000 iterations) | 50 s | 0.5-1 s | 50-100x |
| Full calibration | 60 s | 5-10 s | 6-12x |

### With All Phases Complete

| Component | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| Point cloud processing | 10 s | 1-2 s | 5-10x |
| Calibration optimization | 50 s | 0.5-1 s | 50-100x |
| Full pipeline | 60 s | 2-3 s | 20-30x |

**Note:** Actual speedup depends on:
- GPU model (RTX 3090 vs V100 vs A100)
- Point cloud size (10k vs 100k points)
- Number of masks
- CPU baseline (single-threaded vs multi-threaded)

## Risks & Mitigation

### Risk 1: Memory Constraints
**Issue:** Large point clouds may exceed GPU memory
**Mitigation:**
- Batch processing for very large clouds
- Use CUDA unified memory as fallback
- Allow user to configure max GPU memory usage

### Risk 2: Atomic Contention
**Issue:** Many threads updating same mask counters
**Mitigation:**
- Use warp-level primitives for reduction
- Implement hierarchical accumulation
- Profile and optimize hot spots

### Risk 3: Precision Issues
**Issue:** Float vs double precision differences
**Mitigation:**
- Use double precision for accumulation
- Validate against CPU reference
- Configurable precision tolerance

### Risk 4: Platform Compatibility
**Issue:** Code must work on systems without CUDA
**Mitigation:**
- Make CUDA optional at compile time
- CPU fallback always available
- Clear error messages for GPU issues

## Questions to Resolve Before Implementation

1. **GPU Memory Budget:** What's the target GPU? (affects max point cloud size)

2. **Precision Requirements:** Is float32 sufficient or need float64? (affects performance)

3. **Build Requirements:** Should CUDA be optional or required? (affects usability)

4. **PCL Version:** Is PCL compiled with CUDA support? (affects Phase 2 options)

5. **Testing Data:** Do you have test datasets with known ground truth? (for validation)

6. **Performance Goals:** What speedup target is needed? (affects scope)

## Next Steps

1. **Clarify requirements** with user (answer questions above)
2. **Set up development environment** with CUDA toolkit
3. **Implement Phase 1** (core CalScore acceleration)
4. **Validate and benchmark** Phase 1
5. **Proceed to Phase 2/3** if needed
6. **Document and deliver** final implementation

---

**Estimated Total Effort:**
- Phase 1: 3-5 days (highest ROI)
- Phase 2: 2-3 days (optional, incremental benefit)
- Phase 3: 2-3 days (optional, incremental benefit)
- Testing & docs: 1-2 days

**Total: 8-13 days for full implementation**
