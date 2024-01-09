#include "fluid_sim.cuh"

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/common.hpp>
DISABLE_WARNINGS_POP()

#include <device/utils.cuh>


template<typename T>
__global__ void add_sources(T* densities, T* sources, float time_step, unsigned int num_cells) {
    unsigned int tidX   = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tidY   = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int offset = tidX + tidY * blockDim.x * gridDim.x;
    if (offset < num_cells) { densities[offset] += time_step * sources[offset]; }
}

template<typename T>
__global__ void diffuse(T* old_field, T* new_field, uint2 field_extents, unsigned int num_cells,
                        BoundaryStrategy bs, float time_step, float diffusion_rate, unsigned int sim_steps) {
    // Shared memory for storing old and new field data locally for fast access
    // Stores old field and new field data (offset needed to access latter)
    T *gridData                 = shared_memory_proxy<T>();
    unsigned int newFieldOffset = (blockDim.x + 2U) * (blockDim.y + 2U);

    // Global thread indexing (current thread and axial neigbhours)
    unsigned int globalIdX          = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int globalIdY          = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int globalOffset       = globalIdX + globalIdY * blockDim.x * gridDim.x;
    unsigned int globalLeftOffset   = (globalIdX - 1U) + globalIdY * blockDim.x * gridDim.x;
    unsigned int globalRightOffset  = (globalIdX + 1U) + globalIdY * blockDim.x * gridDim.x;
    unsigned int globalUpOffset     = globalIdX + (globalIdY - 1U) * blockDim.x * gridDim.x;
    unsigned int globalDownOffset   = globalIdX + (globalIdY + 1U) * blockDim.x * gridDim.x;

    // Thread status flags
    bool validThread                = globalOffset < num_cells;                         // Thread is actually handling a cell within field bounds
    bool handlingInterior           = 0U < globalIdX && globalIdX <= field_extents.x && // Thread is handling interior cell
                                      0U < globalIdY && globalIdY <= field_extents.y;
    
    // Local thread indexing for shared memory (current thread and axial neigbhours)
    // Old field
    unsigned int verticalStride     = blockDim.x + 2U;                                                          // We store blockDim.x + 2 (ghost cells) elements per row
    unsigned int threadOffsetOld    = (threadIdx.x + 1U)         + ((threadIdx.y + 1U)           * verticalStride); // +1 accounts for storage taken up by ghost cells
    unsigned int leftOffsetOld      = ((threadIdx.x - 1U) + 1U)  + ((threadIdx.y + 1U)           * verticalStride);    
    unsigned int rightOffsetOld     = ((threadIdx.x + 1U) + 1U)  + ((threadIdx.y + 1U)           * verticalStride);
    unsigned int upOffsetOld        = (threadIdx.x + 1U)         + (((threadIdx.y - 1U) + 1U)    * verticalStride);
    unsigned int downOffsetOld      = (threadIdx.x + 1U)         + (((threadIdx.y + 1U) + 1U)    * verticalStride);
    // New field
    unsigned int threadOffsetNew    = threadOffsetOld + newFieldOffset;
    unsigned int leftOffsetNew      = leftOffsetOld + newFieldOffset;
    unsigned int rightOffsetNew     = rightOffsetOld + newFieldOffset;
    unsigned int upOffsetNew        = upOffsetOld + newFieldOffset;
    unsigned int downOffsetNew      = downOffsetOld + newFieldOffset;
    
    // Fetch data from global memory and store in shared memory
    if (validThread) {
        CellLocationType2d locationType = determineLocationType();

        // All threads transfer their corresponding cell's data for both fields
        gridData[threadOffsetOld]   = old_field[globalOffset];
        gridData[threadOffsetNew]   = new_field[globalOffset];

        // Corner and edge cells additionally transfer neighbouring cells' data in the new field for Gauss-Seidel iterations
        // if they do not lie on a field boundary
        if (handlingInterior) {
            switch (locationType) {
                // Corners
                case TopLeft:
                    gridData[upOffsetNew]       = new_field[globalUpOffset];
                    gridData[leftOffsetNew]     = new_field[globalLeftOffset];
                    break;
                case TopRight:
                    gridData[upOffsetNew]       = new_field[globalUpOffset];
                    gridData[rightOffsetNew]    = new_field[globalRightOffset];
                    break;
                case BottomLeft:
                    gridData[downOffsetNew]     = new_field[globalDownOffset];
                    gridData[leftOffsetNew]     = new_field[globalLeftOffset];
                    break;
                case BottomRight:
                    gridData[downOffsetNew]     = new_field[globalDownOffset];
                    gridData[rightOffsetNew]    = new_field[globalRightOffset];
                    break;
                // Edges
                case Top:
                    gridData[upOffsetNew]       = new_field[globalUpOffset];
                    break;
                case Bottom:
                    gridData[downOffsetNew]     = new_field[globalDownOffset];
                    break;
                case Left:
                    gridData[leftOffsetNew]     = new_field[globalLeftOffset];
                    break;
                case Right:
                    gridData[rightOffsetNew]    = new_field[globalRightOffset];
                    break;
            }
        }
    }

    // Simulation parameters
    float diffSpeed         = time_step * diffusion_rate * field_extents.x * field_extents.y;
    float relaxationDenom   = 1.0f + (4.0f * diffSpeed);

    for (unsigned int step = 0U; step < sim_steps; step++) {
        // Gauss-Seidel relaxation step for non-boundary cells
        if (handlingInterior) {
            gridData[threadOffsetNew] = (gridData[threadOffsetOld] +
                                         diffSpeed * (gridData[leftOffsetNew] +
                                                      gridData[rightOffsetNew] +
                                                      gridData[upOffsetNew] +
                                                      gridData[downOffsetNew])) / relaxationDenom; }

        // Ensure simulation step is fully carried out
        __syncthreads();
        
        // Handle boundary cells
        if (validThread && !handlingInterior) {
            handle_boundary(gridData, field_extents, bs,
                            globalIdX, globalIdY,
                            threadOffsetNew, leftOffsetNew, rightOffsetNew, upOffsetNew, downOffsetNew);
        }
    }

    // Write new field value to global memory
    if (validThread) { new_field[globalOffset] = gridData[threadOffsetNew]; }
}

template<typename FieldT, typename VelocityT>
__global__ void advect(FieldT* old_field, FieldT* new_field, VelocityT* velocity_field,
                       uint2 field_extents, unsigned int num_cells,
                       BoundaryStrategy bs, float time_step) {
    // Global thread indexing (current thread and axial neigbhours)
    unsigned int globalIdX          = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int globalIdY          = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int globalOffset       = globalIdX + globalIdY * blockDim.x * gridDim.x;

    // Thread status flags
    bool validThread                = globalOffset < num_cells;                         // Thread is actually handling a cell within field bounds
    bool handlingInterior           = 0U < globalIdX && globalIdX <= field_extents.x && // Thread is handling interior cell
                                      0U < globalIdY && globalIdY <= field_extents.y;

    // Timestep in both directions (adjust by field size)
    glm::vec2 timeStepAxial(time_step *  field_extents.x, time_step * field_extents.y);
    
    if (handlingInterior) {
        // Trace backwards according to velocity fields to find cell whose density ends up in the cell managed by this thread
        VelocityT velocity          = velocity_field[globalOffset];
        glm::vec2 backTrace(globalIdX - (timeStepAxial.x * velocity.x),
                            globalIdY - (timeStepAxial.y * velocity.y));
        backTrace                   = glm::clamp(backTrace,
                                                 glm::vec2(0.5f),
                                                 glm::vec2(field_extents.x + 0.5f, field_extents.y + 0.5f));
        // x = i - dt0 * u[IX(i, j)];
        // y = j - dt0 * v[IX(i, j)];
        // if (x < 0.5) x = 0.5;
        // if (x > N + 0.5) x = N + 0.5;
        // if (y < 0.5)
        //     y = 0.5;
        // if (y > N + 0.5)
        //     y = N + 0.5;

        // Coordinates of neighbours of backtraced points (whose coordinates are non-integer)
        glm::uvec2 backTraceRound   = backTrace;
        unsigned int backtraceLeftUpOffset      = backTraceRound.x + backTraceRound.y * blockDim.x * gridDim.x;
        unsigned int backtraceRightUpOffset     = (backTraceRound.x + 1U) + backTraceRound.y * blockDim.x * gridDim.x;
        unsigned int backtraceLeftDownOffset    = backTraceRound.x + (backTraceRound.y + 1U) * blockDim.x * gridDim.x;
        unsigned int backtraceRightDownOffset   = (backTraceRound.x + 1U) + (backTraceRound.y + 1U) * blockDim.x * gridDim.x;
        // j0 = (int)y;
        // i0 = (int)x;
        // i1 = i0 + 1;
        // j1 = j0 + 1;

        // Linear interpolation coefficients
        float s1                      = backTrace.x - backTraceRound.x;
        float s0                      = 1 - s1;
        float t0                      = backTrace.y - backTraceRound.y;
        float t1                      = 1 - t0;
        
        // Bilinear interpolation of cell values
        new_field[globalOffset] =   s0 * (t0 * old_field[backtraceLeftUpOffset] + t1 * old_field[backtraceLeftDownOffset]) +
                                    s1 * (t0 * old_field[backtraceRightUpOffset] + t1 * old_field[backtraceRightDownOffset]);
    }

    // Ensure simulation step is fully carried out
    __syncthreads();
    
    // Handle boundary cells
    if (validThread && !handlingInterior) {
        unsigned int globalLeftOffset   = (globalIdX - 1U) + globalIdY * blockDim.x * gridDim.x;
        unsigned int globalRightOffset  = (globalIdX + 1U) + globalIdY * blockDim.x * gridDim.x;
        unsigned int globalUpOffset     = globalIdX + (globalIdY - 1U) * blockDim.x * gridDim.x;
        unsigned int globalDownOffset   = globalIdX + (globalIdY + 1U) * blockDim.x * gridDim.x;
        handle_boundary(new_field, field_extents, bs,
                        globalIdX, globalIdY,
                        globalOffset, globalLeftOffset, globalRightOffset, globalUpOffset, globalDownOffset);
    }
}

template<typename T>
__device__ void handle_boundary(T* field, uint2 field_extents, BoundaryStrategy bs,
                                unsigned int tidX, unsigned int tidY, unsigned int offset,
                                unsigned int leftIdx, unsigned int rightIdx, unsigned int upIdx, unsigned int downIdx) {
    // Corners are average of ghost cell neighbours
    if (tidX == 0U && tidY == 0U)                                              { field[offset] = (0.5f * field[rightIdx]) + (0.5f * field[downIdx]); }  // Top-left    
    else if (tidX == (field_extents.x + 1U) && tidY == 0U)                     { field[offset] = (0.5f * field[leftIdx]) + (0.5f * field[downIdx]); }   // Top-right
    else if (tidX == 0U && tidY == (field_extents.y + 1U))                     { field[offset] = (0.5f * field[rightIdx]) + (0.5f * field[upIdx]); }    // Bottom-left
    else if (tidX == (field_extents.x + 1U) && tidY == (field_extents.y + 1U)) { field[offset] = (0.5f * field[leftIdx]) + (0.5f * field[upIdx]); }     // Bottom-right
    // Edge values depend on chosen strategy
    else if (tidX == 0U)                     { field[offset] = bs == Reverse    ? -field[rightIdx]  : field[rightIdx]; }    // Left edge
    else if (tidX == (field_extents.x + 1U)) { field[offset] = bs == Reverse    ? -field[leftIdx]   : field[leftIdx]; }     // Right edge
    else if (tidY == 0U)                     { field[offset] = bs == Reverse    ? -field[downIdx]   : field[downIdx]; }     // Top edge
    else if (tidY == (field_extents.y + 1U)) { field[offset] = bs == Reverse    ? -field[upIdx]     : field[upIdx]; }       // Bottom edge
}
