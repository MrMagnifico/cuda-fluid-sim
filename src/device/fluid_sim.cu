#include "fluid_sim.cuh"

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/common.hpp>
DISABLE_WARNINGS_POP()

#include <device/utils.cuh>


template<typename T>
__global__ void add_sources(T* densities, T* sources, uint2 field_extents, unsigned int num_cells, SimulationParams sim_params) {
    unsigned int tidX   = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tidY   = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int offset = tidX + tidY * (field_extents.x + 2U);
    if (offset < num_cells) { densities[offset] += sim_params.timeStep * sources[offset]; }
}

template<typename T>
__global__ void diffuse(T* old_field, T* new_field, uint2 field_extents, unsigned int num_cells, BoundaryStrategy bs, SimulationParams sim_params) {
    // Shared memory for storing old and new field data locally for fast access
    // Stores old field and new field data (offset needed to access latter)
    T *gridData                 = shared_memory_proxy<T>();
    unsigned int newFieldOffset = (blockDim.x + 2U) * (blockDim.y + 2U);

    // Global thread indexing (current thread and axial neigbhours)
    unsigned int globalIdX          = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int globalIdY          = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int globalOffset       = globalIdX + globalIdY * (field_extents.x + 2U);
    unsigned int globalLeftOffset   = (globalIdX - 1U) + globalIdY * (field_extents.x + 2U);
    unsigned int globalRightOffset  = (globalIdX + 1U) + globalIdY * (field_extents.x + 2U);
    unsigned int globalUpOffset     = globalIdX + (globalIdY - 1U) * (field_extents.x + 2U);
    unsigned int globalDownOffset   = globalIdX + (globalIdY + 1U) * (field_extents.x + 2U);

    // Thread status flags
    bool validThread                = globalOffset < num_cells;                         // Thread is actually handling a cell within field bounds
    bool handlingInterior           = 0U < globalIdX && globalIdX <= field_extents.x && // Thread is handling interior cell
                                      0U < globalIdY && globalIdY <= field_extents.y;
    
    // Local thread indexing for shared memory (current thread and axial neigbhours)
    // Old field
    unsigned int verticalStride     = blockDim.x + 2U;                                                              // We store blockDim.x + 2 (ghost cells) elements per row
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
    float diffSpeed         = sim_params.timeStep * sim_params.diffusionRate * field_extents.x * field_extents.y;
    float relaxationDenom   = 1.0f + (4.0f * diffSpeed);

    for (unsigned int step = 0U; step < sim_params.diffusionSimSteps; step++) {
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
                       BoundaryStrategy bs, SimulationParams sim_params) {
    // Global thread indexing (current thread and axial neigbhours)
    unsigned int globalIdX          = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int globalIdY          = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int globalOffset       = globalIdX + globalIdY * (field_extents.x + 2U);

    // Thread status flags
    bool validThread                = globalOffset < num_cells;                         // Thread is actually handling a cell within field bounds
    bool handlingInterior           = 0U < globalIdX && globalIdX <= field_extents.x && // Thread is handling interior cell
                                      0U < globalIdY && globalIdY <= field_extents.y;

    // Timestep in both directions (adjust by field size)
    glm::vec2 timeStepAxial(sim_params.advectionMultiplier * sim_params.timeStep * field_extents.x,
                            sim_params.advectionMultiplier * sim_params.timeStep * field_extents.y);
    
    if (handlingInterior) {
        // Trace backwards according to velocity fields to find coordinates whose density ends up in the cell managed by this 
        VelocityT velocity  = velocity_field[globalOffset];
        glm::vec2 backTrace = glm::vec2(globalIdX - (timeStepAxial.x * velocity.x),
                                        globalIdY - (timeStepAxial.y * velocity.y));
        
        // Bilinear interpolation of cell values
        // I tried texture samplers, man. They made my squares fly away, man
        glm::uvec2 topLeft              = glm::uvec2(backTrace);
        glm::uvec2 bottomRight          = glm::uvec2(backTrace + 1.0f);
        float rightProportion           = backTrace.x - topLeft.x;
        float downProportion            = backTrace.y - topLeft.y;
        unsigned int topLeftOffset      = topLeft.x     +   topLeft.y * (field_extents.x + 2U);
        unsigned int topRightOffset     = bottomRight.x +   topLeft.y * (field_extents.x + 2U);
        unsigned int bottomLeftOffset   = topLeft.x     +   bottomRight.y * (field_extents.x + 2U);
        unsigned int bottomRightOffset  = bottomRight.x +   bottomRight.y * (field_extents.x + 2U);
        FieldT topInterpolation         = glm::mix(old_field[topLeftOffset], old_field[topRightOffset], rightProportion);
        FieldT bottomInterpolation      = glm::mix(old_field[bottomLeftOffset], old_field[bottomRightOffset], rightProportion);
        new_field[globalOffset]         = glm::mix(topInterpolation, bottomInterpolation, downProportion);
    }

    // Ensure simulation step is fully carried out
    __syncthreads();
    
    // Handle boundary cells
    if (validThread && !handlingInterior) {
        unsigned int globalLeftOffset   = (globalIdX - 1U) + globalIdY * (field_extents.x + 2U);
        unsigned int globalRightOffset  = (globalIdX + 1U) + globalIdY * (field_extents.x + 2U);
        unsigned int globalUpOffset     = globalIdX + (globalIdY - 1U) * (field_extents.x + 2U);
        unsigned int globalDownOffset   = globalIdX + (globalIdY + 1U) * (field_extents.x + 2U);
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
