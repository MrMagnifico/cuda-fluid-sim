#include "fluid_sim.cuh"

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/common.hpp>
DISABLE_WARNINGS_POP()

#include <device/utils.cuh>


template<typename T>
__global__ void add_sources(T* densities, T* sources, uint2 field_extents, SimulationParams sim_params) {
    GlobalIndexing global   = generate_global_indices(field_extents);
    StatusFlags statusFlags = generate_status_flags(global.idX, global.idY, field_extents);
    if (statusFlags.validThread) { densities[global.offset] += sim_params.timeStep * sources[global.offset]; }
}

template<typename T>
__global__ void diffuse(T* old_field, T* new_field, uint2 field_extents, BoundaryStrategy bs, SimulationParams sim_params) {
    // Shared memory for storing old and new field data locally for fast access
    // Stores old field and new field data (offset needed to access latter)
    T *gridData                 = shared_memory_proxy<T>();
    unsigned int newFieldOffset = (blockDim.x + 2U) * (blockDim.y + 2U);

    GlobalIndexing global   = generate_global_indices(field_extents);
    StatusFlags statusFlags = generate_status_flags(global.idX, global.idY, field_extents);
    
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
    if (statusFlags.validThread) {
        CellLocationType2d locationType = determineLocationType();

        // All threads transfer their corresponding cell's data for both fields
        gridData[threadOffsetOld]   = old_field[global.offset];
        gridData[threadOffsetNew]   = new_field[global.offset];

        // Corner and edge cells additionally transfer neighbouring cells' data in the new field for Gauss-Seidel iterations
        // if they do not lie on a field boundary
        if (statusFlags.handlingInterior) {
            switch (locationType) {
                // Corners
                case TopLeft:
                    gridData[upOffsetNew]       = new_field[global.upOffset];
                    gridData[leftOffsetNew]     = new_field[global.leftOffset];
                    break;
                case TopRight:
                    gridData[upOffsetNew]       = new_field[global.upOffset];
                    gridData[rightOffsetNew]    = new_field[global.rightOffset];
                    break;
                case BottomLeft:
                    gridData[downOffsetNew]     = new_field[global.downOffset];
                    gridData[leftOffsetNew]     = new_field[global.leftOffset];
                    break;
                case BottomRight:
                    gridData[downOffsetNew]     = new_field[global.downOffset];
                    gridData[rightOffsetNew]    = new_field[global.rightOffset];
                    break;
                // Edges
                case Top:
                    gridData[upOffsetNew]       = new_field[global.upOffset];
                    break;
                case Bottom:
                    gridData[downOffsetNew]     = new_field[global.downOffset];
                    break;
                case Left:
                    gridData[leftOffsetNew]     = new_field[global.leftOffset];
                    break;
                case Right:
                    gridData[rightOffsetNew]    = new_field[global.rightOffset];
                    break;
            }
        }
    }

    // Simulation parameters
    float diffSpeed         = sim_params.timeStep * sim_params.diffusionRate * field_extents.x * field_extents.y;
    float relaxationDenom   = 1.0f + (4.0f * diffSpeed);

    for (unsigned int step = 0U; step < sim_params.diffusionSimSteps; step++) {
        // Gauss-Seidel relaxation step for non-boundary cells
        if (statusFlags.handlingInterior) {
            gridData[threadOffsetNew] = (gridData[threadOffsetOld] +
                                         diffSpeed * (gridData[leftOffsetNew] +
                                                      gridData[rightOffsetNew] +
                                                      gridData[upOffsetNew] +
                                                      gridData[downOffsetNew])) / relaxationDenom; }

        // Ensure simulation step is fully carried out
        __syncthreads();
        
        // Handle boundary cells
        if (statusFlags.validThread && !statusFlags.handlingInterior) {
            handle_boundary(gridData, field_extents, bs,
                            global.idX, global.idY,
                            threadOffsetNew, leftOffsetNew, rightOffsetNew, upOffsetNew, downOffsetNew);
        }
    }

    // Write new field value to global memory
    if (statusFlags.validThread) { new_field[global.offset] = gridData[threadOffsetNew]; }
}

template<typename FieldT, typename VelocityT>
__global__ void advect(FieldT* old_field, FieldT* new_field, VelocityT* velocity_field, uint2 field_extents,
                       BoundaryStrategy bs, SimulationParams sim_params) {
    GlobalIndexing global   = generate_global_indices(field_extents);
    StatusFlags statusFlags = generate_status_flags(global.idX, global.idY, field_extents);

    // Timestep in both directions (adjust by field size)
    glm::vec2 timeStepAxial(sim_params.advectionMultiplier * sim_params.timeStep * field_extents.x,
                            sim_params.advectionMultiplier * sim_params.timeStep * field_extents.y);
    
    if (statusFlags.handlingInterior) {
        // Trace backwards according to velocity fields to find coordinates whose density ends up in the cell managed by this 
        VelocityT velocity  = velocity_field[global.offset];
        glm::vec2 backTrace = glm::vec2(global.idX - (timeStepAxial.x * velocity.x),
                                        global.idY - (timeStepAxial.y * velocity.y));
        
        // Bilinear interpolation of cell values
        // I tried texture samplers, man. They made my squares fly away, man
        // Go to the commit with tag hardware-interp and hash 2a09909ff5260d843b4bc8ac591d6e9679598852 if you wanna try fixing my mistake(s)
        glm::uvec2 topLeft              = glm::uvec2(backTrace);
        glm::uvec2 bottomRight          = glm::uvec2(backTrace + 1.0f);
        float rightProportion           = backTrace.x - topLeft.x;
        float downProportion            = backTrace.y - topLeft.y;
        unsigned int topLeftOffset      = topLeft.x     +   topLeft.y * global.verticalStride;
        unsigned int topRightOffset     = bottomRight.x +   topLeft.y * global.verticalStride;
        unsigned int bottomLeftOffset   = topLeft.x     +   bottomRight.y * global.verticalStride;
        unsigned int bottomRightOffset  = bottomRight.x +   bottomRight.y * global.verticalStride;
        FieldT topInterpolation         = glm::mix(old_field[topLeftOffset], old_field[topRightOffset], rightProportion);
        FieldT bottomInterpolation      = glm::mix(old_field[bottomLeftOffset], old_field[bottomRightOffset], rightProportion);
        new_field[global.offset]         = glm::mix(topInterpolation, bottomInterpolation, downProportion);
    }

    // Ensure simulation step is fully carried out
    __syncthreads();
    
    // Handle boundary cells
    if (statusFlags.validThread && !statusFlags.handlingInterior) {
        unsigned int globalLeftOffset   = (global.idX - 1U) + global.idY * global.verticalStride;
        unsigned int globalRightOffset  = (global.idX + 1U) + global.idY * global.verticalStride;
        unsigned int globalUpOffset     = global.idX + (global.idY - 1U) * global.verticalStride;
        unsigned int globalDownOffset   = global.idX + (global.idY + 1U) * global.verticalStride;
        handle_boundary(new_field, field_extents, bs,
                        global.idX, global.idY,
                        global.offset, globalLeftOffset, globalRightOffset, globalUpOffset, globalDownOffset);
    }
}

__global__ void project(glm::vec2* velocities, float* gradient_field, float* projection_field) {

}

__device__ GlobalIndexing generate_global_indices(uint2 field_extents) {
    GlobalIndexing globalIndices;
    globalIndices.idX               = threadIdx.x + blockIdx.x * blockDim.x;
    globalIndices.idY               = threadIdx.y + blockIdx.y * blockDim.y;
    globalIndices.verticalStride    = field_extents.x + 2U;
    globalIndices.offset            = globalIndices.idX         + globalIndices.idY * globalIndices.verticalStride;
    globalIndices.leftOffset        = (globalIndices.idX  - 1U) + globalIndices.idY * globalIndices.verticalStride;
    globalIndices.rightOffset       = (globalIndices.idX  + 1U) + globalIndices.idY * globalIndices.verticalStride;
    globalIndices.upOffset          = globalIndices.idX         + (globalIndices.idY - 1U) * globalIndices.verticalStride;
    globalIndices.downOffset        = globalIndices.idX         + (globalIndices.idY + 1U) * globalIndices.verticalStride;
    return globalIndices;
}

__device__ StatusFlags generate_status_flags(unsigned int globalIdX, unsigned int globalIdY, uint2 field_extents) {
    StatusFlags StatusFlags;
    StatusFlags.validThread         = globalIdX <= field_extents.x + 1U && globalIdY <= field_extents.y + 1U;
    StatusFlags.handlingInterior    = 0U < globalIdX && globalIdX <= field_extents.x && 
                                      0U < globalIdY && globalIdY <= field_extents.y;
    return StatusFlags;
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
