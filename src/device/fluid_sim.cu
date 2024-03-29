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
    T *gridData             = shared_memory_proxy<T>();
    GlobalIndexing global   = generate_global_indices(field_extents);
    SharedIndexing shared   = generate_shared_indices();
    StatusFlags statusFlags = generate_status_flags(global.idX, global.idY, field_extents);
    global_to_shared(gridData, old_field, new_field, statusFlags, global, shared);

    // Simulation parameters
    float diffSpeed         = sim_params.timeStep * sim_params.diffusionRate;
    float relaxationDenom   = 1.0f + (4.0f * diffSpeed);

    for (unsigned int step = 0U; step < sim_params.diffusionSimSteps; step++) {
        // Gauss-Seidel relaxation step for non-boundary cells
        if (statusFlags.handlingInterior) {
            gridData[shared.offsetNew] = (gridData[shared.offsetOld] +
                                          diffSpeed * (gridData[shared.leftOffsetNew] +
                                                       gridData[shared.rightOffsetNew] +
                                                       gridData[shared.upOffsetNew] +
                                                       gridData[shared.downOffsetNew])) / relaxationDenom; }
        __syncthreads();
        
        // Handle boundary cells
        if (statusFlags.validThread && !statusFlags.handlingInterior) {
            handle_boundary(gridData, field_extents, bs,
                            global.idX, global.idY,
                            shared.offsetNew, shared.leftOffsetNew, shared.rightOffsetNew, shared.upOffsetNew, shared.downOffsetNew);
        }
        __syncthreads();
    }

    // Write new field value to global memory
    if (statusFlags.validThread) { new_field[global.offset] = gridData[shared.offsetNew]; }
}

template<typename FieldT, typename VelocityT>
__global__ void advect(FieldT* old_field, FieldT* new_field, VelocityT* velocity_field, uint2 field_extents,
                       BoundaryStrategy bs, SimulationParams sim_params) {
    GlobalIndexing global   = generate_global_indices(field_extents);
    StatusFlags statusFlags = generate_status_flags(global.idX, global.idY, field_extents);

    float advectionTimeStep = sim_params.advectionMultiplier * sim_params.timeStep;
    if (statusFlags.handlingInterior) {
        // Trace backwards according to velocity fields to find coordinates whose density ends up in the cell managed by this 
        VelocityT velocity  = velocity_field[global.offset];
        glm::vec2 backTrace = glm::vec2(global.idX - (advectionTimeStep * velocity.x),
                                        global.idY - (advectionTimeStep * velocity.y));
        backTrace           = glm::clamp(backTrace, glm::vec2(0.0f), glm::vec2(field_extents.x, field_extents.y));
        
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
        new_field[global.offset]        = glm::mix(topInterpolation, bottomInterpolation, downProportion);
    }

    // Ensure simulation step is fully carried out
    __syncthreads();
    
    // Handle boundary cells
    if (statusFlags.validThread && !statusFlags.handlingInterior) {
        handle_boundary(new_field, field_extents, bs,
                        global.idX, global.idY,
                        global.offset, global.leftOffset, global.rightOffset, global.upOffset, global.downOffset);
    }
}

__global__ void project(glm::vec2* velocities, uint2 field_extents, SimulationParams sim_params) {
    // Shared memory for storing derivative field and projection field data locally for fast access
    extern __shared__ float derivProjFields[];
    GlobalIndexing global   = generate_global_indices(field_extents);
    SharedIndexing shared   = generate_shared_indices();
    StatusFlags statusFlags = generate_status_flags(global.idX, global.idY, field_extents);
    float hysteresis        = rsqrtf(field_extents.x * field_extents.y);

    // Compute derivative field (finite horizontal and vertical gradients) and zero out projection field
    if (statusFlags.handlingInterior) {
        derivProjFields[shared.offsetOld]  = -0.5f * hysteresis * (velocities[global.rightOffset].x    - velocities[global.leftOffset].x +
                                                                   velocities[global.upOffset].y       - velocities[global.downOffset].y);
        derivProjFields[shared.offsetNew]  = 0.0f;

        // Corner and edge cells additionally zero out neighbouring cells' projection data
        CellLocationType2d locationType = determineLocationType();
        switch (locationType) {
                // Corners
                case TopLeft:
                    derivProjFields[shared.upOffsetNew]      = 0.0f;
                    derivProjFields[shared.leftOffsetNew]    = 0.0f;
                    break;
                case TopRight:
                    derivProjFields[shared.upOffsetNew]      = 0.0f;;
                    derivProjFields[shared.rightOffsetNew]   = 0.0f;;
                    break;
                case BottomLeft:
                    derivProjFields[shared.downOffsetNew]    = 0.0f;;
                    derivProjFields[shared.leftOffsetNew]    = 0.0f;;
                    break;
                case BottomRight:
                    derivProjFields[shared.downOffsetNew]    = 0.0f;;
                    derivProjFields[shared.rightOffsetNew]   = 0.0f;;
                    break;
                // Edges
                case Top:
                    derivProjFields[shared.upOffsetNew]      = 0.0f;;
                    break;
                case Bottom:
                    derivProjFields[shared.downOffsetNew]    = 0.0f;;
                    break;
                case Left:
                    derivProjFields[shared.leftOffsetNew]    = 0.0f;;
                    break;
                case Right:
                    derivProjFields[shared.rightOffsetNew]   = 0.0f;;
                    break;
            }
    }
    __syncthreads();
    if (statusFlags.validThread && !statusFlags.handlingInterior) {
        handle_boundary(derivProjFields, field_extents, Conserve,
                        global.idX, global.idY,
                        shared.offsetOld, shared.leftOffsetOld, shared.rightOffsetOld, shared.upOffsetOld, shared.downOffsetOld);
        handle_boundary(derivProjFields, field_extents, Conserve,
                        global.idX, global.idY,
                        shared.offsetNew, shared.leftOffsetNew, shared.rightOffsetNew, shared.upOffsetNew, shared.downOffsetNew);
    }

    // Compute projection field via Gauss-Seidel iteration
    for (unsigned int step = 0U; step < sim_params.projectionSimSteps; step++) {
        // Gauss-Seidel relaxation step for non-boundary cells
        if (statusFlags.handlingInterior) {
            derivProjFields[shared.offsetNew] = (derivProjFields[shared.offsetOld]                                            +
                                                 derivProjFields[shared.leftOffsetNew]    + derivProjFields[shared.rightOffsetNew]   +
                                                 derivProjFields[shared.upOffsetNew]      + derivProjFields[shared.downOffsetNew]) / 4.0f;
        }
        __syncthreads();
        
        // Handle boundary cells
        if (statusFlags.validThread && !statusFlags.handlingInterior) {
            handle_boundary(derivProjFields, field_extents, Conserve,
                            global.idX, global.idY,
                            shared.offsetNew, shared.leftOffsetNew, shared.rightOffsetNew, shared.upOffsetNew, shared.downOffsetNew);
        }
        __syncthreads();
    }

    // Compute mass-conserved velocity field from projection field
    if (statusFlags.handlingInterior) {
        glm::vec2& cellValue    = velocities[global.offset];
        cellValue.x             -= 0.5f * (derivProjFields[shared.rightOffsetNew] - derivProjFields[shared.leftOffsetNew]) / hysteresis;
        cellValue.y             -= 0.5f * (derivProjFields[shared.upOffsetNew] - derivProjFields[shared.downOffsetNew]) / hysteresis;
    }
    __syncthreads();
    if (statusFlags.validThread && !statusFlags.handlingInterior) {
        handle_boundary(velocities, field_extents, Reverse,
                        global.idX, global.idY,
                        global.offset, global.leftOffset, global.rightOffset, global.upOffset, global.downOffset);
    }
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

__device__ SharedIndexing generate_shared_indices() {
    SharedIndexing shared;
    
    // First field
    shared.verticalStride   = blockDim.x + 2U;                                                                      // We store blockDim.x + 2 (ghost cells) elements per row
    shared.offsetOld        = (threadIdx.x + 1U)        + ((threadIdx.y + 1U)           * shared.verticalStride);   // +1 accounts for storage taken up by ghost cells
    shared.leftOffsetOld    = ((threadIdx.x - 1U) + 1U) + ((threadIdx.y + 1U)           * shared.verticalStride);    
    shared.rightOffsetOld   = ((threadIdx.x + 1U) + 1U) + ((threadIdx.y + 1U)           * shared.verticalStride);
    shared.upOffsetOld      = (threadIdx.x + 1U)        + (((threadIdx.y - 1U) + 1U)    * shared.verticalStride);
    shared.downOffsetOld    = (threadIdx.x + 1U)        + (((threadIdx.y + 1U) + 1U)    * shared.verticalStride);
    
    // Second field
    shared.newFieldOffset   = (blockDim.x + 2U) * (blockDim.y + 2U);
    shared.offsetNew        = shared.offsetOld      + shared.newFieldOffset;
    shared.leftOffsetNew    = shared.leftOffsetOld  + shared.newFieldOffset;
    shared.rightOffsetNew   = shared.rightOffsetOld + shared.newFieldOffset;
    shared.upOffsetNew      = shared.upOffsetOld    + shared.newFieldOffset;
    shared.downOffsetNew    = shared.downOffsetOld  + shared.newFieldOffset;

    return shared;
}

__device__ StatusFlags generate_status_flags(unsigned int globalIdX, unsigned int globalIdY, uint2 field_extents) {
    StatusFlags StatusFlags;
    StatusFlags.validThread         = globalIdX <= field_extents.x + 1U && globalIdY <= field_extents.y + 1U;
    StatusFlags.handlingInterior    = 0U < globalIdX && globalIdX <= field_extents.x && 
                                      0U < globalIdY && globalIdY <= field_extents.y;
    return StatusFlags;
}

template<typename T>
__device__ void global_to_shared(T* shared_mem, T* first_field, T* second_field,
                                 const StatusFlags& status_flags, const GlobalIndexing& global, const SharedIndexing& shared) {
    if (status_flags.validThread) {
        CellLocationType2d locationType = determineLocationType();

        // All threads transfer their corresponding cell's data
        shared_mem[shared.offsetOld]    = first_field[global.offset];
        shared_mem[shared.offsetNew]    = second_field[global.offset];

        // Corner and edge cells additionally transfer neighbouring cells' data if they do not lie on a field boundary
        if (status_flags.handlingInterior) {
            switch (locationType) {
                // Corners
                case TopLeft:
                    shared_mem[shared.upOffsetOld]      = first_field[global.upOffset];
                    shared_mem[shared.leftOffsetOld]    = first_field[global.leftOffset];
                    shared_mem[shared.upOffsetNew]      = second_field[global.upOffset];
                    shared_mem[shared.leftOffsetNew]    = second_field[global.leftOffset];
                    break;
                case TopRight:
                    shared_mem[shared.upOffsetOld]      = first_field[global.upOffset];
                    shared_mem[shared.rightOffsetOld]   = first_field[global.rightOffset];
                    shared_mem[shared.upOffsetNew]      = second_field[global.upOffset];
                    shared_mem[shared.rightOffsetNew]   = second_field[global.rightOffset];
                    break;
                case BottomLeft:
                    shared_mem[shared.downOffsetOld]    = first_field[global.downOffset];
                    shared_mem[shared.leftOffsetOld]    = first_field[global.leftOffset];
                    shared_mem[shared.downOffsetNew]    = second_field[global.downOffset];
                    shared_mem[shared.leftOffsetNew]    = second_field[global.leftOffset];
                    break;
                case BottomRight:
                    shared_mem[shared.downOffsetOld]    = first_field[global.downOffset];
                    shared_mem[shared.rightOffsetOld]   = first_field[global.rightOffset];
                    shared_mem[shared.downOffsetNew]    = second_field[global.downOffset];
                    shared_mem[shared.rightOffsetNew]   = second_field[global.rightOffset];
                    break;
                // Edges
                case Top:
                    shared_mem[shared.upOffsetOld]      = first_field[global.upOffset];
                    shared_mem[shared.upOffsetNew]      = second_field[global.upOffset];
                    break;
                case Bottom:
                    shared_mem[shared.downOffsetOld]    = first_field[global.downOffset];
                    shared_mem[shared.downOffsetNew]    = second_field[global.downOffset];
                    break;
                case Left:
                    shared_mem[shared.leftOffsetOld]    = first_field[global.leftOffset];
                    shared_mem[shared.leftOffsetNew]    = second_field[global.leftOffset];
                    break;
                case Right:
                    shared_mem[shared.rightOffsetOld]   = first_field[global.rightOffset];
                    shared_mem[shared.rightOffsetNew]   = second_field[global.rightOffset];
                    break;
            }
        }
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
    else if (tidX == 0U) {                          // Left edge
        T newVal        = field[rightIdx];
        if (bs == Reverse) { newVal = handle_reverse(newVal, ReverseAxis::X); }
        field[offset]   = newVal;
    } else if (tidX == (field_extents.x + 1U)) {    // Right edge
        T newVal        = field[leftIdx];
        if (bs == Reverse) { newVal = handle_reverse(newVal, ReverseAxis::X); }
        field[offset]   = newVal;
    } else if (tidY == 0U) {                        // Top edge
        T newVal        = field[downIdx];
        if (bs == Reverse) { newVal = handle_reverse(newVal, ReverseAxis::Y); }
        field[offset]   = newVal;
    } else if (tidY == (field_extents.y + 1U)) {    // Bottom edge
        T newVal        = field[upIdx];
        if (bs == Reverse) { newVal = handle_reverse(newVal, ReverseAxis::Y); }
        field[offset]   = newVal;
    }
}

__device__ float handle_reverse(float value, ReverseAxis axis) { return -value; }

__device__ glm::vec2 handle_reverse(glm::vec2 value, ReverseAxis axis) {
    value[axis] = -value[axis];
    return value;
}

__device__ glm::vec3 handle_reverse(glm::vec3 value, ReverseAxis axis) {
    value[axis] = -value[axis];
    return value;
}
