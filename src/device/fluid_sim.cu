#include "fluid_sim.cuh"

#include <device/utils.cuh>

#include <cstdio>

__global__ void add_sources(glm::vec3* densities, glm::vec3* sources, float time_step, unsigned int num_cells) {
    unsigned int tidX   = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tidY   = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int offset = tidX + tidY * blockDim.x * gridDim.x;
    if (offset < num_cells) { densities[offset] += time_step * sources[offset]; }
}

__global__ void diffuse(glm::vec3* old_field, glm::vec3* new_field, uint2 field_extents, unsigned int num_cells,
                        float time_step, float diffusion_rate, unsigned int sim_steps) {
    // Shared memory for storing old and new field data locally for fast access
    // Stores old field and new field data (offset needed to access latter)
    extern __shared__ glm::vec3 gridData[];
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
        // Simulation step
        if (handlingInterior) { // Gauss-Seidel relaxation step
            gridData[threadOffsetNew] = (gridData[threadOffsetOld] +
                                         diffSpeed * (gridData[leftOffsetNew] +
                                                      gridData[rightOffsetNew] +
                                                      gridData[upOffsetNew] +
                                                      gridData[downOffsetNew])) / relaxationDenom;
        } else if (validThread) {
            // TODO: Reimplement boundary handling with shared memory
        }

        // Ensure simulation step is fully carried out
        __syncthreads();
    }

    // Write new field value to global memory
    // TODO: Replace with actual newly computed value instead of plonking old value
    if (validThread) { new_field[globalOffset] = gridData[threadOffsetNew]; }
}

__device__ void handle_boundary_global(glm::vec3* field, uint2 field_extents, BoundaryStrategy bs,
                                       unsigned int tidX, unsigned int tidY, unsigned int offset,
                                       unsigned int leftIdx, unsigned int rightIdx, unsigned int upIdx, unsigned int downIdx) {
    // Corners are average of ghost cell neighbours
    if (tidX == 0U && tidY == 0U)                                              { field[offset] = (0.5f * field[rightIdx]) + (0.5f * field[downIdx]); }  // Top-left    
    else if (tidX == (field_extents.x + 1U) && tidY == 0U)                     { field[offset] = (0.5f * field[leftIdx]) + (0.5f * field[downIdx]); }   // Top-right
    else if (tidX == 0U && tidY == (field_extents.y + 1U))                     { field[offset] = (0.5f * field[rightIdx]) + (0.5f * field[upIdx]); }    // Bottom-left
    else if (tidX == (field_extents.x + 1U) && tidY == (field_extents.y + 1U)) { field[offset] = (0.5f * field[leftIdx]) + (0.5f * field[upIdx]); }     // Bottom-right
    // Edge values depend on chosen strategy
    else if (tidX == 0U)                     { field[offset] = bs == ReverseHorizontal  ? -field[rightIdx]  : field[rightIdx]; }    // Left edge
    else if (tidX == (field_extents.x + 1U)) { field[offset] = bs == ReverseHorizontal  ? -field[leftIdx]   : field[leftIdx]; }     // Right edge
    else if (tidY == 0U)                     { field[offset] = bs == ReverseVertical    ? -field[downIdx]   : field[downIdx]; }     // Top edge
    else if (tidY == (field_extents.y + 1U)) { field[offset] = bs == ReverseVertical    ? -field[upIdx]     : field[upIdx]; }       // Bottom edge
}
