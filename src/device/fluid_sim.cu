#include "fluid_sim.cuh"

__global__ void add_sources(glm::vec3* densities, glm::vec3* sources, float time_step, unsigned int num_cells) {
    unsigned int tidX   = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tidY   = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int offset = tidX + tidY * blockDim.x * gridDim.x;
    if (offset < num_cells) { densities[offset] += time_step * sources[offset]; }
}

__global__ void diffuse(glm::vec3* old_field, glm::vec3* new_field, uint2 field_extents, unsigned int num_cells,
                        float time_step, float diffusion_rate, unsigned int sim_steps) {
    // Thread indexing
    unsigned int tidX   = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tidY   = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int offset = tidX + tidY * blockDim.x * gridDim.x;

    // Compute offsets for axial neighbours
    unsigned int leftIdx    = (tidX - 1U) + tidY * blockDim.x * gridDim.x;
    unsigned int rightIdx   = (tidX + 1U) + tidY * blockDim.x * gridDim.x;
    unsigned int upIdx      = tidX + (tidY - 1U) * blockDim.x * gridDim.x;
    unsigned int downIdx    = tidX + (tidY + 1U) * blockDim.x * gridDim.x;

    // Simulation parameters
    float diffSpeed         = time_step * diffusion_rate * field_extents.x * field_extents.y;
    float relaxationDenom   = 1.0f + (4.0f * diffSpeed);

    for (unsigned int step = 0U; step < sim_steps; step++) {
        // Simulation step
        if ((0U < tidX && tidX <= field_extents.x) && (0U < tidY && tidY <= field_extents.y)) { // Thread is handling valid non-ghost cell. Gauss-Seidel relaxation step
            new_field[offset] = (old_field[offset] + diffSpeed * (old_field[leftIdx] + old_field[rightIdx] + old_field[upIdx] + old_field[downIdx])) / relaxationDenom;
        } else if (offset < num_cells) { // Thread is handling valid ghost cell
            handle_boundary(new_field, field_extents, Conserve,
                            tidX, tidY, offset,
                            leftIdx, rightIdx, upIdx, downIdx);
        }

        // Ensure simulation step is fully carried out
        __syncthreads();
    }
}

__device__ void handle_boundary(glm::vec3* field, uint2 field_extents, BoundaryStrategy bs,
                                unsigned int tidX, unsigned int tidY, unsigned int offset,
                                unsigned int leftIdx, unsigned int rightIdx, unsigned int upIdx, unsigned int downIdx) {
    // Corners are average of ghost cell neighbours
    // TODO: See if this leads to race condition
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
