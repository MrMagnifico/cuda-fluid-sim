#include "fluid_sim.cuh"

__global__ void add_sources(float3* densities, float3* sources, float time_step, unsigned int num_cells) {
    unsigned int tidX   = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tidY   = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int offset = tidX + tidY * blockDim.x * gridDim.x;
    if (offset < num_cells) {
        densities[offset].x += time_step * sources[offset].x;
        densities[offset].y += time_step * sources[offset].y;
        densities[offset].z += time_step * sources[offset].z;
    }
}

__global__ void diffuse(float3* old_field, float3* new_field, uint2 field_extents, unsigned int num_cells,
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
        if ((0U < tidX && tidX <= field_extents.x) && (0U < tidY && tidY <= field_extents.y)) { // Thread is handling valid non-ghost cell
            // Gauss-Seidel relaxation steps
            new_field[offset].x = (old_field[offset].x + diffSpeed * (old_field[leftIdx].x + old_field[rightIdx].x + old_field[upIdx].x + old_field[downIdx].x)) / relaxationDenom;
            new_field[offset].y = (old_field[offset].y + diffSpeed * (old_field[leftIdx].y + old_field[rightIdx].y + old_field[upIdx].y + old_field[downIdx].y)) / relaxationDenom;
            new_field[offset].z = (old_field[offset].z + diffSpeed * (old_field[leftIdx].z + old_field[rightIdx].z + old_field[upIdx].z + old_field[downIdx].z)) / relaxationDenom;
        } else if (offset < num_cells) { // Thread is handling valid ghost cell
            handle_boundary(new_field, field_extents, Conserve,
                            tidX, tidY, offset,
                            leftIdx, rightIdx, upIdx, downIdx);
        }

        // Ensure simulation step is fully carried out
        __syncthreads();
    }
}

__device__ void handle_boundary(float3* field, uint2 field_extents, BoundaryStrategy bs,
                                unsigned int tidX, unsigned int tidY, unsigned int offset,
                                unsigned int leftIdx, unsigned int rightIdx, unsigned int upIdx, unsigned int downIdx) {
    // Corners are average of ghost cell neighbours
    // TODO: See if this leads to race condition
    if (tidX == 0U && tidY == 0U)                                              { field[offset].x = (0.5f * field[rightIdx].x) + (0.5f * field[downIdx].x);      // Top-left
                                                                                 field[offset].y = (0.5f * field[rightIdx].y) + (0.5f * field[downIdx].y);
                                                                                 field[offset].z = (0.5f * field[rightIdx].z) + (0.5f * field[downIdx].z); }    
    else if (tidX == (field_extents.x + 1U) && tidY == 0U)                     { field[offset].x = (0.5f * field[leftIdx].x) + (0.5f * field[downIdx].x);       // Top-right
                                                                                 field[offset].y = (0.5f * field[leftIdx].y) + (0.5f * field[downIdx].y);
                                                                                 field[offset].z = (0.5f * field[leftIdx].z) + (0.5f * field[downIdx].z); }
    else if (tidX == 0U && tidY == (field_extents.y + 1U))                     { field[offset].x = (0.5f * field[rightIdx].x) + (0.5f * field[upIdx].x);        // Bottom-left
                                                                                 field[offset].y = (0.5f * field[rightIdx].y) + (0.5f * field[upIdx].y);
                                                                                 field[offset].z = (0.5f * field[rightIdx].z) + (0.5f * field[upIdx].z); }
    else if (tidX == (field_extents.x + 1U) && tidY == (field_extents.y + 1U)) { field[offset].x = (0.5f * field[leftIdx].x) + (0.5f * field[upIdx].x);         // Bottom-right
                                                                                 field[offset].y = (0.5f * field[leftIdx].y) + (0.5f * field[upIdx].y);
                                                                                 field[offset].z = (0.5f * field[leftIdx].z) + (0.5f * field[upIdx].z); }
    // Edge values depend on chosen strategy
    else if (tidX == 0U)                     { field[offset].x = bs == ReverseHorizontal    ? -field[rightIdx].x    : field[rightIdx].x;    // Left edge
                                               field[offset].y = bs == ReverseHorizontal    ? -field[rightIdx].y    : field[rightIdx].y;
                                               field[offset].z = bs == ReverseHorizontal    ? -field[rightIdx].z    : field[rightIdx].z; }
    else if (tidX == (field_extents.x + 1U)) { field[offset].x = bs == ReverseHorizontal    ? -field[leftIdx].x     : field[leftIdx].x;     // Right edge
                                               field[offset].y = bs == ReverseHorizontal    ? -field[leftIdx].y     : field[leftIdx].y;
                                               field[offset].z = bs == ReverseHorizontal    ? -field[leftIdx].z     : field[leftIdx].z; }
    else if (tidY == 0U)                     { field[offset].x = bs == ReverseVertical      ? -field[downIdx].x     : field[downIdx].x;     // Top edge
                                               field[offset].y = bs == ReverseVertical      ? -field[downIdx].y     : field[downIdx].y;
                                               field[offset].z = bs == ReverseVertical      ? -field[downIdx].z     : field[downIdx].z; }
    else if (tidY == (field_extents.y + 1U)) { field[offset].x = bs == ReverseVertical      ? -field[upIdx].x       : field[upIdx].x;       // Bottom edge
                                               field[offset].y = bs == ReverseVertical      ? -field[upIdx].y       : field[upIdx].y;
                                               field[offset].z = bs == ReverseVertical      ? -field[upIdx].z       : field[upIdx].z; }
}
