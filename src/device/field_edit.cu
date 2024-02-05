#include "field_edit.cuh"

#include <utils/misc_utils.hpp>

template<typename T>
__global__ void set_source(T* sources, uint2 coords, T val, uint2 field_extents) {
    unsigned int offset = coords.x + (coords.y * field_extents.x);
    sources[offset]     = val;
}

template<typename T>
__global__ void update_field(T* field, T value, uint2 field_extents, uint2 top_left, uint2 bottom_right) {
    // Map thread to field coordinates
    unsigned int threadIdX  = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int threadIdY  = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int updateX    = threadIdX + top_left.x;
    unsigned int updateY    = threadIdY + top_left.y;

    // Threads falling within selection box update their respective cell in the field
    if (updateX <= bottom_right.x && updateY <= bottom_right.y) {
        unsigned int verticalStride = field_extents.x + 2U;
        unsigned int offset         = (updateX + 1U) + (updateY + 1U) * verticalStride; // +1 accounts for ghost cells
        field[offset]               += value;
    }
}
