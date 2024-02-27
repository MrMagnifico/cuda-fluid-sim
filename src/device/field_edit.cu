#include "field_edit.cuh"

DISABLE_WARNINGS_PUSH()
#include <glm/common.hpp>
DISABLE_WARNINGS_POP()


template<typename T>
__global__ void set_source(T* sources, uint2 coords, T val, uint2 field_extents) {
    unsigned int offset = coords.x + (coords.y * field_extents.x);
    sources[offset]     = val;
}

template<typename T>
__global__ void update_field(T* field, T value, uint2 field_extents, uint2 top_left, uint2 bottom_right,
                             UpdateType update_type, bool clampToZero) {
    // Map thread to field coordinates
    unsigned int threadIdX  = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int threadIdY  = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int updateX    = threadIdX + top_left.x;
    unsigned int updateY    = threadIdY + top_left.y;

    // Threads falling within selection box update their respective cell in the field
    if (updateX <= bottom_right.x && updateY <= bottom_right.y) {
        unsigned int verticalStride = field_extents.x + 2U;
        unsigned int offset         = (updateX + 1U) + (updateY + 1U) * verticalStride; // +1 accounts for ghost cells

        switch (update_type) {
            case Add:       { field[offset] += value; } break;
            case Remove:    {
                field[offset] -= value;
                if (clampToZero) { field[offset] = glm::max(field[offset], T(0.0f)); }
            } break;
        }
    }
}

template<typename T>
__global__ void copyOldField(T* old_field, T* new_field, uint2 old_field_extents, uint2 new_field_extents) {
    unsigned int threadIdX  = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int threadIdY  = threadIdx.y + blockIdx.y * blockDim.y;
    uint2 smallestExtents   = make_uint2(min(old_field_extents.x, new_field_extents.x),
                                         min(old_field_extents.y, new_field_extents.y));

    // Only copy over values if the thread is operating on an interior cell within the smaller of the two bounds
    if (1U <= threadIdX && threadIdX <= smallestExtents.x &&
        1U <= threadIdY && threadIdY <= smallestExtents.y) {
        unsigned int oldFieldOffset = threadIdX + threadIdY * (old_field_extents.x + 2U);
        unsigned int newFieldOffset = threadIdX + threadIdY * (new_field_extents.x + 2U);
        new_field[newFieldOffset]   = old_field[oldFieldOffset];
    }
}
