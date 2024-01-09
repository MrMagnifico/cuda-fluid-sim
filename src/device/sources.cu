#include "sources.cuh"

template<typename T>
__global__ void set_source(T* sources, uint2 coords, T val, uint2 field_extents) {
    unsigned int offset = coords.x + (coords.y * field_extents.x);
    sources[offset]     = val;
}
