#include "sources.cuh"

__global__ void set_source(float3* sources, uint2 coords, float3 val, uint2 field_extents) {
    unsigned int offset = coords.x + (coords.y * field_extents.x);
    sources[offset]     = val;
}
