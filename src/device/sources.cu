#include "sources.cuh"

__global__ void set_source(glm::vec3* sources, uint2 coords, glm::vec3 val, uint2 field_extents) {
    unsigned int offset = coords.x + (coords.y * field_extents.x);
    sources[offset]     = val;
}
