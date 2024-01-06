#include "gl_interop.cuh"

__global__ void copyFieldToTexture(glm::vec3* field, cudaSurfaceObject_t texture_surface, uint2 texture_extents) {
    unsigned int tidX   = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tidY   = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidX < texture_extents.x && tidY < texture_extents.y) { 
        unsigned int surfaceX   = tidX + 1U;
        unsigned int surfaceY   = tidY + 1U;
        unsigned int offset     = surfaceX + surfaceY * blockDim.x * gridDim.x;
        glm::vec3 value         = field[offset];

        // Use float4 as the expected texture format is GL_RGBA32F
        float4 valuePadded = make_float4(value.x, value.y, value.z, 1.0f);
        surf2Dwrite(valuePadded, texture_surface, tidX * sizeof(float4), tidY);
    }
}
