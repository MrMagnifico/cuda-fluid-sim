#include "gl_interop.cuh"

#include <device/utils.cuh>

template<typename T>
__global__ void copyFieldToTexture(T* field, cudaSurfaceObject_t texture_surface, uint2 texture_extents) {
    unsigned int tidX   = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int tidY   = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidX < texture_extents.x && tidY < texture_extents.y) { 
        unsigned int surfaceX   = tidX + 1U;
        unsigned int surfaceY   = texture_extents.y - (tidY + 1U); // OpenGL's y-axis grows upwards while CUDA's grows downwards, so flip our coordinates around 
        unsigned int offset     = surfaceX + surfaceY * (texture_extents.x + 2U);
        T value                 = field[offset];

        // Use float4 as the expected texture format is GL_RGBA32F
        float4 valuePadded = toRGBA(value);
        surf2Dwrite(valuePadded, texture_surface, tidX * sizeof(float4), tidY);
    }
}
