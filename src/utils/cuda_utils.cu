#include "cuda_utils.cuh"

#include <cstdio>

inline void utils::HandleError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

cudaSurfaceObject_t utils::createSurfaceFromTextureResource(cudaGraphicsResource_t textureResource) {
    CUDA_ERROR(cudaGraphicsMapResources(1, &textureResource));
    cudaArray_t textureArray;
    CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&textureArray, textureResource, 0, 0));
    cudaResourceDesc textureResourceDescriptor;
    textureResourceDescriptor.resType           = cudaResourceTypeArray;
    textureResourceDescriptor.res.array.array   = textureArray;
    cudaSurfaceObject_t textureSurface;
    CUDA_ERROR(cudaCreateSurfaceObject(&textureSurface, &textureResourceDescriptor));
    return textureSurface;
}
