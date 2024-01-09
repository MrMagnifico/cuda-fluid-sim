#include "cuda_utils.cuh"


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
