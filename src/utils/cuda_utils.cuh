#ifndef _CUDA_UTILS_CUH_
#define _CUDA_UTILS_CUH_

#include <cuda_gl_interop.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_ERROR(err) (utils::HandleError(err, __FILE__, __LINE__))

namespace utils {
    inline void HandleError(cudaError_t err, const char* file, int line) {
        if (err != cudaSuccess) {
            fprintf(stderr, "\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
            exit(EXIT_FAILURE);
        }
    }

    cudaSurfaceObject_t createSurfaceFromTextureResource(cudaGraphicsResource_t textureResource) {
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
}

#endif
