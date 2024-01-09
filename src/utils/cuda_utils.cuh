#ifndef _CUDA_UTILS_CUH_
#define _CUDA_UTILS_CUH_

#include <cstdio>

#define CUDA_ERROR(err) (utils::HandleError(err, __FILE__, __LINE__))

namespace utils {
    /**
     * Print error message and exit program if err is not cudaSuccess
     * @param err Return code of host function call
     * @param file Name of the file in which the call was mode
     * @param line Code line where error occurred
    */
    inline void HandleError(cudaError_t err, const char* file, int line) {
        if (err != cudaSuccess) {
            fprintf(stderr, "\n%s in %s at line %d\n", cudaGetErrorString(err), file, line);
            exit(EXIT_FAILURE);
        }
    }

    /**
     * Create a surface object handle for usage in a CUDA kernel from an graphics resource descriptor
     * @param textureResource Handle for an OpenGL texture resources
     * @return Surface object handle
    */
    cudaSurfaceObject_t createSurfaceFromTextureResource(cudaGraphicsResource_t textureResource);
}

#endif
