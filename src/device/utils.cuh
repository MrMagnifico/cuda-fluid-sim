#ifndef _UTILS_CUH_
#define _UTILS_CUH_


#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
DISABLE_WARNINGS_POP()


// Indicates if a cell is in the interior of a 2D block or on one of the edges/corners
enum CellLocationType2d { Interior = 0,
                          Top, Bottom, Left, Right,
                          TopLeft, TopRight, BottomLeft, BottomRight };


__device__ CellLocationType2d determineLocationType() {
    // Corners
    if      (threadIdx.x == 0U && threadIdx.y == 0U)                                { return TopLeft; }    
    else if (threadIdx.x == (blockDim.x - 1U) && threadIdx.y == 0U)                 { return TopRight; }
    else if (threadIdx.x == 0U && threadIdx.y == (blockDim.y - 1U))                 { return BottomLeft; }
    else if (threadIdx.x == (blockDim.x - 1U) && threadIdx.y == (blockDim.y - 1U))  { return BottomRight; }
    // Edges
    else if (threadIdx.x == 0U)                 { return Left; }
    else if (threadIdx.x == (blockDim.x - 1U))  { return Right; }
    else if (threadIdx.y == 0U)                 { return Top; }
    else if (threadIdx.y == (blockDim.y - 1U))  { return Bottom; }
    // If it's neither a corner or edge, it's an interior
    return Interior;
}

__device__ float4 toRGBA(glm::vec2 val) { return make_float4(val.x, val.y, 0.0f, 1.0f); }

__device__ float4 toRGBA(glm::vec3 val) { return make_float4(val.x, val.y, val.z, 1.0f); }

__device__ float4 toRGBA(glm::vec4 val) { return make_float4(val.x, val.y, val.z, 1.0f); } // Assume that the w-axis coordinate is 

__device__ glm::vec2 toGLM(float2 data) { return glm::vec2(data.x, data.y); }

__device__ glm::vec3 toGLM(float3 data) { return glm::vec3(data.x, data.y, data.z); }

__device__ glm::vec4 toGLM(float4 data) { return glm::vec4(data.x, data.y, data.z, data.w); }


#endif
