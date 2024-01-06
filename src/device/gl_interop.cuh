#ifndef _GL_INTEROP_CUH_
#define _GL_INTEROP_CUH_

__global__ void copyFieldToTexture(float3* field, cudaSurfaceObject_t texture_surface, uint2 texture_extents);

#endif
