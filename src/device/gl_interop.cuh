#ifndef _GL_INTEROP_CUH_
#define _GL_INTEROP_CUH_


template<typename T>
__global__ void copyFieldToTexture(T* field, cudaSurfaceObject_t texture_surface, uint2 texture_extents);


#endif
