#ifndef _GL_INTEROP_CUH_
#define _GL_INTEROP_CUH_

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/vec3.hpp>
DISABLE_WARNINGS_POP()

__global__ void copyFieldToTexture(glm::vec3* field, cudaSurfaceObject_t texture_surface, uint2 texture_extents);

#endif
