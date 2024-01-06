#ifndef _SOURCES_CUH_
#define _SOURCES_CUH_

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/vec3.hpp>
DISABLE_WARNINGS_POP()

__global__ void set_source(glm::vec3* sources, uint2 coords, glm::vec3 val, uint2 field_extents);

#endif
