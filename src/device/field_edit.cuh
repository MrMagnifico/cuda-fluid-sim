#ifndef _FIELD_EDIT_CUH_
#define _FIELD_EDIT_CUH_

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
DISABLE_WARNINGS_POP()

template<typename T>
__global__ void set_source(T* sources, uint2 coords, T val, uint2 field_extents);

template<typename T>
__global__ void update_field(T* field, T value, uint2 field_extents, uint2 top_left, uint2 bottom_right);

#endif
