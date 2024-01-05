#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/vec3.hpp>
DISABLE_WARNINGS_POP()
#include <filesystem>

namespace utils {
    // Viewport parameters
    constexpr int32_t INITIAL_WIDTH     = 1920;
    constexpr int32_t INITIAL_HEIGHT    = 1040;

    // Resource directories 
    const std::filesystem::path SHADERS_DIR_PATH = SHADERS_DIR;
}

#endif
