#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glm/vec3.hpp>
DISABLE_WARNINGS_POP()

#include <filesystem>
#include <string>

namespace utils {
    // CUDA parameters
    #ifdef __CUDACC__
    const dim3 BLOCK_SIZE               = { 4U, 8U, 1U };   // Number of threads per block
    constexpr size_t FIELDS_PER_TYPE    = 3UL;              // Number of fields for each type of field (density, velocity): {current, previous, sources}
    #endif

    // OpenGL
    constexpr size_t NUM_TEXTURES = 4UL; // Total number of OpenGL textures being managed

    // Viewport parameters
    constexpr int32_t INITIAL_WIDTH     = 1918;
    constexpr int32_t INITIAL_HEIGHT    = 1040;

    // Shitty strings.xml
    constexpr std::string_view WINDOW_TITLE = "CUDA Fluid Solver";

    // Brush parameters
    constexpr float MIN_BRUSH_SIZE              = 0.03f;
    constexpr float MAX_BRUSH_SIZE              = 0.5f;
    constexpr float BRUSH_SIZE_CHANGE_INCREMENT = 0.01f;

    // Resource directories 
    const std::filesystem::path SHADERS_DIR_PATH    = SHADERS_DIR;
    const std::filesystem::path RESOURCES_DIR_PATH  = RESOURCES_DIR;
}

#endif
