#include "config.h"

DISABLE_WARNINGS_PUSH()
#include <GLFW/glfw3.h>
DISABLE_WARNINGS_POP()

#include <utils/constants.h>
#include <utils/magic_enum.hpp>

#include <cstdint>

void RenderConfig::keyCallback(int key, int scancode, int action, int mods) {
    if ((key == GLFW_KEY_LEFT_BRACKET || key == GLFW_KEY_RIGHT_BRACKET) && // Change brush size with bracket keys
        (action == GLFW_PRESS || action == GLFW_REPEAT)) {
        float sizeChange    = key == GLFW_KEY_RIGHT_BRACKET ? utils::BRUSH_SIZE_CHANGE_INCREMENT : -utils::BRUSH_SIZE_CHANGE_INCREMENT;
        brushParams.scale   += sizeChange;
        brushParams.scale   = std::clamp(brushParams.scale, utils::MIN_BRUSH_SIZE, utils::MAX_BRUSH_SIZE);
    } else if (key == GLFW_KEY_TAB && (action == GLFW_PRESS || action == GLFW_REPEAT)) { // Use tab to cycle between drawing modes
        size_t rawValue             = magic_enum::enum_integer(brushParams.brushEditMode) + 1;
        rawValue                    = rawValue % magic_enum::enum_count<BrushEditMode>();
        brushParams.brushEditMode   = magic_enum::enum_value<BrushEditMode>(rawValue); 
    }
}
