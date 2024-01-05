#ifndef _CALLBACKS_HPP_
#define _CALLBACKS_HPP_

DISABLE_WARNINGS_PUSH()
#include <GLFW/glfw3.h>
DISABLE_WARNINGS_POP()

#include <iostream>

namespace ui {
    // In here you can handle key presses
    // key - Integer that corresponds to numbers in https://www.glfw.org/docs/latest/group__keys.html
    // mods - Any modifier keys pressed, like shift or control
    void onKeyPressed(int key, int mods)
    {
        std::cout << "Key pressed: " << key << std::endl;
    }

    // In here you can handle key releases
    // key - Integer that corresponds to numbers in https://www.glfw.org/docs/latest/group__keys.html
    // mods - Any modifier keys pressed, like shift or control
    void onKeyReleased(int key, int mods)
    {
        std::cout << "Key released: " << key << std::endl;
    }

    // If the mouse is moved this function will be called with the x, y screen-coordinates of the mouse
    void onMouseMove(const glm::dvec2& cursorPos)
    {
        std::cout << "Mouse at position: " << cursorPos.x << " " << cursorPos.y << std::endl;
    }

    // If one of the mouse buttons is pressed this function will be called
    // button - Integer that corresponds to numbers in https://www.glfw.org/docs/latest/group__buttons.html
    // mods - Any modifier buttons pressed
    void onMouseClicked(int button, int mods)
    {
        std::cout << "Pressed mouse button: " << button << std::endl;
    }

    // If one of the mouse buttons is released this function will be called
    // button - Integer that corresponds to numbers in https://www.glfw.org/docs/latest/group__buttons.html
    // mods - Any modifier buttons pressed
    void onMouseReleased(int button, int mods)
    {
        std::cout << "Released mouse button: " << button << std::endl;
    }

    void keyCallback(int key, int scancode, int action, int mods) {
        if (action == GLFW_PRESS) { onKeyPressed(key, mods); }
        else if (action == GLFW_RELEASE) { onKeyReleased(key, mods); }
    }

    void mouseButtonCallback(int button, int action, int mods) {
        if (action == GLFW_PRESS) { onMouseClicked(button, mods); }
        else if (action == GLFW_RELEASE) { onMouseReleased(button, mods); }
    }
}

#endif
