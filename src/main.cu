#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui/imgui.h>
DISABLE_WARNINGS_POP()

#include <framework/shader.h>
#include <framework/window.h>

#include <ui/callbacks.hpp>
#include <ui/menu.h>
#include <utils/constants.h>

#include <functional>
#include <iostream>


int main(int argc, char* argv[]) {
    // Init core object(s)
    Window m_window("CUDA Fluid Solver", glm::ivec2(utils::INITIAL_WIDTH, utils::INITIAL_HEIGHT), OpenGLVersion::GL46);
    ui::Menu m_menu;

    // Register UI callbacks
    m_window.registerKeyCallback(ui::keyCallback);
    m_window.registerMouseMoveCallback(ui::onMouseMove);
    m_window.registerMouseButtonCallback(ui::mouseButtonCallback);

    // Main loop
    while (!m_window.shouldClose()) {
        // This is your game loop
        // Put your real-time logic and rendering in here
        m_window.updateInput();

        // Clear the screen
        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Controls and UI
        ImGuiIO io = ImGui::GetIO();
        m_menu.draw();
        if (!io.WantCaptureMouse) { /* Non ImGUI UI code */ }

        // Processes input and swaps the window buffer
        m_window.swapBuffers();
    }

    return EXIT_SUCCESS;
}