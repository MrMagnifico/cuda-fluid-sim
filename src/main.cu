#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/vec3.hpp>
#include <imgui/imgui.h>
DISABLE_WARNINGS_POP()

#include <framework/window.h>

#include <device/field_manager.cuh>
#include <device/fluid_sim.cuh>
#include <render/config.h>
#include <render/field_renderer.h>
#include <ui/callbacks.hpp>
#include <ui/menu.h>
#include <utils/constants.h>
#include <utils/cuda_utils.cuh>


int main(int argc, char* argv[]) {
    // Init core object(s)
    const uint2 fieldExtents = { utils::INITIAL_WIDTH, utils::INITIAL_HEIGHT };
    Window m_window(utils::WINDOW_TITLE, glm::ivec2(utils::INITIAL_WIDTH, utils::INITIAL_HEIGHT), OpenGLVersion::GL46);
    RenderConfig m_renderConfig;
    ui::Menu m_menu(m_renderConfig);
    FieldRenderer m_fieldRenderer(m_renderConfig, fieldExtents.x, fieldExtents.y);
    FieldManager m_fieldManager(m_renderConfig, fieldExtents,
                                m_fieldRenderer.getSourcesTex(), m_fieldRenderer.getDensitiesTex());
    
    // Set sources
    for (unsigned int i = 0U; i < 50U; i++) {
        for (unsigned int j = 0U; j < 50U; j++) {
            m_fieldManager.setSource(make_uint2(300U + i, 300U + j), glm::vec3(0.2f, 0.0f, 0.0f));
            m_fieldManager.setSource(make_uint2(600U + i, 400U + j), glm::vec3(0.0f, 0.2f, 0.0f));
            m_fieldManager.setSource(make_uint2(500U + i, 400U + j), glm::vec3(0.0f, 0.0f, 0.2f));
        }
    }
    CUDA_ERROR(cudaDeviceSynchronize());

    // Register UI callbacks
    m_window.registerKeyCallback(ui::keyCallback);
    m_window.registerMouseMoveCallback(ui::onMouseMove);
    m_window.registerMouseButtonCallback(ui::mouseButtonCallback);

    // Main loop
    while (!m_window.shouldClose()) {
        double frameStart = glfwGetTime();

        // Main simulation
        m_fieldManager.simulate();

        // Write fields to OpenGL textures
        m_fieldManager.copyFieldsToTextures();

        // Clear the screen
        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Process controls
        ImGuiIO io = ImGui::GetIO();
        m_window.updateInput();
        if (!io.WantCaptureMouse) { /* Non ImGUI UI code */ }

        // Bind render shader, set uniforms, and render
        m_fieldRenderer.render();

        // Draw UI
        m_menu.draw();

        // Processes input and swaps the window buffer
        m_window.swapBuffers();

        // Display FPS in window title
        double frameEnd     = glfwGetTime();
        double frameTime    = frameEnd - frameStart; 
        std::string fpsStr  = std::format("{} - {:.1f}FPS", utils::WINDOW_TITLE, 1.0 / frameTime);
        glfwSetWindowTitle(m_window.getGlfwWindow(), fpsStr.c_str());
    }

    return EXIT_SUCCESS;
}
