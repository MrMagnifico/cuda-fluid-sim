#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/vec3.hpp>
#include <imgui/imgui.h>
DISABLE_WARNINGS_POP()

#include <framework/window.h>

#include <device/field_manager.cuh>
#include <device/field_edit.cuh>
#include <render/config.h>
#include <render/renderer.h>
#include <render/texture.h>
#include <ui/debug_callbacks.hpp>
#include <ui/menu.h>
#include <utils/constants.h>
#include <utils/cuda_utils.cuh>


int main(int argc, char* argv[]) {
    // Init core object(s)
    const uint2 fieldExtents = { utils::INITIAL_WIDTH, utils::INITIAL_HEIGHT };
    Window m_window(utils::WINDOW_TITLE, glm::ivec2(utils::INITIAL_WIDTH, utils::INITIAL_HEIGHT), OpenGLVersion::GL46);
    RenderConfig m_renderConfig;
    ui::Menu m_menu(m_renderConfig);
    TextureManager m_textureManager;
    m_textureManager.addTexture(utils::RESOURCES_DIR_PATH / "cursor.png");
    Renderer m_fieldRenderer(m_renderConfig, m_window, m_textureManager.getTexture("cursor.png"), fieldExtents.x, fieldExtents.y);
    FieldManager m_fieldManager(m_renderConfig, m_window,
                                fieldExtents,
                                m_fieldRenderer.getSourcesDensityTex(),     m_fieldRenderer.getDensitiesTex(),
                                m_fieldRenderer.getSourcesVelocityTex(),    m_fieldRenderer.getVelocitiesTex());


    // Set initial sources
    for (unsigned int i = 0U; i < 50U; i++) {
        for (unsigned int j = 0U; j < 50U; j++) {
            m_fieldManager.setSourceDensity(make_uint2(300U + i, 300U + j), glm::vec3(0.2f, 0.0f, 0.0f));
            m_fieldManager.setSourceDensity(make_uint2(600U + i, 400U + j), glm::vec3(0.0f, 0.2f, 0.0f));
            m_fieldManager.setSourceDensity(make_uint2(500U + i, 400U + j), glm::vec3(0.0f, 0.0f, 0.2f));
            m_fieldManager.setSourceVelocity(make_uint2(300U + i, 300U + j), glm::vec2(0.2f, 0.0f));
            m_fieldManager.setSourceVelocity(make_uint2(600U + i, 400U + j), glm::vec2(0.0f, 0.2f));
            m_fieldManager.setSourceVelocity(make_uint2(500U + i, 400U + j), glm::vec2(0.2f, 0.2f));
        }
    }
    CUDA_ERROR(cudaDeviceSynchronize());

    // Register functional UI callbacks
    // Lambda to bind relevant objects in callback's scope. Fuck me
    m_window.registerKeyCallback([&m_renderConfig](int key, int scancode, int action, int mods) { m_renderConfig.keyCallback(key, scancode, action, mods); });
    m_window.registerMouseMoveCallback([&m_fieldManager](glm::vec2 cursorPos) { m_fieldManager.mouseMoveCallback(cursorPos); });
    m_window.registerMouseButtonCallback([&m_fieldManager](int button, int action, int mods) { m_fieldManager.mouseButtonCallback(button, action, mods); });

    // Enable additive blending based on source (incoming) alpha to draw brush billboard
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Main loop
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    while (!m_window.shouldClose()) {
        double frameStart = glfwGetTime();

        // Main simulation
        m_fieldManager.simulate();

        // Write fields to OpenGL textures
        m_fieldManager.copyFieldsToTextures();

        // Process controls
        m_window.updateInput();

        // Clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Render fields and render brush if cursor is not over an ImGUI window
        m_fieldRenderer.renderFields();
        ImGuiIO io = ImGui::GetIO();
        if (!io.WantCaptureMouse) { m_fieldRenderer.renderBrush(); }

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
