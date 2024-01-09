#include <framework/disable_all_warnings.h>
DISABLE_WARNINGS_PUSH()
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/vec3.hpp>
#include <imgui/imgui.h>
DISABLE_WARNINGS_POP()

#include <framework/shader.h>
#include <framework/window.h>

#include <device/field_manager.cuh>
#include <device/fluid_sim.cuh>
#include <device/gl_interop.cuh>
#include <device/sources.cuh>
#include <render/config.h>
#include <ui/callbacks.hpp>
#include <ui/menu.h>
#include <utils/constants.h>
#include <utils/cuda_utils.cuh>
#include <utils/render_utils.hpp>

#include <cuda_gl_interop.h>
#include <iostream>


int main(int argc, char* argv[]) {
    // Init core object(s)
    const uint2 fieldExtents = { utils::INITIAL_WIDTH, utils::INITIAL_HEIGHT };
    Window m_window(utils::WINDOW_TITLE, glm::ivec2(utils::INITIAL_WIDTH, utils::INITIAL_HEIGHT), OpenGLVersion::GL46);
    RenderConfig m_renderConfig;
    ui::Menu m_menu(m_renderConfig);
    FieldManager m_fieldManager(fieldExtents, m_renderConfig);
    
    // Set sources
    for (unsigned int i = 0U; i < 50U; i++) {
        for (unsigned int j = 0U; j < 50U; j++) {
            m_fieldManager.setSource(make_uint2(300U + i, 300U + j), glm::vec3(0.2f, 0.0f, 0.0f));
            m_fieldManager.setSource(make_uint2(600U + i, 400U + j), glm::vec3(0.0f, 0.2f, 0.0f));
            m_fieldManager.setSource(make_uint2(500U + i, 400U + j), glm::vec3(0.0f, 0.0f, 0.2f));
        }
    }
    CUDA_ERROR(cudaDeviceSynchronize());

    // Compile quad render w/ HDR shader program
    ShaderBuilder quadHdrBuilder;
    quadHdrBuilder.addStage(GL_VERTEX_SHADER, utils::SHADERS_DIR_PATH / "screen-quad.vert");
    quadHdrBuilder.addStage(GL_FRAGMENT_SHADER, utils::SHADERS_DIR_PATH / "hdr.frag");
    Shader quadHdr = quadHdrBuilder.build();

    // Init OpenGL textures and bind them to CUDA resources
    GLuint densitiesTex, sourcesTex;
    cudaGraphicsResource_t densitiesResource, sourcesResource;
    glCreateTextures(GL_TEXTURE_2D, 1, &densitiesTex);
    glTextureStorage2D(densitiesTex, 1, GL_RGBA32F, fieldExtents.x, fieldExtents.y);
    glTextureParameteri(densitiesTex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(densitiesTex, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, densitiesTex);
    CUDA_ERROR(cudaGraphicsGLRegisterImage(&densitiesResource, densitiesTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
    glCreateTextures(GL_TEXTURE_2D, 1, &sourcesTex);
    glTextureStorage2D(sourcesTex, 1, GL_RGBA32F, fieldExtents.x, fieldExtents.y);
    glTextureParameteri(sourcesTex, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTextureParameteri(sourcesTex, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, sourcesTex);
    CUDA_ERROR(cudaGraphicsGLRegisterImage(&sourcesResource, sourcesTex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

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
        cudaSurfaceObject_t densitiesSurface    = utils::createSurfaceFromTextureResource(densitiesResource);
        cudaSurfaceObject_t sourcesSurface      = utils::createSurfaceFromTextureResource(sourcesResource);
        m_fieldManager.copyFieldsToTexture(sourcesSurface, densitiesSurface);
        CUDA_ERROR(cudaDeviceSynchronize()); // Ensure that copying is over before terminating resource handles
        CUDA_ERROR(cudaDestroySurfaceObject(densitiesSurface));
        CUDA_ERROR(cudaDestroySurfaceObject(sourcesSurface));
        CUDA_ERROR(cudaGraphicsUnmapResources(1, &densitiesResource));
        CUDA_ERROR(cudaGraphicsUnmapResources(1, &sourcesResource));

        // Clear the screen
        glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Process controls
        ImGuiIO io = ImGui::GetIO();
        m_window.updateInput();
        if (!io.WantCaptureMouse) { /* Non ImGUI UI code */ }

        // Bind render shader, set uniforms, and render
        quadHdr.bind();
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, densitiesTex);
        glUniform1i(0, 0);
        glUniform1i(1, m_renderConfig.renderDensities);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, sourcesTex);
        glUniform1i(2, 1);
        glUniform1i(3, m_renderConfig.renderSources);
        glUniform1i(4, m_renderConfig.enableHdr);
        glUniform1f(5, m_renderConfig.exposure);
        glUniform1f(6, m_renderConfig.gamma);
        utils::renderQuad();

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
